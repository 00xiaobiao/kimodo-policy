"""Opt-in, in-memory visual domain randomization for training RGB frames."""

import math
import random
from collections.abc import Mapping

import torch


def build_domain_randomization(config: Mapping | None):
    """No configuration means no augmentation; enabled transforms are explicit."""
    if config is None:
        return None
    if not isinstance(config, Mapping):
        raise ValueError("main.domain_randomization must be a mapping")
    enabled = config.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("main.domain_randomization.enabled must be true or false")
    if not enabled:
        return None
    unknown = set(config) - {"enabled", "color_jitter", "view_crop"}
    if unknown:
        raise ValueError(f"Unknown main.domain_randomization keys: {sorted(unknown)}")
    transforms = []
    # Crop before color jitter. Each transform has its own RNG, so adding or
    # disabling cropping does not change the color factors or sample ordering.
    if "view_crop" in config:
        crop_config = config["view_crop"]
        if not isinstance(crop_config, Mapping):
            raise ValueError("main.domain_randomization.view_crop must be a mapping")
        crop_enabled = crop_config.get("enabled", False)
        if not isinstance(crop_enabled, bool):
            raise ValueError("main.domain_randomization.view_crop.enabled must be true or false")
        if crop_enabled:
            transforms.append(ViewCropAugmentation(crop_config))
    if "color_jitter" in config:
        color_config = config["color_jitter"]
        if not isinstance(color_config, Mapping):
            raise ValueError("main.domain_randomization.color_jitter must be a mapping")
        transforms.append(ColorJitterAugmentation(color_config))
    if not transforms:
        raise ValueError(
            "main.domain_randomization.enabled=true requires color_jitter "
            "or an enabled view_crop"
        )
    if len(transforms) == 1:
        return transforms[0]
    return ImageDomainRandomization(transforms)


class ImageDomainRandomization:
    def __init__(self, transforms) -> None:
        self.transforms = tuple(transforms)

    def __call__(self, image: torch.Tensor, *, seed: int, index: int) -> torch.Tensor:
        for transform in self.transforms:
            image = transform(image, seed=seed, index=index)
        return image


class ViewCropAugmentation:
    """Random crop with the original aspect ratio, resized to the original size.

    Opt in with view_crop.enabled=true and explicit probability/min_scale.
    min_scale is a SIDE-length fraction (not area); area is sampled uniformly
    in [min_scale**2, 1], as in Psi0's RandomViewPerturb. Integer rounding may
    change the aspect ratio by a pixel. No flipping, rotation, or padding.
    """

    def __init__(self, config: Mapping) -> None:
        required = {"probability", "min_scale"}
        missing = required - set(config)
        unknown = set(config) - required - {"enabled"}
        if missing or unknown:
            raise ValueError(
                "main.domain_randomization.view_crop requires explicit parameters; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        for name in sorted(required):
            try:
                if isinstance(config[name], bool):
                    raise ValueError("boolean is not a crop parameter")
                value = float(config[name])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"main.domain_randomization.view_crop.{name} must be a number"
                ) from error
            lower_valid = value > 0 if name == "min_scale" else value >= 0
            if not math.isfinite(value) or not lower_valid or value > 1:
                interval = "(0, 1]" if name == "min_scale" else "[0, 1]"
                raise ValueError(
                    f"main.domain_randomization.view_crop.{name} must be finite and in {interval}"
                )
            setattr(self, name, value)

    def __call__(self, image: torch.Tensor, *, seed: int, index: int) -> torch.Tensor:
        rng = random.Random(f"view-crop:{int(seed)}:{int(index)}")
        if rng.random() >= self.probability or self.min_scale == 1.0:
            return image
        if image.dtype != torch.uint8 or image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("View crop expects a uint8 RGB CHW frame")
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as TF

        height, width = image.shape[-2:]
        side_scale = math.sqrt(rng.uniform(self.min_scale**2, 1.0))
        crop_height = max(1, round(height * side_scale))
        crop_width = max(1, round(width * side_scale))
        top = rng.randint(0, height - crop_height)
        left = rng.randint(0, width - crop_width)
        if crop_height == height and crop_width == width:
            return image
        cropped = TF.resized_crop(
            TF.to_pil_image(image), top, left, crop_height, crop_width,
            [height, width], interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        return TF.pil_to_tensor(cropped).contiguous()


class ColorJitterAugmentation:
    def __init__(self, config: Mapping) -> None:
        required = {"probability", "brightness", "contrast", "saturation", "hue"}
        missing, unknown = required - set(config), set(config) - required
        if missing or unknown:
            raise ValueError(
                "main.domain_randomization.color_jitter requires explicit parameters; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        for name in sorted(required):
            try:
                if isinstance(config[name], bool):
                    raise ValueError("boolean is not a strength")
                value = float(config[name])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"main.domain_randomization.color_jitter.{name} must be a number"
                ) from error
            upper = 0.5 if name == "hue" else 1.0 if name == "probability" else math.inf
            if not math.isfinite(value) or not 0 <= value <= upper:
                raise ValueError(
                    f"main.domain_randomization.color_jitter.{name} "
                    f"must be finite and in [0, {upper}]"
                )
            setattr(self, name, value)

    def __call__(self, image: torch.Tensor, *, seed: int, index: int) -> torch.Tensor:
        # A separate, process-independent RNG preserves the episode/window sampler
        # and reproduces the same augmentation after resuming or changing workers.
        rng = random.Random(f"color-jitter:{int(seed)}:{int(index)}")
        if rng.random() >= self.probability:
            return image
        if image.dtype != torch.uint8 or image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("Color jitter expects a uint8 RGB CHW frame")
        from torchvision.transforms import functional as TF

        operations = [
            (TF.adjust_brightness, rng.uniform(max(0.0, 1 - self.brightness), 1 + self.brightness)),
            (TF.adjust_contrast, rng.uniform(max(0.0, 1 - self.contrast), 1 + self.contrast)),
            (TF.adjust_saturation, rng.uniform(max(0.0, 1 - self.saturation), 1 + self.saturation)),
            (TF.adjust_hue, rng.uniform(-self.hue, self.hue)),
        ]
        rng.shuffle(operations)
        # PIL keeps the color transforms inexpensive in CPU data-loader workers.
        # Return the same uint8 layout/range expected by DINO's preprocessing.
        augmented = TF.to_pil_image(image)
        for operation, factor in operations:
            identity = 0.0 if operation is TF.adjust_hue else 1.0
            if factor != identity:
                augmented = operation(augmented, factor)
        return TF.pil_to_tensor(augmented).contiguous()

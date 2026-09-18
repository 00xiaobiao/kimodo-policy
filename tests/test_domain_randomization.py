import pickle
import random
import unittest
from unittest.mock import patch

import torch
from torchvision.transforms import functional as TF

from data.domain_randomization import (
    ColorJitterAugmentation,
    ViewCropAugmentation,
    build_domain_randomization,
)


class ViewCropTest(unittest.TestCase):
    def setUp(self):
        y = torch.arange(120, dtype=torch.uint8).view(120, 1).expand(120, 200)
        x = torch.arange(200, dtype=torch.uint8).view(1, 200).expand(120, 200)
        self.image = torch.stack((y, x, torch.full_like(x, 71)))
        self.color = dict(probability=0.8, brightness=0.2, contrast=0.2,
                          saturation=0.2, hue=0.02)

    @staticmethod
    def crop(**overrides):
        return dict(enabled=True, probability=1.0, min_scale=0.95, **overrides)

    def test_omitted_or_disabled_crop_preserves_existing_color_results(self):
        legacy = ColorJitterAugmentation(self.color)
        for crop in (None, {"enabled": False}, {},
                     {"enabled": True, "probability": 0.0, "min_scale": 0.95}):
            config = {"enabled": True, "color_jitter": self.color}
            if crop is not None:
                config["view_crop"] = crop
            augment = build_domain_randomization(config)
            for index in range(5):
                with self.subTest(crop=crop, index=index):
                    self.assertTrue(torch.equal(
                        augment(self.image, seed=42, index=index),
                        legacy(self.image, seed=42, index=index),
                    ))

    def test_crop_preserves_layout_and_geometry_without_mutating_input(self):
        original = self.image.clone()
        augment = ViewCropAugmentation(self.crop())
        with patch.object(TF, "resized_crop", wraps=TF.resized_crop) as crop:
            for index in range(12):
                result = augment(self.image, seed=42, index=index)
                self.assertEqual(result.shape, self.image.shape)
                self.assertEqual(result.dtype, torch.uint8)
                self.assertTrue(result.is_contiguous())
                # Coordinate ramps remain ordered; channels and constant areas
                # are preserved (no mirroring, rotation, noise, or padding).
                self.assertTrue(torch.all(torch.diff(result[0].float(), dim=0) >= 0))
                self.assertTrue(torch.all(torch.diff(result[1].float(), dim=1) >= 0))
                self.assertTrue(torch.all(result[2] == 71))
            self.assertGreater(crop.call_count, 0)
            for call in crop.call_args_list:
                _, top, left, height, width, size = call.args
                self.assertGreaterEqual(top, 0)
                self.assertGreaterEqual(left, 0)
                self.assertLessEqual(top + height, 120)
                self.assertLessEqual(left + width, 200)
                self.assertGreaterEqual(height, round(120 * 0.95))
                self.assertGreaterEqual(width, round(200 * 0.95))
                self.assertAlmostEqual(height / 120, width / 200, delta=1 / 120)
                self.assertEqual(size, [120, 200])
        self.assertTrue(torch.equal(original, self.image))

    def test_crop_only_and_color_composition_are_independent(self):
        crop = build_domain_randomization({"enabled": True, "view_crop": self.crop()})
        color = build_domain_randomization({"enabled": True, "color_jitter": self.color})
        both = build_domain_randomization({
            "enabled": True, "view_crop": self.crop(), "color_jitter": self.color,
        })
        for index in range(5):
            expected = color(crop(self.image, seed=42, index=index), seed=42, index=index)
            self.assertTrue(torch.equal(both(self.image, seed=42, index=index), expected))

    def test_noop_settings_keep_original_pixels(self):
        for config in (
            dict(probability=0.0, min_scale=0.95),
            dict(probability=1.0, min_scale=1.0),
        ):
            augment = ViewCropAugmentation(config)
            self.assertIs(augment(self.image, seed=42, index=0), self.image)
        self.assertIsNone(build_domain_randomization({
            "enabled": False, "view_crop": self.crop(), "color_jitter": self.color,
        }))

    def test_probability_can_skip_and_changes_across_samples(self):
        augment = ViewCropAugmentation(dict(probability=0.3, min_scale=0.95))
        frames = [augment(self.image, seed=42, index=i) for i in range(40)]
        changed = [not torch.equal(frame, self.image) for frame in frames]
        self.assertTrue(any(changed))
        self.assertFalse(all(changed))
        self.assertGreater(len({frame.numpy().tobytes() for frame in frames}), 2)

    def test_pickle_and_global_rng_independence(self):
        augment = build_domain_randomization({
            "enabled": True, "view_crop": self.crop(), "color_jitter": self.color,
        })
        python_state, torch_state = random.getstate(), torch.get_rng_state().clone()
        expected = augment(self.image, seed=42, index=17)
        restored = pickle.loads(pickle.dumps(augment))
        self.assertTrue(torch.equal(expected, restored(self.image, seed=42, index=17)))
        self.assertEqual(python_state, random.getstate())
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        self.assertFalse(torch.equal(expected, augment(self.image, seed=43, index=17)))

    def test_invalid_or_incomplete_crop_parameters_fail_early(self):
        cases = [
            {"enabled": True}, {"enabled": True, "probability": 0.3},
            {"enabled": True, "min_scale": 0.95}, {"enabled": "false"},
            {**self.crop(), "min_sclae": 0.95},
        ]
        for key, value in (
            ("min_scale", 0), ("min_scale", -0.1), ("min_scale", 1.01),
            ("min_scale", float("nan")), ("min_scale", True),
            ("probability", -0.1), ("probability", 1.01),
            ("probability", float("inf")), ("probability", None),
        ):
            cases.append({**self.crop(), key: value})
        for crop in cases:
            with self.subTest(crop=crop), self.assertRaisesRegex(ValueError, "view_crop"):
                build_domain_randomization({"enabled": True, "view_crop": crop})


if __name__ == "__main__":
    unittest.main()

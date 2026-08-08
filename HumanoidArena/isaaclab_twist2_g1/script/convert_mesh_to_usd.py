#!/usr/bin/env python3
"""Convert a mesh asset such as GLB/GLTF/OBJ/STL/FBX into USD with Isaac Sim."""

import argparse
import asyncio
import os
from pathlib import Path

from isaacsim import SimulationApp


SUPPORTED_EXTENSIONS = {".glb", ".gltf", ".obj", ".stl", ".fbx"}


async def convert_asset(input_path: str, output_path: str, load_materials: bool) -> bool:
    import omni.kit.asset_converter

    converter_context = omni.kit.asset_converter.AssetConverterContext()
    converter_context.ignore_materials = not load_materials
    converter_context.ignore_animations = True
    converter_context.ignore_camera = True
    converter_context.ignore_light = True
    converter_context.merge_all_meshes = True
    converter_context.use_meter_as_world_unit = True
    converter_context.baking_scales = True
    converter_context.use_double_precision_to_usd_transform_op = True

    instance = omni.kit.asset_converter.get_instance()
    task = instance.create_converter_task(input_path, output_path, None, converter_context)
    return await task.wait_until_finished()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert mesh assets to USD with Isaac Sim.")
    parser.add_argument("input", type=str, help="Input mesh file path")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output USD path. Defaults to the same path with a .usd suffix.",
    )
    parser.add_argument(
        "--no-materials",
        action="store_true",
        help="Skip importing materials from the source asset.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input asset not found: {input_path}")

    input_suffix = Path(input_path).suffix.lower()
    if input_suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported extension: {input_suffix}. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    output_path = os.path.abspath(args.output) if args.output else str(Path(input_path).with_suffix(".usd"))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    simulation_app = SimulationApp({"headless": True})
    try:
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("omni.kit.asset_converter")
        success = asyncio.get_event_loop().run_until_complete(
            convert_asset(input_path, output_path, load_materials=not args.no_materials)
        )
        if not success:
            raise RuntimeError(f"Conversion failed: {input_path} -> {output_path}")
        print(output_path)
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations
from pathlib import Path
import argparse
import asyncio
import yaml

from components import EventsFileReaderComponent, OpencvWindow, ImageSegmentator, ImageSegmentatorConfig, ImageSegmentatorComponent
from limbus.core.pipeline import Pipeline


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


async def main(config: dict) -> None:
    # load the segmentation model
    image_segmentator = ImageSegmentator(
        config=ImageSegmentatorConfig(
            model_path=Path(config["model"]["path"]),
            model_uri=config["model"]["uri"],
            model_checkpoint=config["model"]["checkpoint"],
            device=config["runtime"]["device"]
        )
    )

    # Create a pipeline
    pipeline = Pipeline()

    # Create components
    reader = EventsFileReaderComponent(
        "reader",
        events_file=config["data"]["events_file"],
        device=config["runtime"]["device"])
    pipeline.add_nodes([reader])

    # Add components to the pipeline

    for output in reader.outputs:
        # instantiate the components
        segmentator = ImageSegmentatorComponent(
            f"segmentator_{output.name}", model=image_segmentator)
        viz = OpencvWindow(f"viz_{output.name}")

        # connect the components
        output >> segmentator.inputs.image
        output >> viz.inputs.image
        segmentator.outputs.mask >> viz.inputs.mask

    # Run pipeline
    await pipeline.async_run()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
        help="Path to config file in YAML format.")
    args = parser.parse_args()

    config: dict = load_config(args.config)

    try:
        asyncio.run(main(config))
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    except asyncio.CancelledError:
        print("CancelledError")
    finally:
        print("Done")

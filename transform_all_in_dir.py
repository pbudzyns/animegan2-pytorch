import argparse
import json
import os
import pathlib
import subprocess

from PIL import Image
import torch

from model import Generator


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "images_path",
        type=str,
        help="Path to directory with images of faces.",
    )

    parser.add_argument(
        "output_path",
        type=str,
        help="Path to output directory.",
    )

    parser.add_argument(
        "--model",
        dest="model",
        default="weights/generator_Paprika.pt",
        type=str,
        help="Path to model checkpoint to use.",
    )

    parser.add_argument(
        "--device",
        dest="device",
        default="cuda",
        type=str,
        help="Device to run models on."
    )

    return parser.parse_args()


def main() -> None:
    """Do main."""
    args = _parse_args()

    model = Generator()
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.to(args.device).eval()

    face2paint = torch.hub.load(
        'AK391/animegan2-pytorch:main',
        'face2paint',
        size=512,
        device="cuda",
        side_by_side=False,
    )

    images_path = pathlib.Path(args.images_path)
    output_path = pathlib.Path(args.output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    for image_file in images_path.iterdir():
        image = Image.open(image_file).convert("RGB")
        output = face2paint(model, image)
        output.save(output_path / image_file.name)


if __name__ == "__main__":
    main()

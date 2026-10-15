"""Run inference with a trained YOLO checkpoint."""

import argparse

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to best.pt or another checkpoint")
    parser.add_argument("--source", required=True, help="Image, directory, video, or stream")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--project", default="runs/predict")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    YOLO(args.model).predict(
        source=args.source,
        conf=args.conf,
        save=True,
        project=args.project,
        name="predictions",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()


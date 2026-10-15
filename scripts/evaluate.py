"""Evaluate a trained YOLO checkpoint and print its detection metrics."""

import argparse
import json

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to best.pt or another checkpoint")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = YOLO(args.model).val(data=args.data, split=args.split, imgsz=args.imgsz)
    print(json.dumps(metrics.results_dict, indent=2))


if __name__ == "__main__":
    main()


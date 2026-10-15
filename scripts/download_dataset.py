"""Download the Roboflow Vehicles dataset in YOLO11 format."""

import os
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow


def main() -> None:
    load_dotenv()
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("Set ROBOFLOW_API_KEY in the environment or a .env file.")

    output_dir = Path("datasets")
    output_dir.mkdir(exist_ok=True)
    project = Roboflow(api_key=api_key).workspace("roboflow-100").project(
        "vehicles-q0x2v"
    )
    dataset = project.version(2).download("yolov11", location=str(output_dir / "vehicles-2"))
    print(f"Dataset downloaded to: {dataset.location}")


if __name__ == "__main__":
    main()


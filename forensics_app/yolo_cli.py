"""Command line for the long Set 9 jobs, so they can also run outside the GUI.

    .venv/bin/python -m forensics_app.yolo_cli convert "<lab>/blood_cell.v1i.coco"
    .venv/bin/python -m forensics_app.yolo_cli train --epochs 50
    .venv/bin/python -m forensics_app.yolo_cli evaluate --split test

The results are the same files the app uses (datasets/blood_cell_yolo and
models/yolo/blood_cell_yolo11n.pt), so a model trained here appears in the app.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from forensics_app.tools import yolo_dataset as yd
from forensics_app.tools import yolo_model as ym


def _print_progress(done: int, total: int, label: str) -> None:
    print(f"\r[{done}/{total}] {label}".ljust(100), end="", flush=True)


def convert(args: argparse.Namespace) -> int:
    names, summaries, yaml_path = yd.convert_coco_dataset(args.coco_root, args.output, _print_progress)
    print()
    print(f"Classes: {names}")
    for s in summaries:
        print(f"  {s.split:5s}: {s.images} images, {s.boxes} boxes {s.per_class} (skipped {s.skipped_boxes})")
        if s.missing_images or s.unlisted_images:
            print(f"         {s.missing_images} listed in the json but missing, "
                  f"{s.unlisted_images} not in the json (copied without labels)")
    print(f"Wrote {yaml_path}")
    return 0


def train(args: argparse.Namespace) -> int:
    settings = ym.TrainSettings(
        data_yaml=Path(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        start_from=args.start_from,
        patience=args.patience,
        device=args.device,
        name=args.name,
        workers=args.workers,
    )
    outcome = ym.train(settings, _print_progress)
    print()
    print(f"Run folder: {outcome.run_dir}")
    print(f"Best weights copied to: {outcome.installed_weights}")
    print(f"{outcome.epochs_done} epochs in {outcome.seconds / 60:.1f} min")
    return 0


def evaluate(args: argparse.Namespace) -> int:
    from forensics_app.tools.yolo_eval import evaluate_split

    model = ym.load_model(args.weights)
    report = evaluate_split(model, Path(args.data), args.split, _print_progress)
    print()
    result = report.result
    print(f"{report.images} images, mAP@0.5 = {result.map_at(0.5):.3f}, mAP@[.5:.95] = {result.map:.3f}")
    for name, ap50, ap in zip(result.class_names, result.ap_at(0.5), result.ap_per_class):
        print(f"  {name:5s} AP50 = {ap50:.3f}  AP50-95 = {ap:.3f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m forensics_app.yolo_cli", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("convert", help="COCO -> YOLO dataset")
    p.add_argument("coco_root", help="folder with train/valid/test and _annotations.coco.json")
    p.add_argument("--output", default=str(ym.BLOOD_CELL_DATASET))
    p.set_defaults(func=convert)

    p = sub.add_parser("train", help="train or fine-tune YOLO")
    p.add_argument("--data", default=str(ym.BLOOD_CELL_DATASET / "data.yaml"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--start-from", default="yolo11n.pt", help="yolo11n.pt = fine-tune, yolo11n.yaml = from scratch")
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--device", default=None, help="0 for the first GPU, cpu, or empty for automatic")
    p.add_argument("--name", default="blood_cell")
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(func=train)

    p = sub.add_parser("evaluate", help="our mAP / PR / FPPI on a split")
    p.add_argument("--weights", default=str(ym.BLOOD_CELL_WEIGHTS))
    p.add_argument("--data", default=str(ym.BLOOD_CELL_DATASET / "data.yaml"))
    p.add_argument("--split", default="test")
    p.set_defaults(func=evaluate)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError) as error:
        print(f"\nError: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

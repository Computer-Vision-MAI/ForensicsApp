"""Set 9.2: convert the blood-cell dataset from COCO to the YOLO format, and explore it.

COCO keeps all the boxes of a split in one ``_annotations.coco.json``, in pixels and as
``[x, y, w, h]`` from the top-left corner. YOLO (Ultralytics) wants this layout::

    data.yaml
    train/images/<name>.jpg   train/labels/<name>.txt
    valid/images/...          valid/labels/...
    test/images/...           test/labels/...

with one ``.txt`` per image and one line per box: ``class_id x_center y_center width
height``, all normalised to [0, 1] by the image size, and ``class_id`` from 0 to N-1.

Roboflow exports add a parent category (``1-2-3-SmhR``, id 0) that is only the
"supercategory" of PC/RBC/WBC and has no boxes. The course notebook handles it with
``category_id - 1``; here the real classes are found from the file itself, so the
conversion also works for COCO files without that extra category.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import json
from pathlib import Path
import shutil

import numpy as np

from .detection_metrics import Detections, coco_to_yolo_box, yolo_to_xyxy
from .dialogs import TaskCancelled

ANNOTATIONS_FILE = "_annotations.coco.json"
SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

Report = Callable[[int, int, str], None]


def _no_report(done: int, total: int, label: str) -> None:
    return None


@dataclass(frozen=True)
class CocoClasses:
    """Which COCO category ids are real classes, and their YOLO index (0..N-1)."""

    names: list[str]
    coco_to_yolo: dict[int, int]


def coco_classes(coco: dict) -> CocoClasses:
    """Keep the categories that are real classes, sorted by id, and number them from 0.

    A category is dropped when it is the ``supercategory`` of another one and no box
    uses it (the Roboflow parent class). Everything else is kept, even without boxes,
    so the class ids stay the same in every split.
    """
    categories = sorted(coco.get("categories", []), key=lambda c: c["id"])
    if not categories:
        raise ValueError("The COCO file has no categories.")
    used = {a["category_id"] for a in coco.get("annotations", [])}
    parents = {c.get("supercategory") for c in categories}
    real = [c for c in categories if c["id"] in used or c["name"] not in parents]
    return CocoClasses([c["name"] for c in real], {c["id"]: i for i, c in enumerate(real)})


@dataclass
class SplitSummary:
    split: str
    images: int = 0
    boxes: int = 0
    skipped_boxes: int = 0  # empty boxes (zero width or height) or unknown category
    missing_images: int = 0  # listed in the json but not on disk
    unlisted_images: int = 0  # on disk but not in the json: copied, without a label file
    per_class: dict[str, int] = field(default_factory=dict)


def label_lines(annotations: list[dict], width: float, height: float, classes: CocoClasses) -> tuple[list[str], int]:
    """YOLO lines of one image; also returns how many boxes had to be skipped."""
    lines: list[str] = []
    skipped = 0
    for ann in annotations:
        cls = classes.coco_to_yolo.get(ann["category_id"])
        if cls is None:
            skipped += 1
            continue
        xc, yc, w, h = coco_to_yolo_box(ann["bbox"], width, height)
        if w <= 0 or h <= 0:
            skipped += 1
            continue
        lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return lines, skipped


def coco_to_yolo(
    input_folder: str | Path,
    output_folder: str | Path,
    classes: CocoClasses | None = None,
    report: Report = _no_report,
) -> SplitSummary:
    """Convert one split, as asked in the lab.

    * creates ``images`` and ``labels`` in ``output_folder``;
    * copies **all** the image files of ``input_folder`` there;
    * reads ``input_folder/_annotations.coco.json``;
    * writes ``labels/{image_name}.txt`` with ``class_id x_center y_center width height``
      per box. Images listed in the json without boxes get an empty file (a "background"
      image for YOLO). Images that the json does not mention are copied too, but get no
      label file because nothing is known about them; they are counted in
      ``unlisted_images`` so they can be checked (YOLO would treat them as background).

    ``classes`` lets several splits share the same numbering; by default it is read
    from this split's json.
    """
    source, target = Path(input_folder), Path(output_folder)
    json_path = source / ANNOTATIONS_FILE
    if not json_path.is_file():
        raise FileNotFoundError(f"There is no {ANNOTATIONS_FILE} in {source}.")
    with json_path.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)
    classes = classes or coco_classes(coco)
    images_dir, labels_dir = target / "images", target / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    by_image: dict[int, list[dict]] = {}
    for ann in coco.get("annotations", []):
        by_image.setdefault(ann["image_id"], []).append(ann)

    summary = SplitSummary(source.name, per_class={name: 0 for name in classes.names})
    entries = coco.get("images", [])
    for index, info in enumerate(entries):
        file_name = Path(info["file_name"]).name
        image_path = source / file_name
        if not image_path.is_file():
            summary.missing_images += 1
            continue
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image type: {file_name}")
        shutil.copy2(image_path, images_dir / file_name)
        lines, skipped = label_lines(by_image.get(info["id"], []), info["width"], info["height"], classes)
        (labels_dir / f"{Path(file_name).stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        summary.images += 1
        summary.boxes += len(lines)
        summary.skipped_boxes += skipped
        for line in lines:
            summary.per_class[classes.names[int(line.split()[0])]] += 1
        if index % 50 == 0 or index == len(entries) - 1:
            report(index + 1, len(entries), f"{source.name}: {index + 1}/{len(entries)} images")

    # The lab asks to copy every image of the folder, also those the json does not list
    listed = {Path(info["file_name"]).name for info in entries}
    for image_path in list_images(source):
        if image_path.name not in listed:
            shutil.copy2(image_path, images_dir / image_path.name)
            summary.unlisted_images += 1
    return summary


def write_data_yaml(dataset_root: str | Path, names: list[str], splits: tuple[str, ...] = SPLITS) -> Path:
    """Write ``data.yaml`` with paths relative to the folder that contains it.

    There is no ``path:`` key on purpose: Ultralytics then uses the folder of the yaml
    file as the dataset root (a *relative* ``path:`` would be resolved against its own
    ``datasets_dir`` setting instead). So the dataset folder can be moved or copied to
    another computer and still works, from any working directory.
    """
    root = Path(dataset_root).resolve()
    keys = {"train": "train", "valid": "val", "test": "test"}
    lines = []
    for split in splits:
        lines.append(f"{keys.get(split, split)}: {split}/images")
    lines += ["", f"nc: {len(names)}", f"names: {json.dumps(names)}", ""]
    path = root / "data.yaml"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def convert_coco_dataset(
    coco_root: str | Path,
    yolo_root: str | Path,
    report: Report = _no_report,
    cancelled: Callable[[], bool] = lambda: False,
) -> tuple[list[str], list[SplitSummary], Path]:
    """Convert every split found in ``coco_root`` (train/valid/test) and write ``data.yaml``.

    The classes are read from the train split so all the splits share the same ids.
    """
    coco_root, yolo_root = Path(coco_root), Path(yolo_root)
    splits = [s for s in SPLITS if (coco_root / s / ANNOTATIONS_FILE).is_file()]
    if "train" not in splits:
        raise FileNotFoundError(
            f"{coco_root} must contain train/{ANNOTATIONS_FILE} (and ideally valid/ and test/)."
        )
    with (coco_root / "train" / ANNOTATIONS_FILE).open("r", encoding="utf-8") as handle:
        classes = coco_classes(json.load(handle))
    summaries = []
    for split in splits:
        if cancelled():
            raise TaskCancelled()
        summaries.append(coco_to_yolo(coco_root / split, yolo_root / split, classes, report))
    return classes.names, summaries, write_data_yaml(yolo_root, classes.names, tuple(splits))


# --------------------------------------------------------------------------- reading


def read_data_yaml(path: str | Path) -> dict:
    """Minimal reader for the ``data.yaml`` written above (or by Roboflow).

    PyYAML comes with Ultralytics, but this keeps the dataset tools usable without it.
    """
    try:
        import yaml  # type: ignore[import-untyped]

        with Path(path).open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except ImportError:
        data = {}
        for raw in Path(path).read_text(encoding="utf-8").splitlines():
            if ":" not in raw or raw.lstrip().startswith("#") or raw.startswith(" "):
                continue
            key, value = raw.split(":", 1)
            value = value.strip()
            if value.startswith("["):
                value = json.loads(value.replace("'", '"'))
            data[key.strip()] = value
    names = data.get("names", [])
    if isinstance(names, dict):  # {0: PC, 1: RBC, ...}
        names = [names[k] for k in sorted(names)]
    data["names"] = [str(n) for n in names]
    return data


def split_images_dir(data: dict, yaml_path: str | Path, split: str) -> Path:
    """Folder of images of ``split`` ('train', 'val'/'valid' or 'test') from a data.yaml."""
    key = "val" if split in ("val", "valid") else split
    if key not in data:
        raise KeyError(f"data.yaml has no '{key}' split.")
    folder = Path(str(data[key]))
    if not folder.is_absolute():
        base = Path(str(data["path"])) if data.get("path") else Path(yaml_path).parent
        if not base.is_absolute():
            base = Path(yaml_path).parent / base
        folder = (base / folder).resolve()
    return folder


def read_yolo_labels(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Classes (n,) and normalised ``xc yc w h`` boxes (n, 4) of one label file."""
    rows = []
    for number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        parts = raw.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"{Path(path).name}, line {number}: expected 5 values, got {len(parts)}.")
        rows.append([float(v) for v in parts])
    if not rows:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 4))
    array = np.asarray(rows)
    return array[:, 0].astype(np.int64), array[:, 1:]


def label_path_for(image_path: str | Path) -> Path:
    """``.../images/x.jpg`` -> ``.../labels/x.txt`` (the Ultralytics convention)."""
    image_path = Path(image_path)
    parts = list(image_path.parts)
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def ground_truth(image_path: str | Path, width: int, height: int) -> Detections | None:
    """The ground-truth boxes of an image in pixels, or None if it has no label file."""
    label = label_path_for(image_path)
    if not label.is_file():
        return None
    classes, boxes = read_yolo_labels(label)
    return Detections(yolo_to_xyxy(boxes, width, height), classes)


def list_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


@dataclass
class DatasetStats:
    """The exploration asked in the notebook, for one split of a YOLO dataset."""

    split: str
    images: int
    boxes: int
    per_class: np.ndarray  # boxes per class
    images_with_class: np.ndarray  # images that contain at least one box of the class
    boxes_per_image: np.ndarray  # (images,)
    sizes: np.ndarray  # (boxes, 2) normalised width, height
    box_classes: np.ndarray  # (boxes,)
    background_images: int


def dataset_stats(images_dir: str | Path, num_classes: int, split: str = "") -> DatasetStats:
    per_class = np.zeros(num_classes, dtype=np.int64)
    with_class = np.zeros(num_classes, dtype=np.int64)
    per_image, sizes, box_classes = [], [], []
    images = list_images(images_dir)
    for image in images:
        label = label_path_for(image)
        classes, boxes = read_yolo_labels(label) if label.is_file() else (np.zeros(0, int), np.zeros((0, 4)))
        if len(classes) and classes.max() >= num_classes:
            raise ValueError(f"{label.name} uses class {classes.max()} but the dataset has {num_classes} classes.")
        per_image.append(len(classes))
        counts = np.bincount(classes, minlength=num_classes)
        per_class += counts
        with_class += counts > 0
        sizes.append(boxes[:, 2:4])
        box_classes.append(classes)
    per_image_array = np.asarray(per_image, dtype=np.int64)
    return DatasetStats(
        split=split or Path(images_dir).parent.name,
        images=len(images),
        boxes=int(per_image_array.sum()),
        per_class=per_class,
        images_with_class=with_class,
        boxes_per_image=per_image_array,
        sizes=np.concatenate(sizes) if sizes else np.zeros((0, 2)),
        box_classes=np.concatenate(box_classes) if box_classes else np.zeros(0, dtype=np.int64),
        background_images=int(np.sum(per_image_array == 0)),
    )

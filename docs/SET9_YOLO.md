# Set 9: Object detection with YOLO

Lab 9 (18 Nov) asks for three things, plus the theory of lecture 9 (NMS, IoU, mAP,
YOLO family). Everything is in the sidebar section **Object Detection (YOLO)**.

| Lab item | Tool | Code |
|---|---|---|
| 9.1 Load a pretrained Ultralytics model and run it on an image | YOLO: Detect Objects (pretrained) | `tools/yolo_model.py`, `tools/yolo_tools.py` |
| 9.2 COCO → YOLO conversion (`coco_to_yolo`) and `data.yaml` | YOLO: COCO → YOLO Dataset | `tools/yolo_dataset.py` |
| 9.2 Train YOLO on the blood cells and save it | YOLO: Train Blood-Cell Detector | `tools/yolo_model.py::train` |
| 9.3 Load the trained model, test on an unseen image, GT and predicted boxes | YOLO: Blood Cells (GT vs Prediction) | `tools/yolo_tools.py::detect_cells` |
| Extra: evaluation from scratch | YOLO: Evaluate (mAP, PR, FPPI) | `tools/detection_metrics.py`, `tools/yolo_eval.py` |

## Setup

```
.venv/bin/pip install -r requirements-yolo.txt
```

Ultralytics is optional, like PyTorch for Set 7: the YOLO tools explain how to install
it if it is missing, and the tests use a fake model, so CI needs neither.

Training needs a GPU to be practical: about 10 min per epoch on CPU and about 40-45 s
on a laptop RTX 5070 (PyTorch with CUDA, see `requirements-yolo.txt`).

## 9.1 Pretrained detector

* `YOLO("yolo11n.pt")` (2.6 M parameters, trained on COCO, 80 classes). The weights are
  downloaded to `models/yolo/` and never to the folder where the app was started.
* The dialog sets the confidence threshold, the NMS IoU threshold, the input size and
  how duplicate boxes are removed:
  1. Ultralytics NMS (default);
  2. **our own greedy NMS** (`detection_metrics.nms` / `batched_nms`): the library NMS
     is practically switched off (IoU 0.99) and the candidates are drawn before and
     after our NMS. On `bus.jpg` 35 candidates become the same 5 boxes as Ultralytics;
  3. **no NMS at all** with YOLO26 (`nms=False`): its one-to-one head was trained to
     give one box per object (lecture 9: "YOLO26, NMS-free end-to-end inference").
* On the slides' bus image the result matches the slide: bus 0.94, person 0.89, 0.88,
  0.86, 0.62.
* **Why are people blue?** Ultralytics follows OpenCV: `results[0].plot()` returns a
  **BGR** array, and `plt.imshow` reads it as RGB, so red and blue are swapped (the
  orange bus becomes blue, and so do the people). In the app the PIL image goes in as
  RGB and the boxes are drawn on the RGB image, so the colours are right.

## 9.2 Dataset and training

### COCO → YOLO

`coco_to_yolo(input_folder, output_folder)` does exactly what the lab asks: it creates
`images/` and `labels/`, copies the images, reads `_annotations.coco.json` and writes one
`{image_name}.txt` per image with `class_id x_center y_center width height` normalised to
[0, 1]. `convert_coco_dataset` runs it for train/valid/test and writes `data.yaml` with
absolute paths.

Things to know for the Q&A:

* **COCO** boxes are `[x, y, w, h]` in pixels from the top-left corner; **YOLO** boxes
  are the centre and size divided by the image width/height.
* The Roboflow export has an extra category `1-2-3-SmhR` (id 0) that is only the
  *supercategory* of PC/RBC/WBC and has no boxes. The notebook uses `category_id - 1`;
  `coco_classes` detects the parent category instead, so plain COCO files also work.
* Boxes are clipped to the image, empty boxes are skipped, and images without boxes get
  an empty label file (a background image for YOLO).
* Checked against Roboflow's own YOLO export of the same data: all 3,459 label files
  have the same classes and boxes (max. difference 0.0008 from rounding).

### Exploration

3,459 images of 640 × 640 (train 3,027 / valid 288 / test 144), 63,920 boxes, ~18.5
cells per image. **Very imbalanced:** RBC 91 %, WBC 5.7 %, PC 3.2 % of the boxes. PCs
are small (8 % of the image side), WBCs large (40 %). The slide titled "number of images
per class" actually counts boxes; the tool shows both (RBC appears in every image, WBC in
99 %, PC in 37 %).

### Training

`train()` calls `model.train(data=..., epochs, batch, imgsz, patience, ...)`, with
callbacks for the progress bar. **Cancel** sets the trainer's own `stop` flag, so
Ultralytics stops after the current batch, validates and still saves `best.pt`. The
best weights are copied to `models/yolo/blood_cell_<model>.pt` (`_scratch` when trained
from a `.yaml`), and the run folder (`models/yolo/runs/...`) keeps Ultralytics' plots.

**Result** (`yolo11n.pt` fine-tuned, 50 epochs, batch 16, 640 px, RTX 5070, 31.8 min):
best epoch 48, validation mAP@0.5 = 0.977 and mAP@[.5:.95] = 0.835. The losses drop
suddenly at epoch 40: Ultralytics turns off the *mosaic* augmentation for the last 10
epochs (`close_mosaic=10`), so the last epochs see images like the real ones.

The notebook trains **from scratch** (`YOLO("yolo11n.yaml")`, random weights); the
slides recommend starting from the pretrained `yolo11n.pt`, which reached mAP@0.5 0.915
after a single epoch. Both are options in the training dialog, together with fine-tuning
YOLO26n, and each model is saved under its own name so they can be compared.

## 9.3 Test on an unseen image

The tool takes the working image or a random image of the **test** split (never used for
training or model selection). The ground truth is found from the image path
(`.../images/x.jpg → .../labels/x.txt`) or by name in the converted dataset, and the
tool says which split the image belongs to (it warns about train images).

The figure shows ground truth | prediction | errors: true positives in green, false
positives in red and missed cells dashed in yellow, plus the cell counts per class. Some
"errors" are annotation problems: there are duplicated boxes on the same cell and
cells cut by the border that are not labelled.

## Extra: our own evaluation

`detection_metrics.py` implements, in NumPy, what lecture 9 uses to validate a detector:

* **IoU** of every pair of boxes, **greedy NMS** per class or class-agnostic.
* **Matching**: detections sorted by score; each one takes the free ground-truth box of
  the same class with the highest IoU; it is a TP if IoU ≥ threshold, otherwise a FP
  (also a second detection of an object already found).
* **PR curve** and **AP** with all-point interpolation (area under the monotone envelope).
* **mAP** as in the lecture: AP of each class at every IoU threshold 0.50, 0.55, …, 0.95,
  averaged over thresholds and then over classes (COCO mAP@[.5:.95]); also mAP@0.5 and 0.75.
* **Miss rate vs false positives per image (FPPI)** and a **confusion matrix** with a
  background row/column (false alarms and missed objects).

The course notebook's `compute_map_table` reuses the matches made at IoU 0.5 for every
threshold, so it prints the same AP (0.9845) from 0.1 to 0.95. Here each threshold
re-matches the boxes, and a unit test checks that a box with IoU 0.6 counts at 0.5-0.6
and not above.

**Result on the 144 test images** (`blood_cell_yolo11n.pt`, 4 s on the GPU):

| | AP@0.5 | AP@0.75 | AP@[.5:.95] | objects |
|---|---|---|---|---|
| PC (platelets) | 0.970 | 0.910 | 0.784 | 91 |
| RBC (red cells) | 0.988 | 0.954 | 0.884 | 2,439 |
| WBC (white cells) | 0.989 | 0.989 | 0.889 | 148 |
| **mAP** | **0.982** | **0.951** | **0.852** | |

Ultralytics `model.val` on the same split: mAP@0.5 = 0.985, mAP@[.5:.95] = 0.865. The
small difference comes from the 101-point interpolation and the duplicate-label removal
of Ultralytics against our all-point interpolation and greedy matching.

Reading the plots: PC is the hardest class (small, few examples), and its AP falls
fastest when the IoU threshold rises (a few pixels of error is a large IoU change for
a small box). At confidence ≥ 0.25 only 2 cells get the wrong class; most errors are
background (false alarms, many of them partial cells at the border that are not
labelled) and 27 missed cells. Miss rate is 5.4 % at 1 false positive per image.

## Tests

`tests/test_detection_metrics.py`, `tests/test_yolo_dataset.py` and
`tests/test_yolo_tools.py` (62 tests) need no Ultralytics, PyTorch or network: the tools
are tested with a fake model that has the same `predict` interface.

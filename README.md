# ForensicsApp course starter

ForensicsApp is the semester project shell for the MSc Computer Vision course.  
Students add a small, demonstrable image-forensics feature as the course moves  
from classical image processing to deep learning and interpretability.

The starter already provides:

- a Tkinter desktop interface with tool, image, and results areas;
- opening and exporting common image formats;
- a fit-to-window image preview;
- undo, redo, and reset-to-original behavior;
- a tiny plug-in-style contract for adding course functionality;
- one analysis example and one image-processing example.

## Quick start

Python 3.10 or newer is recommended.

```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python run.py
```

On some Linux distributions, Tkinter is a separate system package. For example,  
Ubuntu/Debian users may first need `sudo apt install python3-tk`.

The complete student handout is [docs/STUDENT_GUIDE.md](docs/STUDENT_GUIDE.md).

## Useful commands

```
python -m forensics_app
python -m unittest discover -s tests -v
python -m compileall forensics_app
```

## Set 8: CNN models (optional)

The tools *CNN: Classify Image (Set 8.2)* and *CNN: Feature Projection (Set 8.3)* use  
CNNs trained in Colab with [docs/cnn_classification_set8.ipynb](docs/cnn_classification_set8.ipynb).  
Run the notebook, download `cnn_models.zip` and unzip it into `models/cnn/`, so every  
model has its own folder (`models/cnn/<model>/passport.json`, `model.keras`, ...).  
Loading the models needs TensorFlow, and UMAP needs umap-learn. See
[docs/SET8_CNN.md](docs/SET8_CNN.md) for the design and the results.

Install them with:

```
python -m pip install -r requirements-keras.txt
```

## Set 9: object detection with YOLO (optional dependencies)

The tools in the **Object Detection (YOLO)** sidebar section use Ultralytics,
which is optional; the rest of the app and the tests work without it.

```
.venv/bin/pip install -r requirements-yolo.txt   # see the file for CPU / CUDA PyTorch
```

1. **YOLO: Detect Objects (pretrained)**: COCO model on the working image (weights
   are downloaded to `models/yolo` the first time).
2. **YOLO: COCO → YOLO Dataset**: converts the Roboflow blood-cell export
   (`blood_cell.v1i.coco`) into `datasets/blood_cell_yolo` and shows its statistics.
3. **YOLO: Train Blood-Cell Detector**: trains and saves `models/yolo/blood_cell_*.pt`
   (about 45 s per epoch on a laptop GPU, about 10 min on CPU).
4. **YOLO: Blood Cells (GT vs Prediction)**: trained model on an unseen test image.
5. **YOLO: Evaluate (mAP, PR, FPPI)**: our own metrics on the test split.

The long steps also run from the command line:

```
.venv/bin/python -m forensics_app.yolo_cli convert "<lab folder>/blood_cell.v1i.coco"
.venv/bin/python -m forensics_app.yolo_cli train --epochs 50
.venv/bin/python -m forensics_app.yolo_cli evaluate --split test
```

`datasets/` and `models/` are git-ignored. See [docs/SET9_YOLO.md](docs/SET9_YOLO.md)
for the design and the results.

## Sets 10-11: CLIP, Vision Transformers and U-Net segmentation (optional dependencies)

Everything, training included, runs inside the app (no Colab). PyTorch trains on the
Apple GPU (MPS) or CUDA when there is one; CLIP also needs Hugging Face transformers.

```
.venv/bin/pip install -r requirements-clip.txt   # torch, torchvision, transformers
```

* **Vision-Language (CLIP)**: zero-shot classification with any number of classes and any
  text per class (templates, prompt ensembling, two models side by side) and text → image
  search. The weights are downloaded once to `models/clip`.
* **Vision Transformers (ViT)**: the extra notebook. Patches and shifted patch tokens,
  training on CIFAR-100 (downloaded to `data/` on first use), classification with attention
  rollout, passport and comparison (confusion matrix, classification report).
* **Segmentation (U-Net)**: TrayDataset (unzip it to `data/TrayDataset`): explore, train
  (binary / 3 classes / 43 classes, losses, skip connections), segment an image, passport
  and evaluation (threshold, IoU, worst images), compare models. Models go to `models/unet`.

See [docs/SET10_11_CLIP_VIT_UNET.md](docs/SET10_11_CLIP_VIT_UNET.md) for the design and the results.

## Course roadmap

The shell is intentionally neutral about the algorithms. Possible tool modules  
can follow the syllabus: filtering and edges, HOG, SIFT, bag of visual words,  
face detection/recognition, CNN classification and detection, transformers,  
semantic segmentation, and visualization/interpretability. Model weights and  
datasets should not be committed to the starter repository; document where they  
come from and keep paths configurable.

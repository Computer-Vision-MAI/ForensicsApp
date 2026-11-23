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

## Course roadmap

The shell is intentionally neutral about the algorithms. Possible tool modules  
can follow the syllabus: filtering and edges, HOG, SIFT, bag of visual words,  
face detection/recognition, CNN classification and detection, transformers,  
semantic segmentation, and visualization/interpretability. Model weights and  
datasets should not be committed to the starter repository; document where they  
come from and keep paths configurable.

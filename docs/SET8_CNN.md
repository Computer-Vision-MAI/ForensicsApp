# Set 8: Image classification with CNNs (passport, Grad-CAM, feature projection)

This branch (`feat/cnn`) implements Lab 8. The models are trained in Colab with a notebook,
saved together with a **model passport**, and two new app tools use them. Everything is in
the new sidebar section **Deep Learning (CNN)**.

| Lab item | Where | Code |
|---|---|---|
| 8.1 Train a small CNN, save the weights | Notebook, section 8.1a / 8.1b | `docs/cnn_classification_set8.ipynb` |
| 8.1 Transfer learning with InceptionV3, save the weights | Notebook, section 8.1c | same notebook |
| 8.1 Save the model passport (dataset, sample size, class names, preprocessing, history) | Notebook, `save_model_with_passport` | same notebook |
| 8.2 Load a pretrained model, display the passport, classify an image | **CNN: Classify Image (Set 8.2)** | `tools/cnn_model.py`, `tools/cnn_model_tools.py` |
| 8.3 Extract and save the test-set features, load them, project with t-SNE and UMAP | Notebook, section 8.3 + **CNN: Feature Projection t-SNE / UMAP (Set 8.3)** | same files |
| Extra: Grad-CAM, where in the image the network found the class | Classify tool (checkbox) + notebook | `cnn_model.py::gradcam_heatmap`, `KerasNetwork.gradcam_maps` |

## How the pieces fit together

```
Colab: docs/cnn_classification_set8.ipynb
   trains 3 models on CIFAR-10 and writes, per model:
   cnn_models/<model>/model.keras, passport.json, test_features.npz, test_samples/
        │  download cnn_models.zip, unzip into models/cnn/
        ▼
App: models/cnn/<model>/...
   ├─ CNN: Classify Image      reads passport.json + model.keras (+ test_samples/)
   └─ CNN: Feature Projection  reads passport.json + test_features.npz
```

The notebook and the app never share code (the notebook runs in Colab), so the
**passport is the contract** between them: the app knows nothing about a model except what
its passport says.

## Setup

```
.venv/bin/pip install -r requirements-keras.txt     # TensorFlow (Keras 3) + umap-learn
```

* TensorFlow and umap-learn are **optional**, like PyTorch for the Set 7 bonus and
  Ultralytics for Set 9. Without TensorFlow, *Classify Image* explains how to install it;
  without umap-learn, *Feature Projection* offers t-SNE only (t-SNE comes with scikit-learn).
  The tests replace the network by a NumPy fake, so CI needs neither.
* The trained models are **not committed**: `models/` is git-ignored. Run the notebook in
  Colab (GPU runtime), download `cnn_models.zip` and unzip it into `models/cnn/`, so every
  model has its own folder (`models/cnn/vgg_cnn/passport.json`, ...). The models used to
  check this branch were saved with Keras 3.13 / TensorFlow 2.21 in Colab and load with
  Keras 3.15 / TensorFlow 2.21 on Python 3.13 locally.

## 8.1 The notebook: training and saving

`docs/cnn_classification_set8.ipynb` reuses the lab notebook (`image_classification_using_cnn.ipynb`):
CIFAR-10 from `keras.datasets` (50,000 training and 10,000 test images of 32 × 32 px, 10
classes), the 10 × 10 grid of examples, one-hot labels, the categorical cross-entropy and
the lab's architectures. Three models are trained:

| Model | Architecture | Starting weights | Training data | Parameters |
|---|---|---|---|---|
| `small_cnn` | Conv 32 3×3 → MaxPool → Dense 100 → Dense 10 | random (from scratch) | 45,000 train + 5,000 validation | 722,006 |
| `vgg_cnn` | 2 × Conv 32 3×3 (same padding) → MaxPool → Dense 128 → Dense 10 | random (from scratch) | 45,000 + 5,000 | 1,060,138 |
| `inceptionv3_transfer` | Resizing 32→128 → InceptionV3 (no top) → GlobalAvgPool → Dense 1024 → Dense 10 | **ImageNet** (transfer learning) | balanced subset: 9,000 + 1,000 | 23,911,210 |

* **From scratch vs transfer learning.** The two small CNNs start from random weights and
  must discover every filter from CIFAR-10 alone. InceptionV3 starts from filters learnt on
  ImageNet (1.2 M photos, 1000 classes) and only adapts them. That is the comparison the
  slides ask for: "train a small model" and "transfer learning (InceptionV3)".
* **Small CNNs**: SGD (learning rate 0.01), batch 32, 30 epochs, `validation_split=0.1`
  (Keras keeps the last 10% of the training images for validation).
* **InceptionV3**, in two phases, as the lab asks:
  1. the base is **frozen** (`trainable = False`) and only the new layers learn (Adam 1e-3, 5 epochs);
  2. the base is **unfrozen** and the whole network is fine-tuned with a very low learning
     rate (Adam 1e-5, 3 epochs).

  The base is called with `training=False`, so its batch-normalisation layers stay in
  inference mode during fine-tuning (the standard Keras recipe).
* **Why the InceptionV3 model takes 32 × 32 inputs.** InceptionV3 needs at least 75 × 75 px,
  so the model upsamples the CIFAR images to 128 × 128 itself with a `Resizing` layer.
  The model input stays at CIFAR resolution, all three passports say "32 × 32", and a photo
  seen by the app is first reduced to the resolution the model was trained on.
  Upsampling all 50,000 images in memory would need about 10 GB; inside the model it is done
  batch by batch on the GPU.
* **Why a subset for InceptionV3.** `TRANSFER_TRAIN_SAMPLES = 10000` keeps training to about
  6 minutes on a Colab GPU. Setting it to 50000 uses the whole training set.

### Layer names the app relies on

Every model names a few layers so that the app can find them in any architecture:

| Name | Layer | Used for |
|---|---|---|
| `features` | last hidden layer (Dense 100 / 128 / 1024) | the feature vectors of 8.3 |
| `logits` | the 10 class scores **before** the softmax | Grad-CAM (the paper explains the pre-softmax score) |
| `predictions` | the softmax | the class probabilities |
| `last_conv` | last convolution of the two small CNNs | Grad-CAM |

The Grad-CAM layer is also written in the passport (`architecture.gradcam_layer`): `last_conv`
for the small CNNs, `inception_v3` (the whole pre-trained base, whose output is the last
feature maps) for transfer learning.

### What is saved per model

`save_model_with_passport` writes one self-contained folder, so a model can be copied around
on its own:

```
cnn_models/<model>/
    model.keras          architecture + trained weights (Keras 3 format)
    passport.json        the model passport (next section)
    test_features.npz    8.3: features of the 10,000 test images, labels and predictions
    test_samples/        10 test images per class, named <index>_<class>.png (e.g. 00013_horse.png)
```

The test samples give the app images whose **true class is known** (from the file name), so
it can say whether a prediction is correct.

## The model passport (`passport.json`)

The slides ask for the dataset and sample size, the class names, the preprocessing
instructions and the training history. The passport has those and a few more fields that
the app shows:

| Section | Content |
|---|---|
| `format_version` | `1`. The app refuses other versions with a clear message instead of misreading them |
| `name`, `title`, `description`, `created` | identity of the model |
| `framework` | Keras and TensorFlow versions used for training |
| `files` | names of the model, features and samples (defaults: `model.keras`, `test_features.npz`, `test_samples`) |
| `architecture` | readable summary, number of parameters, `feature_layer` + `feature_size`, `gradcam_layer` |
| `dataset` | name, source, image size, **train / validation / test sample sizes** |
| `class_names` | the meaning of every network output, in order |
| `preprocessing` | **how to prepare any image** (below) |
| `training` | optimizer, loss, epochs, batch size, validation split, time, and the phases of transfer learning |
| `history` | **loss, accuracy, val_loss, val_accuracy per epoch** (both phases merged for InceptionV3) |
| `evaluation` | test loss, test accuracy and the 10 × 10 test confusion matrix |

### Preprocessing instructions

```json
"preprocessing": {
  "input_size": [32, 32],
  "color_mode": "rgb",
  "resize_mode": "center_crop",
  "interpolation": "bilinear",
  "scale": 0.00784313725490196,
  "offset": -1.0
}
```

1. Convert to RGB (any PIL mode is accepted: grayscale, RGBA, 16-bit...).
2. `center_crop`: cut the largest **centred** region with the input's aspect ratio (a square
   here) and resize it to `input_size` with bilinear interpolation. A 640 × 480 webcam photo
   becomes its central 480 × 480 square, so objects keep their proportions instead of being
   squashed. (`stretch` would resize the whole image instead.)
3. Every pixel value `v` (0–255) becomes `v * scale + offset`: `1/255, 0` (range [0, 1]) for
   the small CNNs, `1/127.5, -1` (range [-1, 1], what `inception_v3.preprocess_input`
   does) for InceptionV3.

The notebook (`prepare_image`) and the app (`cnn_model.prepare_input`) implement these steps
separately. They were compared on six image shapes (wide, tall, LFW-shaped, square,
1000 × 3, odd sizes) and both scalings: the arrays are identical.

## 8.2 Tool: CNN: Classify Image (Set 8.2)

### Dialog

* **Trained model**: every folder of `models/cnn/` with a valid passport, labelled with its
  test accuracy. A folder with a broken passport is skipped and its error is reported; if
  there is no model at all, the error explains how to get one.
* **Image to classify**:
  * a **test image saved with the model** (true class known);
  * the **current image** in the app. A photo taken with *Face Analysis › Webcam Photo
    Capture* becomes the current image, so the webcam needs no separate option here. YuNet
    face cropping is not used: the CIFAR-10 classes are not faces, and the passport's centre
    crop already keeps the proportions;
  * **an image file** from disk.
* **Explanation**: "Show Grad-CAM" (on by default).

### What happens

1. Read the image and, if its name is `<index>_<class>.png` with a known class, its true class.
2. On a worker thread behind a progress window (TensorFlow takes seconds to import the first
   time): load the network with `keras.models.load_model(..., compile=False)` (inference
   only), preprocess the image exactly as the passport says, run it, rank the 10 classes.
3. If asked, compute Grad-CAM (next section).
4. Draw the figure and fill the results panel. The working image is not changed
   (`preview_only`).

The last loaded model stays in memory (keyed by file and modification time), so classifying
more images with the same model is instant: about 0.3 s for the small CNNs and 5 s for
InceptionV3, including loading.

### Result figure

* **Top row**:
  * the image, with a red box around the region the network sees (when it is cropped);
  * the **network input** itself (the 32 × 32 pixels, drawn as squares);
  * the **Grad-CAM** heatmap;
  * the **probability of every class**. The prediction is green when correct, red when wrong
    (then the true class is orange), and blue when the true class is unknown.
* **Bottom row: the passport.** Loss and accuracy per epoch (train, validation, and the test
  result as a dashed line; for InceptionV3 a dotted line marks where fine-tuning starts), and
  the test confusion matrix as percentages of each true class.

### Results panel

Prediction, ground truth, correct or not, the probability of the true class, top 3, the image
and the region given to the network, the Grad-CAM layer and map size, then **the whole
passport**: model, description, folder, date, framework, architecture, parameters, feature
layer, dataset, train / validation / test samples, classes, preprocessing, optimizer, loss,
epochs, batch size, training time, final train and validation accuracy, test accuracy and loss.

## Extra: Grad-CAM

Grad-CAM (Selvaraju et al., 2017) shows **which part of the image made the network choose
its class**. It helps catch a right answer for the wrong reason (e.g. "ship" because of blue
water), which matters in a forensics app.

`cnn_model.gradcam_heatmap(activations, gradients)`, plain NumPy:

1. take the `k` feature maps `A^k` of the Grad-CAM layer and the gradients `∂y^c/∂A^k` of
   the class score (the logit, before the softmax);
2. weight each map by the mean of its gradients: `α_k = mean(∂y^c/∂A^k)`;
3. add them up, `Σ_k α_k A^k`, and keep the positive part (ReLU: evidence *for* the class);
4. divide by the maximum (0 to 1).

The heatmap is stretched over the region the network saw and drawn with the `jet` colour map
(red = strongest evidence).

`KerasNetwork.gradcam_maps` gets the maps and the gradients with a TensorFlow
`GradientTape`. The **layers are called one by one inside the tape**. The usual recipe (a
helper `keras.Model(model.inputs, [layer.output, model.output])`) returned no gradient
(`None`) with the saved models: the helper graph was disconnected from the score. This only
works for models that are a simple chain of layers, which the Set 8 models are; otherwise
the app says so. Models without a `logits` layer still work (the log-probability is
explained instead), and models without `gradcam_layer` in the passport get the last
convolution found automatically.

**Resolution of the map** = resolution of the layer: 30 × 30 for `small_cnn` (3×3 convolution
without padding), 32 × 32 for `vgg_cnn` (same padding), and only **2 × 2** for InceptionV3,
whose last feature maps are tiny for a 128 × 128 input. The InceptionV3 heatmap therefore
only says which quarter of the image mattered.

Grad-CAM is an extra: if it fails, the prediction is still shown and the results panel
says "Grad-CAM: not available (reason)".

## 8.3 Feature projection (t-SNE, UMAP)

### In the notebook

`save_test_features` builds `keras.Model(model.inputs, model.get_layer("features").output)`,
runs the 10,000 test images and saves `test_features.npz` with:

* `features`: (10000, d) float32, with d = 100, 128 or 1024;
* `labels`: the true classes;
* `predictions`: the network's predictions.

The notebook then loads them and plots t-SNE and UMAP side by side.

### Tool: CNN: Feature Projection t-SNE / UMAP (Set 8.3)

* **Dialog**:
  * the model (only models with `test_features.npz`);
  * t-SNE, UMAP or both side by side (UMAP is disabled if umap-learn is missing);
  * the number of test images (default 2000);
  * the t-SNE perplexity (30) and UMAP `n_neighbors` (15);
  * "mark the misclassified images".
* **Steps** (on a worker thread):
  1. load the features;
  2. take a random subset (seed 0, **the same 2000 images as the notebook**), because t-SNE
     gets slow with many points;
  3. **PCA to 50 dimensions** (the usual first step: less noise, much faster t-SNE);
  4. project to 2-D: t-SNE with `init="pca"`, UMAP with `min_dist=0.1`, both with seed 0.
     The perplexity and `n_neighbors` are capped by the number of points.
* **Figure**: one scatter plot per method, every test image coloured by its true class, the
  misclassified ones marked with a small ×, one shared legend.
* **Results panel**:
  * the number of vectors and their dimension;
  * the parameters;
  * a **class separation score per method**: the silhouette of the true classes in 2-D,
    from -1 to 1, where 1 means compact, separate clusters;
  * the network accuracy on the projected images;
  * the points per class;
  * a short "how to read the plot".

## Results with the trained models

| | `small_cnn` | `vgg_cnn` | `inceptionv3_transfer` |
|---|---|---|---|
| Training time (Colab GPU) | 144 s | 181 s | 377 s |
| Final train accuracy | 93.9% | 100.0% | 96.9% |
| Final validation accuracy | 65.3% | 68.2% | 69.2% |
| Epoch with the lowest validation loss | 14 of 30 | 8 of 30 | 1 of 8 |
| **Test accuracy** | **63.9%** | **66.5%** | **68.3%** |
| Test loss | 1.36 | 2.42 | 0.96 |
| Hardest classes (test recall) | cat 45%, dog 51%, deer 55% | cat 46%, bird 48%, dog 54% | cat 53%, bird 58%, deer 62% |
| Most frequent confusion | dog → cat (191) | cat → dog (192) | dog → cat (155) |
| t-SNE silhouette (2000 test images) | -0.09 | -0.05 | 0.06 |

What the passports show:

* **All three overfit.** Training accuracy reaches 94–100% while validation stays at 65–69%.
  For `vgg_cnn` the validation loss is lowest at epoch 8 and then climbs, with a spike at
  epoch 22, while the training loss keeps falling. More epochs would not help. Data
  augmentation, dropout or early stopping would (not done: not in the lab).
* **Transfer learning wins with a fifth of the data**: 68.3% from 9,000 images and 8 epochs,
  with the lowest test loss (0.96 against 1.36 and 2.42). Its validation loss is lowest
  after the first epoch: the pre-trained features are useful immediately. With the full
  training set (`TRANSFER_TRAIN_SAMPLES = 50000`) it should improve further.
* **Cat and dog are the hardest pair for every model** (each one's most frequent
  confusion), followed by bird and deer: animals with similar shapes and backgrounds at
  32 × 32 px. The vehicles (airplane, automobile, ship, truck) and the frog are recognised
  best, at 68–80% recall depending on the model.
* **Feature projections**:
  * in the InceptionV3 projection the vehicles form their own clusters (automobile and
    truck close together, ship next to airplane);
  * the animals overlap in the middle, cat and dog most of all;
  * only InceptionV3 has a positive silhouette, but it is low for every model. The features
    separate some classes well, not all of them, consistent with the 64–68% accuracy.
* **Grad-CAM** on the horse test image (`00013_horse.png`) highlights the horse's back and
  outline for `vgg_cnn`. For InceptionV3 it is limited to the 2 × 2 map described above.

## Files in this branch

New:

* `docs/cnn_classification_set8.ipynb`: the Colab notebook (8.1, 8.2 with Grad-CAM, 8.3, download).
* `docs/SET8_CNN.md`: this document.
* `forensics_app/tools/cnn_model.py`: everything without a user interface, testable without
  TensorFlow:
  * passports (`ModelPassport`, `load_passport`, `list_models`);
  * preprocessing (`Preprocessing`, `crop_box`, `prepare_input`);
  * loading and running the network (`KerasNetwork`, `load_network`, `classify`);
  * test samples;
  * Grad-CAM (`gradcam_heatmap`, `explain`);
  * 8.3: `load_test_features`, `subsample`, `reduce_dimensions`, `project`, `separation_score`.
* `forensics_app/tools/cnn_model_tools.py`: the two dialogs, the figures
  (`render_classification`, `render_projection`, `gradcam_overlay`), the results-panel text,
  and the two tools.
* `requirements-keras.txt`: the optional dependencies (`tensorflow>=2.16`, `umap-learn>=0.5`).
* `tests/test_cnn_model.py`: the tests of this branch.

Modified:

* `forensics_app/tools/__init__.py`: registers `CnnClassifyTool` and `CnnProjectionTool`.
* `pyproject.toml`: optional dependency group `keras`.
* `README.md`: a short "Set 8: CNN models (optional)" section.

## Before merging

* This branch starts from the merge of PR #23. `origin/main` has since received the YOLO
  work (PR #24), which also edits `forensics_app/tools/__init__.py`, `README.md` and
  `pyproject.toml`. Expect conflicts in those three files when merging `main`: **keep both
  sides** (the YOLO tools and the CNN tools in the registry, both README sections, both
  optional dependency groups `yolo` and `keras`).
* The trained models stay out of git. Teammates need to run the notebook or receive
  `cnn_models.zip`, then unzip it into `models/cnn/`.

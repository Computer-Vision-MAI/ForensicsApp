# Sets 10-11: CLIP, Vision Transformers and U-Net segmentation

Lab 10 (materials of 24 Nov) asks for two things, with two reference notebooks:

* **10.1 CLIP zero-shot classification**, with an arbitrary number of classes and
  arbitrary text for each class (`CLIP_playground.ipynb`);
* **11.1 Segmentation with a U-Net** on TrayDataset: train a model and save it, then
  load it and test it, visualising the segmentation (`unet_tray_food_segmentation.ipynb`).

The extra notebook `ExtraMaterial_vision_transformers.ipynb` (ViT and Shifted Patch
Tokenization on CIFAR-100) is also implemented. **Everything runs inside the app, training
included**: there is no Colab step. Every tool has a modal dialog for its options and draws
its result as a figure, with the numbers in the Results panel.

| Lab item | Tool (sidebar) | Code (`forensics_app/tools/`) |
|---|---|---|
| 10.1 CLIP zero-shot classification (any classes, any text) | **CLIP: Zero-Shot Classification** | `clip_model.py`, `clip_tools.py`, `clip_render.py` |
| CLIP notebook 1.4: text → image search | **CLIP: Text → Image Search** | `clip_model.py::search` |
| 11.1 Explore TrayDataset | **U-Net: Explore Tray Dataset** | `tray_dataset.py` |
| 11.1 Train a U-Net and save it | **U-Net: Train Segmentation Model** | `unet_net.py`, `unet_model.py::train` |
| 11.1 Load a model and test it (visualise) | **U-Net: Segment Image** | `unet_model.py::segment_image`, `unet_tools.py::segment` |
| U-Net exercises 1, 4, 5, 6 and the passport | **U-Net: Model Passport & Evaluation** | `segmentation_metrics.py`, `unet_render.py` |
| U-Net exercises 2, 3, 6: models side by side | **U-Net: Compare Models** | `unet_render.py::render_comparison` |
| Extra notebook: patches and SPT | **ViT: Patches & Shifted Patch Tokens** | `vit_data.py`, `vit_render.py` |
| Extra notebook: train the ViT / the SPT ViT | **ViT: Train Classifier** | `vit_net.py`, `vit_model.py::train` |
| Extra notebook: use the trained ViT | **ViT: Classify Image** (with attention rollout) | `vit_model.py::classify` |
| Extra notebook: confusion matrix, classification report, comparison | **ViT: Passport & Compare Models** | `vit_model.py::classification_report` |

## Setup

```
.venv/bin/pip install -r requirements-clip.txt     # torch, torchvision, transformers
```

(The project venv was created with uv, so `uv pip install --python .venv/bin/python -r requirements-clip.txt`
does the same.) The packages are optional, as for Sets 7-9: without them the tools say
what to install, and the rest of the app and the tests work (the tests that need PyTorch are
skipped, so CI, which only installs `requirements.txt`, still passes).

What is downloaded and where (all git-ignored):

| What | Where | Size | When |
|---|---|---|---|
| CLIP ViT-B/32 (`openai/clip-vit-base-patch32`) | `models/clip/` | ~600 MB | first CLIP run |
| CLIP ViT-B/16 / ViT-L/14 (optional) | `models/clip/` | 600 MB / 1.7 GB | only if chosen in the dialog |
| TrayDataset | `data/TrayDataset/` | 20 MB | unzip `TrayDataset.zip` from the lab materials |
| CIFAR-100 | `data/cifar-100-python/` | 161 MB | the ViT Train tool asks before downloading |
| Trained models | `models/unet/<name>/`, `models/vit/<name>/` | 2-90 MB each | written by the Train tools |

Hardware used for the results below: MacBook Air M2, **8 GB** unified memory, PyTorch 2.15
on the Apple GPU (MPS). The tools pick CUDA, then MPS, then the CPU automatically.

## Design decisions shared by the three parts

* **PyTorch everywhere.** The CLIP and U-Net notebooks are PyTorch already. The ViT notebook
  is Keras, but TensorFlow has no GPU support on macOS (it would train on the CPU, ~10× slower),
  so the ViT was ported layer by layer to PyTorch (see the ViT section for the
  equivalences). The app needs one optional framework for Sets 10-11, not two.
* **Same structure as Sets 8-9.** Every part has a model module with no Tkinter
  (`clip_model`, `unet_model`, `vit_model`; testable), a figure module (`*_render`) and a tools
  module with the dialogs. torch and transformers are imported inside the functions that need
  them, so importing the app never requires them.
* **Training in the app.** Training runs on a worker thread (`run_with_progress`): the
  progress bar shows the epoch, the batch, the loss and the best validation score so far.
  **Cancel** stops after the current batch and, if at least one epoch finished, still
  evaluates and saves the best epoch (as YOLO in Set 9). A model is written to a temporary
  folder and renamed at the end, so a crash never leaves half a model.
* **Passports**, as in Set 8: every trained U-Net and ViT folder has a `passport.json` with the
  task, the classes, the preprocessing, the architecture and number of parameters, all the
  training settings, the history of every epoch and the validation and test metrics. The
  evaluation tools draw it, and the comparison tools read several passports, without needing
  PyTorch.
* **Data on the GPU.** The whole training set is copied to the GPU once as `uint8` and the
  batches are sliced there (normalisation and augmentation also happen on the GPU), which
  avoids moving every batch from NumPy.
* **Safety.** CIFAR is distributed as Python pickles, and `pickle` can run code while loading.
  `vit_data.safe_load` uses a restricted unpickler that only accepts the NumPy functions needed
  to rebuild arrays (a test checks that a malicious pickle is refused). The archive is checked
  against its official MD5 and unpacked with `tarfile`'s `data` filter (no absolute paths or
  links). Models are loaded with `torch.load(weights_only=True)`.

---

## 10.1 CLIP zero-shot classification

### How CLIP works (what the code does)

CLIP has two encoders trained together on 400 M image-caption pairs from the web:

* **image encoder**: a ViT. The image becomes 224 × 224, is cut into 7 × 7 = 49 patches of
  32 × 32 px (ViT-B/32), and the output of the class token is the image's representation;
* **text encoder**: a 12-layer transformer over at most 77 BPE tokens; the output at the
  end-of-text token is the sentence's representation.

A **linear projection** after each encoder maps both into the same 512-D space. Training is
contrastive: in a batch of N pairs, the N × N matrix of cosine similarities should be large on
the diagonal (each image with its own caption) and small elsewhere, with a cross-entropy over
the rows (image → text) and over the columns (text → image). The temperature is learnt:
`exp(logit_scale) = 100`.

**Zero-shot classification** needs no training:

1. each class becomes a sentence with a template, `"a photo of a {}."` → `"a photo of a cat."`;
2. the image and every sentence are encoded and normalised to length 1;
3. `logit_c = 100 · cos(image, text_c)`;
4. a softmax over the classes gives probabilities that always sum to 1.

### Implementation

`ClipEncoder` (`clip_model.py`) loads `CLIPModel` and `CLIPProcessor` from Hugging Face and calls the
pieces **explicitly**: `vision_model(...).pooler_output` → `visual_projection` →
normalise, and `text_model(...).pooler_output` → `text_projection` → normalise. The notebook
calls `get_image_features`, whose return type changed between transformers versions (the
notebook needs an `_as_tensor` workaround); the explicit calls are the same computation, do
not depend on the version, and show the architecture. Everything after the embeddings
(templates, ensembling, cosine, softmax, search) is NumPy, so it is unit-tested with a fake
encoder.

**Prompt ensembling** (CLIP paper, notebook 1.3): for each class, every template is encoded,
the normalised vectors are averaged and the mean is normalised again; that single vector
represents the class.

### The tool: CLIP: Zero-Shot Classification

The dialog has:

* **Model**: ViT-B/32 (notebook), ViT-B/16 or ViT-L/14, and **Compare with** a second model
  (Exercise 2 of the notebook: the two are shown one above the other).
* **Image**: the working image, a scikit-image sample (they work offline, as in the
  notebook: chelsea, coffee, astronaut, rocket, immunohistochemistry, …) or any file.
* **Classes**: a text box, **one class per line, any number of lines, any text**: this is
  exactly the "arbitrary number of classes, arbitrary text for each class" of the slide.
* **Templates**: one per line, `{}` marks where the class goes; validated (each template must
  contain `{}` exactly once).
* With several templates: **average them** (prompt ensembling) or **one chart per template**
  plus the average (prompt engineering, notebook 1.3).
* **Raw cosine similarities**: a second chart before the softmax.
* **Presets** fill the boxes with every example of the notebook and the slides (animals,
  how many cats, where are the cats, photo/painting/cartoon, drinks, the three templates, the
  five-template ensemble, healthy vs tumour tissue, forced choice cat/dog, with an "other"
  option, space, food on a hospital tray).

The figure reproduces the slide: the image, what CLIP sees (the 224 × 224 centre crop of the
preprocessing), and a horizontal bar chart of the probabilities per template, with the winner
in green. The Results panel lists, for every class, `p`, `cos` and `logit`, the tokens per
prompt (limit 77), and the model card (parameters per encoder, layers, patches, embedding
size, temperature, preprocessing).

### Results (ViT-B/32)

| Image | Classes, template | Result |
|---|---|---|
| coffee | coffee / tea / soup / hot chocolate, ensemble of the notebook's 5 templates | coffee **0.925**, tea 0.066, soup 0.005, hot chocolate 0.004 (**exactly the notebook's numbers**) |
| coffee | same, `"{}"` / `"a photo of a {}."` / `"a photo of a cup of {}."` | coffee 0.940 / 0.934 / 0.847 |
| chelsea | cat, dog, tiger, rabbit, fox, owl | cat **0.978**, tiger 0.013 |
| chelsea | one / two / three / no cats, `"a photo of {}."` | one cat **0.953** (chelsea is one cat) |
| chelsea | photo / painting / cartoon / pencil sketch, `"a {} of cats."` | photo 0.853 |
| chelsea | on a sofa / on the grass / in a car / on a bed | 0.31 / 0.28 / 0.28 / 0.13: flat, a close-up shows no place |
| astronaut | astronaut / rocket launch / pilot / scientist | an astronaut **0.997** |
| immunohistochemistry | healthy / tumour tissue, `"a microscopy image of {}."` | tumour 0.62, healthy 0.38 (cosines 0.299 vs 0.294) |
| coffee | cat / dog (forced choice) | cat **0.678** |
| coffee | cat / dog / something else | something else **0.905** |
| tray 1005a | 12 food names, `"a photo of {} on a hospital meal tray."` | jacket potato 0.34, carrot 0.17, soup 0.15 |

What these show (the notebook's discussion, 1.5):

* **The labels are the classifier.** The same image answers different questions just by
  changing the text; nothing is retrained.
* **Prompt engineering**: CLIP was trained on captions, so a sentence works better than a word
  when the classes are ambiguous; "a cup of {}" helps tea and soup *too*, so it lowers
  coffee's share. Ensembling gives a robust average.
* **The softmax always sums to 1.** In the forced choice the cosines are 0.198 vs 0.191, almost
  equal and much lower than for a real match (~0.30), but ×100 and the softmax turn them into
  68 % "cat". With an "other" option the answer is right. The cosine chart shows this
  directly, which is why the tool can show it.
* **Domain shift**: the tissue answer looks like a 62 % decision, but the two cosines differ by
  0.005; CLIP saw few expert microscopy captions (domain models exist: BiomedCLIP, PLIP, CONCH).
* **Single label**: a tray has 7-8 foods, but the softmax picks one; the probability spreads,
  and "carrot" wins over the orange pumpkin. That is why food segmentation needs a model like
  the U-Net, trained on pixel labels.

### CLIP: Text → Image Search (notebook 1.4)

The gallery (the 19 scikit-image samples, or every image in a chosen folder, e.g.
`data/TrayDataset/XTest`) is encoded once and cached per model. Each query (one per line) is
encoded and the images are ranked by cosine similarity. The figure shows the top-k images per
query and, optionally, the whole **text × image similarity matrix**, which is the matrix that
CLIP's contrastive loss makes large on the matching pairs.

| Query | Best images (cosine) |
|---|---|
| something you drink in the morning | coffee 0.259, gravel 0.229, clock 0.225 |
| a person who has been to space | astronaut 0.272, hubble_deep_field 0.244, moon 0.243 |
| a microscope image of tissue | immunohistochemistry 0.288, grass 0.268 |
| the surface of the moon | moon 0.282 |
| a page of printed text | text 0.240 |

---

## 11.1 Semantic segmentation with a U-Net

### The dataset (U-Net: Explore Tray Dataset)

* XTrain/yTrain: **1,241** images and masks; XTest/yTest: **8**. All 416 × 256.
* A mask stores the class index of every pixel: 0 background, 1 tray, 2 cutlery, 3 form,
  4 straw, 5-42 food and drinks (43 classes). Some masks are saved with 3 identical channels;
  `read_mask` keeps the first.
* Background + tray are **87 %** of the pixels and food **10.6 %** (the notebook's 10.6 %).
  So pixel accuracy is misleading: predicting "no food" everywhere is 89.4 % accurate with a
  Dice of 0.
* Nine classes never appear in XTrain (form, straw, meatball, beef, carrot, broccoli, celery,
  apple, juice).
* **XTrain is only 17 different trays, each photographed 73 times** (moved and rotated;
  `1001a01` … `1001a73`). The 8 test images are 8 *other* trays. This matters for validation
  (see "Validation split" below).

The tool shows the pixel share of every class (log scale), the distribution of food per image,
and four images with their masks merged into the chosen task.

### Tasks

The masks are mapped with a lookup table (`Task.map`), so one dataset gives three tasks:

| Task | Classes | Mapping | Output |
|---|---|---|---|
| `binary` (notebook) | not food, food | 0-4 → 0, 5-42 → 1 | 1 logit → sigmoid → threshold |
| `three` (Exercise 6) | background, tray, food | 1 → tray, 5-42 → food, rest → background | 3 logits → softmax → argmax |
| `all` (slides' figure) | the 43 classes | identity | 43 logits → softmax → argmax |

Cutlery, the menu form and the straw are neither tray nor food, so in the three-class task
they go with the background.

Images are resized bilinearly and **masks with nearest-neighbour interpolation**, which never
invents a class index between two classes (a test checks it).

### The network

`unet_net.UNet` is the notebook's class: 3 encoder blocks (16, 32, 64 channels), a
bottleneck (128), 3 decoder blocks with 2 × 2 transposed convolutions, each block two
3 × 3 convolutions with BatchNorm and ReLU, skip connections by concatenation, and a 1 × 1
convolution head. **482,737 parameters**, as in the notebook (a test checks the number).

* `skip=False` (Exercise 2) removes the concatenations; the first convolution of each decoder
  block then has half the input channels: 434,353 parameters.
* The multi-class heads add 34 (3 classes) or 714 (43 classes) parameters.

### Losses (Exercise 3)

* **BCE + Dice** (notebook): binary cross-entropy + (1 − soft Dice), soft Dice computed on the
  probabilities per image, so it is differentiable.
* **BCE only** or **Dice only**.
* For several classes BCE becomes **cross-entropy** and the soft Dice is the mean over the
  classes **present in the batch** (one-hot target × softmax). Classes absent from the batch
  are left out: otherwise every rare class would add a Dice of ~0 that can never improve.

### Training (U-Net: Train Segmentation Model)

Defaults = the notebook: 128 × 208 images, Adam lr 1e-3, batch 16, 15 epochs, random
horizontal flip of the image **and** its mask (the same flip for both), 15 % of XTrain for
validation with seed 42 (**the same 186 images as the notebook**: `RandomState(42).permutation`),
and the weights of the epoch with the **best validation Dice** are kept.

The dialog also sets the task, the loss, the skip connections, the base channels, the image size
(64 × 104, 128 × 208 or the original 256 × 416), epochs, batch, learning rate, validation fraction,
seed, augmentation, the device and the model name; plus **"Validation = whole trays"** (below).
On the M2 one epoch at 128 × 208 takes ~24 s (MPS); on the CPU ~70 s.

At the end the best epoch is evaluated on validation (with the threshold sweep) and on the 8
test images, and the passport figure is shown.

### Metrics (`segmentation_metrics.py`, NumPy)

* **Dice** = 2|P ∩ G| / (|P| + |G|) per image, then the mean over the images (the notebook's
  `evaluate`). When both masks are empty the notebook's `eps` makes it 1; we keep that.
* **IoU** = |P ∩ G| / |P ∪ G|.
* Multi-class: Dice and IoU per class and image, NaN when the class is in neither mask, so the
  class average only counts the images where the class matters. The "main" Dice of an image
  is the mean over its foreground classes.
* Pixel confusion matrix and pixel accuracy.

### Using a model (U-Net: Segment Image)

Choose a model, then the working image, one of the 8 test images, or a random validation image.
The image is resized to the network size, and the **probabilities** are resized back to the
original size (bilinear) before the threshold / argmax, which gives smoother borders than
enlarging the mask. The figure is the slide's figure, **Original / Segmentation mask / Overlay**,
plus, for dataset images, the **ground truth**, an **error map** (TP green, FP red, FN blue,
wrong class orange) and the probability (or confidence) map. The Results panel gives Dice,
IoU, pixel accuracy, the Dice of each class and the TP/FP/FN pixel counts, and says whether the
image was seen in training. The threshold is a dialog option for binary models.

### Results (in the app, M2 GPU, 128 × 208, notebook split)

| Model | Params | Best epoch | Val Dice | Val IoU | Test Dice | Test IoU | Time |
|---|---|---|---|---|---|---|---|
| binary, BCE + Dice (**notebook**) | 482,737 | 13/15 | **0.960** | 0.923 | **0.931** ± 0.044 | 0.875 | 6.0 min |
| binary, BCE + Dice, **no skip** (Ex. 2) | 434,353 | 15/15 | 0.949 | 0.904 | 0.928 ± 0.032 | 0.868 | 6.3 min |
| binary, **BCE only** (Ex. 3) | 482,737 | 15/15 | 0.960 | 0.924 | 0.932 ± 0.045 | 0.876 | 11.7 min* |
| binary, **Dice only** (Ex. 3) | 482,737 | 15/15 | 0.960 | 0.925 | **0.942** ± 0.033 | 0.893 | 9.7 min* |
| background / tray / food (Ex. 6) | 482,771 | 15/15 | 0.968 | 0.938 | 0.957 ± 0.018 | 0.919 | 7.9 min |
| all 43 classes, 40 epochs | 483,451 | 36/40 | 0.929 | 0.888 | **0.691** ± 0.264 | 0.643 | 26.2 min |
| binary, **tray split** | 482,737 | 11/15 | 0.945 | 0.898 | 0.932 ± 0.045 | 0.876 | 4.1 min |
| all 43 classes, **tray split**, 40 epochs | 483,451 | 38/40 | 0.542 | 0.501 | 0.660 ± 0.268 | 0.610 | 15.0 min |

\* these two ran while other jobs shared the GPU, so their times are not comparable.

The notebook reports validation Dice 0.956 and test Dice 0.936 ± 0.042: the same within the
noise of a different GPU and random order.

### The notebook's exercises

**1. Threshold.** Dice of the notebook model on validation: 0.959 at 0.3, **0.960** at 0.5, 0.958
at 0.7 (best 0.45). The curve is flat between ~0.2 and 0.8 (see the passport figure) because
the network is confident: almost every pixel has P(food) near 0 or 1, so moving the threshold
only changes the few border pixels. A lower threshold adds border pixels (more FP, fewer FN),
a higher one removes them. The BCE-only model has a sharper peak at 0.5; the Dice-only model
is flattest (best 0.65). On the 8 test images the best threshold is 0.20: choosing it there
would be tuning on the test set.

**2. Skip connections.** Without them validation Dice drops from 0.960 to 0.949 and the curve
is lower in every epoch (best epoch = the last one). On the test set the two are almost equal
(0.931 vs 0.928, 8 images). The shapes show the difference: with skip connections the mask
follows the real outline (the square corners of a sandwich box, the irregular salad); without
them the decoder only has the 16 × 26 bottleneck to rebuild 128 × 208 pixels, so the masks are
smooth rounded blobs. That smoothing can even win on one image (test 5006a: 0.885 without vs
0.867 with, because the skip model also marks part of a spoon), but on average fine detail
wins. The skip connections give back the spatial detail lost by the pooling, which is the key
idea of the U-Net.

**3. Loss.** Validation Dice per epoch: BCE alone is ahead at the start (0.891 at epoch 2, vs
0.834 for BCE + Dice and 0.851 for Dice), because it gives a gradient to every pixel. Dice alone
is the least stable (it drops to 0.689 at epoch 4: with few food pixels in a batch its gradient is
noisy), but it optimises the evaluation metric directly and ends best on test (0.942). After 15
epochs the three reach the same validation Dice (0.960); with only 8 test images the differences
are within one standard deviation. BCE + Dice combines BCE's stable start with Dice's robustness to
the class imbalance (10 % food). BCE also has the sharpest threshold curve (its probabilities are
calibrated around 0.5), Dice the flattest (it pushes every pixel to 0 or 1).

**4. IoU.** With I = |P ∩ G| and S = |P| + |G|: Dice = 2I/S and IoU = I/(S − I), so
**IoU = Dice / (2 − Dice)** and Dice = 2·IoU / (1 + IoU). The evaluation tool checks it on every
image (maximum error 2e-16) and plots IoU against Dice with that curve. It holds per image,
**not for the averages**: IoU(mean Dice 0.9314) = 0.8715, but the mean IoU is 0.8747 (the function
is convex). Dice ≥ IoU always; IoU punishes partial overlaps more.

**5. Worst cases.** Worst validation images of the notebook model: 4002a05 (0.743), 3001a15
(0.839), 5007a30 (0.848). The error maps show why:

* 4002a05: half of the **white rice** is missed: white food on a white plate;
* 3001a15 and 5007a30: **food on the neighbouring trays** at the image edges is not annotated
  (only the central tray is), so the network's correct "food" there counts as false positives
  (red). Those are labelling choices, not model errors;
* packaged or transparent food (sandwich boxes, drinks) is partly missed at its borders.

**6. Multi-class.** Background / tray / food with softmax and cross-entropy (+ Dice):
Dice per class on validation 0.992 / 0.976 / 0.959, on test 0.992 / 0.973 / **0.941**. The food Dice
is slightly *higher* than the binary model's (0.941 vs 0.931 on test): learning the tray as its
own class helps to separate it from food. The 43-class model (the slides' coloured mask) reaches
0.929 on validation but only 0.691 on test: see the next section.

### Validation split: why the 43-class model collapses on test

The notebook's split is random over the 1,241 images, but XTrain is 17 trays × 73 photos: **all
17 trays appear in both training and validation**, so a validation image is nearly a copy of
some training images. For "food vs not food" this barely matters (validation 0.960, test 0.931),
because colour and texture generalise. For 43 classes the model must recognise *which* food it
is on trays it has never seen, and several classes appear on a single training tray: cucumber,
ham, bean, leaf, tomato, canned fruit and apple juice are each on one tray (72-73 images). On the
test trays cucumber gets Dice 0, ham 0.29, tomato 0.32: the network memorised those trays.

The Train tool therefore has the option **"Validation = whole trays"** (`split_indices(...,
by_tray=True)`): it holds out 3 of the 17 trays (219 images) for validation, so validation is
honest:

| Task | Split | Validation Dice | Test Dice | Gap |
|---|---|---|---|---|
| binary | random (notebook) | 0.960 | 0.931 | validation 0.029 too optimistic |
| binary | by tray | 0.945 | 0.932 | 0.013 |
| 43 classes | random (notebook) | **0.929** | 0.691 | validation 0.24 too optimistic |
| 43 classes | by tray | **0.542** | 0.660 | validation now *pessimistic* |

For the binary task the tray split changes little: the test Dice is the same (0.932 vs 0.931)
and validation becomes a fair estimate. For 43 classes the random-split validation (0.929) was
almost meaningless; with held-out trays it drops to 0.542, even below the test, because the 3
held-out trays contain foods that appear on no other tray (they are never learnt), while the 8
test trays happen to share more foods with the training trays. The tray-split model also trains on
14 trays instead of 17, which costs a little test Dice (0.660 vs 0.691). The lesson: when the data
has groups (patients, videos, trays), split by group, otherwise near-duplicates leak into
validation and model selection (best epoch, threshold) is made on an optimistic number.

---

## Extra: Vision Transformers (`ExtraMaterial_vision_transformers.ipynb`)

### Port from Keras to PyTorch (`vit_net.py`)

Each Keras layer of the notebook has an equivalent:

| Notebook (Keras) | Here (PyTorch) |
|---|---|
| `Normalization().adapt(x_train)` | per-channel mean and std of all training pixels, stored in the model |
| `Resizing(72, 72)`, `RandomFlip`, `RandomRotation(0.02)`, `RandomZoom(0.2, 0.2)` | `Augment`: bilinear resize, then one `affine_grid` per image with flip, ±0.02 turns (±7.2°) and ±20 % zoom per axis, **reflected** borders (Keras' default fill mode); only in training mode |
| `Patches` (`tf.image.extract_patches`) | `patchify`: same (row, column, channel) order (a test compares it with NumPy) |
| `PatchEncoder`: `Dense(64)` + `Embedding(144, 64)` | `PatchTokens` + `nn.Embedding` |
| `MultiHeadAttention(num_heads=4, key_dim=64, dropout=0.1)` | `Attention`: each head has its own 64-D queries, keys and values (4 × 64 = 256 inside), output projected back to 64. PyTorch's `nn.MultiheadAttention` would split 64 into 4 × 16 instead, a different model |
| `mlp(x, [128, 64], 0.1)` with GELU | same |
| `LayerNormalization(epsilon=1e-6)`, `Add()` skips | same (pre-norm blocks) |
| head: LayerNorm → Flatten → Dropout 0.5 → MLP [2048, 1024] → Dense(100) | same; 144 × 64 = 9,216 values flattened |
| `AdamW(1e-3, weight_decay=1e-4)`, batch 256, 10 epochs | same |
| `validation_split=0.1` | the **last** 10 % of the training images (what Keras does) |
| `ModelCheckpoint(monitor="val_accuracy", save_best_only=True)` | keep the epoch with the best validation accuracy |

The model has **21.8 M parameters**, 19 M of them in the first Dense layer of the head
(9,216 × 2,048): the flatten head is the notebook's choice; the ViT paper uses a class token
instead.

**8 GB of memory.** Eight attention layers over 144 tokens for 256 images do not fit together
with the rest in 8 GB; the first run was swapping (60 s per batch). The batch of 256 is therefore
computed in **micro-batches** of 64 whose gradients are added before one optimiser step
(gradient accumulation): the same batch of 256 for AdamW, only the dropout masks differ. It is
a dialog option (0 = the whole batch at once, e.g. on a CUDA GPU).

### Shifted Patch Tokenization and Locality Self-Attention

The exercise asks for an SPT layer. `ShiftedPatchTokens` follows the Keras example of the SPT
paper (Lee et al. 2021, "Vision Transformer for Small-Size Datasets"): the image is shifted by
half a patch (3 px) to the left-up, left-down, right-up and right-down (crop + zero pad), the
five images are stacked on the channel axis (15 channels), cut into 6 × 6 patches (540 values),
**layer-normalised** and projected to 64. Each token then also contains the half patch around
its own patch: a locality bias that a plain ViT lacks. The patch tool draws the four shifted
images and the 5 views that form one token.

The tool also offers **LSA** (the paper's second idea): a learnable temperature instead of the
fixed √d_k, and each token is not allowed to attend to itself (diagonal masked to −∞), which
sharpens the attention on the other tokens.

### The tools

* **ViT: Patches & Shifted Patch Tokens**: the notebook's patch visualisation (the 72 × 72
  image with the 12 × 12 grid and the 144 patches), the four SPT shifts and one SPT token, for a
  random CIFAR test image or the working image; no PyTorch needed.
* **ViT: Train Classifier**: dataset (CIFAR-100, or CIFAR-10 already on disk), tokenizer
  (patches / SPT / SPT + LSA), epochs, batch, micro-batch, learning rate, weight decay, layers,
  heads, image and patch size. Asks before downloading CIFAR-100.
* **ViT: Classify Image**: a test image (true class known) or the working image (centre square
  → 32 × 32), the top-10 probabilities and an **attention rollout** map (Abnar & Zuidema 2020:
  per layer the head-averaged attention plus the identity for the skip connection, multiplied
  over the layers; averaged over all output tokens because this ViT has no class token).
* **ViT: Passport & Compare Models**: one model → loss and accuracy curves, the 100 × 100 test
  confusion matrix and the classification report (precision / recall / F1 per class, best and
  worst classes, most frequent confusions). Two or more → curves, test accuracy / top-5 /
  macro-F1, F1 per class of one model against the other, and an image-by-image table (both
  right, only A, only B, both wrong), as the exercise "compare both models" asks.

### Results (CIFAR-100, 10 epochs, M2 GPU)

Both trained in the app with the notebook's settings (AdamW 1e-3 / 1e-4, batch 256 as 4 × 64,
10 epochs, last 10 % of the training set for validation, best validation accuracy kept):

| | ViT (patches, notebook) | ViT + SPT (exercise) |
|---|---|---|
| Parameters | 21,759,012 | 21,787,740 |
| Best epoch (validation accuracy) | 10: 28.6 % | 9: **33.7 %** |
| **Test accuracy** | 29.1 % | **33.8 %** |
| Test top-5 accuracy | 59.2 % | **64.6 %** |
| Test loss | 2.882 | **2.614** |
| Macro F1 | 0.280 | **0.324** |
| Time (M2 GPU) | 22.5 min | 22.4 min |
| Best classes (F1) | sunflower 0.72, plain, road, lawn mower | chair 0.71, road, plain, sunflower |
| Worst classes (F1) | possum 0.02, couch, otter, squirrel, mouse | otter 0.02, mouse, bear, bowl, woman |
| Most frequent confusions | rose → tulip 41, whale → dolphin 35, shark → dolphin 32 | bus → streetcar 31, dolphin → whale 31, maple → oak tree 25 |

* **SPT helps from the first epoch** (validation 15.4 % vs 11.2 % after epoch 1, 22.4 % vs 19.6 %
  after epoch 3) and ends +4.7 points on test for only 28,728 extra parameters (the LayerNorm and
  the larger input of the projection: 540 instead of 108 values). 73 of the 100 classes get a
  better F1. Image by image: 2,108 test images right with both, 798 only with the plain ViT,
  **1,270 only with SPT**, 5,824 wrong with both. McNemar's test on the 2,068 disagreements gives
  χ² = 107 (> 3.84), so the gain is not chance.
* The largest gains are objects with clear shapes and edges (chair +0.26 F1, wardrobe, couch,
  television): the shifted views put the neighbouring pixels of an edge into the same token, which is
  the locality a CNN has by construction.
* Both models are **under-fitted**, not over-fitted: the training accuracy (measured with the
  augmentation and dropout on) is below the validation accuracy, and validation still rises at
  epoch 9-10. Ten epochs are far too few for a ViT from scratch (the Keras example of this model
  trains for 100 epochs and reaches ~55 %). This is the inductive-bias question of the notebook in
  practice: without locality built in, the transformer needs many more epochs (or data) to learn it.
* The confusions are between visually similar CIFAR-100 classes of the same superclass (rose / tulip,
  whale / dolphin / shark, maple / oak / willow tree, bus / streetcar); small furry mammals (otter,
  possum, mouse, squirrel, bear) are the hardest at 32 × 32 px.
* The attention rollout of the classify tool shows which patches the tokens attend to most; it is an
  approximation (it ignores the MLPs and the flatten head), so read it as a hint, not a proof.

### Answers to the notebook's questions

**CNN vs Transformer.** A CNN applies the same small filters everywhere: it is local
(a pixel only sees its neighbours, and the receptive field grows with depth) and translation
equivariant. A transformer splits the image into patches and lets every patch attend to every
other patch from the first layer: global context at once, weights computed from the data
(attention), but no built-in notion of neighbourhood. CNNs learn well from small datasets; ViTs
need large datasets or pre-training (or tricks such as SPT, LSA, distillation) to beat them.

**Multi-head self-attention.** Each token is projected to a query, a key and a value. The
attention weights of token i are softmax(q_i · k_j / √d) over all tokens j, and its output is the
weighted sum of the values. With several heads this is done in parallel with different
projections, so each head can look for a different relation (colour, shape, position); the head
outputs are concatenated and projected back.

**Inductive bias.** The assumptions built into a model before seeing data. CNNs have strong
ones (locality, weight sharing, translation equivariance, hierarchy); ViTs have very few (only the
patching and the position embeddings, which are learnt), so they must learn spatial structure from
data: more flexible, but data-hungry. That is why the notebook's ViT trained from scratch on
CIFAR-100 stays far below a CNN, and why SPT/LSA (which add locality back) help.

**ViT architecture / Patch Encoding layer / its `call`.** Image → patches → each flattened patch is
projected linearly to the model dimension (the "word embedding" of the patch) and a learnt
**position embedding** is added (`call` builds the positions 0…143, embeds them and adds them
to the projected patches), because attention is permutation-invariant and would otherwise ignore
where each patch is. Then L transformer blocks and a classification head.

**Sequence of operations of the model.** Input 32 × 32 → normalise, resize to 72, augment → 144
patches of 108 values → Dense 64 + position embedding → 8 × [LayerNorm → MHSA → + skip →
LayerNorm → MLP → + skip] → LayerNorm → flatten → dropout → MLP 2048, 1024 → Dense 100 (logits).

**Skip connections.** x + f(x): the block learns a correction to its input. Gradients flow
directly through the additions to the early layers (no vanishing gradient in deep stacks), and a
block that is not useful can stay close to the identity. Without them an 8-layer transformer is
much harder to train.

**Problem of global self-attention and alternatives.** Its cost is quadratic in the number of
tokens: 144 tokens → 20,736 pairs per head and layer; a 224 × 224 image with 4 × 4 patches would
have 3,136 tokens → ~10 M pairs. Alternatives: bigger patches (fewer tokens), **local/windowed
attention** (Swin), shifted windows to connect them, hierarchical token merging (Swin, PVT),
linear/low-rank approximations (Linformer, Performer), or convolutional stems.

**DeiT vs ViT.** DeiT (Data-efficient image Transformers) uses the same ViT architecture but
trains it on ImageNet-1k alone (ViT needed JFT-300M) thanks to strong augmentation and
regularisation and **distillation**: an extra *distillation token* learns to reproduce the
predictions of a teacher network (a CNN, RegNet), while the class token learns the true labels.
The limitation it overcomes: ViT's need for huge datasets.

**Swin Transformer.** A hierarchical ViT: self-attention only inside local windows (cost linear
in the image size), windows shifted between consecutive layers so that information crosses window
borders, and patch merging that halves the resolution and doubles the channels at each stage,
like a CNN pyramid. Unlike ViT/DeiT (one resolution, global attention, one class token), its
multi-scale features suit detection and segmentation.

**Distillation.** Training a student model to match a teacher's outputs (soft probabilities, or
in DeiT's "hard" variant the teacher's predicted class) besides the labels. The teacher's outputs
carry more information than one-hot labels (which classes look alike), and they can transfer the
teacher's inductive biases to the student. **Which teacher?** DeiT found that a **CNN teacher**
works better than a transformer teacher of similar accuracy: the student inherits the CNN's
locality bias, which is what a ViT lacks on limited data.

---

## Files

| File | Content |
|---|---|
| `tools/clip_model.py` | CLIP encoder, zero-shot maths, prompt ensembling, gallery and search |
| `tools/clip_tools.py`, `tools/clip_render.py` | CLIP dialogs, presets, figures |
| `tools/tray_dataset.py` | TrayDataset: pairs, masks, tasks, sizes, cached loading, splits (random / by tray), statistics |
| `tools/segmentation_metrics.py` | Dice, IoU, confusion, threshold sweep, per-image and per-class evaluation |
| `tools/unet_net.py` | the U-Net (with/without skips) and the losses (PyTorch) |
| `tools/unet_model.py` | training, passports, loading, prediction, evaluation |
| `tools/unet_tools.py`, `tools/unet_render.py` | the five U-Net tools and their figures |
| `tools/vit_data.py` | CIFAR download and safe loading, patches, SPT shifts |
| `tools/vit_net.py` | the ViT, SPT and LSA (PyTorch) |
| `tools/vit_model.py` | ViT training, passports, classification report, attention rollout |
| `tools/vit_tools.py`, `tools/vit_render.py` | the four ViT tools and their figures |
| `tools/yolo_dialogs.py` | the shared form dialog gained text (`str`) and folder (`dir`) fields |
| `requirements-clip.txt`, `pyproject.toml` (`clip` extra) | optional dependencies |
| `tests/test_segmentation.py`, `test_unet_torch.py`, `test_clip.py`, `test_vit.py` | 42 new tests |

## Tests

```
.venv/bin/python -m unittest discover -s tests
```

802 tests pass (7 skipped without PyTorch). The tests build a tiny fake TrayDataset and fake
CIFAR pickles, use a fake CLIP encoder (colours as embeddings), check the metrics against the
notebook's 4 × 4 example and sklearn, the IoU-Dice identity, the notebook's validation split, the
tray split, nearest-neighbour mask resizing, the parameter count 482,737, Keras-order patches, SPT
shifts, that a malicious pickle is refused, and train tiny U-Nets and ViTs end to end on the CPU
(save, list, load, segment / classify, evaluate).

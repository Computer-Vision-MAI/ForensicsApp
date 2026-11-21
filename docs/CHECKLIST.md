# Pending Tasks

## Week 1
Python, tkinter, scikit-learn, numpy, image processing/manipulation

### Functionalities:
- [x] Window
- [x] Open image
- [x] Inspect image
- [x] Convert to grayscale 
- [x] Save result

## Week 2

### Functionalities:
- [x] Channel split
- [x] Channel swap
- [x] Masking
- [x] Histogram visualization
- [x] Contrast stretching

## Week 3

### Functionalities:
- [x] Convolution with arbitrary kernel
- [x] Filters from skimage.filters
- [x] Canny edge detection
- [x] Edge visualization (superimposed on original)
- [x] Image sharpening

## Week 4

### Functionalities:
- [x] Normalized cross-correlation image
- [x] Template matching configs (size, angle, threshold)
- [x] HOG extraction and visualization
- [x] Person detection using HOG

## Week 7

### Functionalities:
- [x] Codebook construction (K-Means, K = 10 x classes) and visual word visualization
- [x] Histogram of visual words (features coloured by word, linked to their bar)
- [x] Classification with a Classify button and the top probability p(class)

### Extras:
- [x] Stratified train/test split; codebook fitted on train only (no leakage)
- [x] Test accuracy, per-class precision/recall and confusion matrix
- [x] Chi-squared kernel SVM, C chosen by cross-validation, calibrated probabilities
- [x] TF-IDF weighting and L1/L2 normalization
- [x] Spatial pyramid (PHOW-style, up to 1+4+16 cells)
- [x] Soft assignment and K chosen by cross-validation
- [x] Patches and source classes of each visual word; most similar training images

## Week 9

### Functionalities:
- [x] Load a pretrained Ultralytics YOLO (yolo11n, COCO) and run it on an image
- [x] COCO → YOLO conversion (`coco_to_yolo(input_folder, output_folder)` + `data.yaml`)
- [x] Train YOLO on the blood-cell dataset (PC, RBC, WBC) and save the model
- [x] Load the trained model and test it on an unseen image (GT and predicted boxes)

### Extras:
- [x] Dataset exploration: class imbalance per split, cells per image, box sizes, labelled samples
- [x] Own greedy NMS (per class / class-agnostic) with the boxes before and after it
- [x] Training from the app on a worker thread: live progress, clean cancel, loss and mAP curves
- [x] Start from COCO weights (fine-tune), from scratch (.yaml) or YOLO26; each model saved separately in models/yolo
- [x] YOLO26 NMS-free one-to-one head vs NMS
- [x] TP / FP / missed cells and cell counts per image (prediction vs ground truth)
- [x] Own evaluation: per-class AP at IoU 0.50:0.95, mAP, PR curves, miss rate vs FPPI, confusion matrix,
      compared with Ultralytics `model.val` (fixes the notebook, which reused the IoU 0.5 matches)
- [x] "Why are people blue?": BGR vs RGB handled and explained
- [x] Command line for the long jobs: `python -m forensics_app.yolo_cli convert|train|evaluate`

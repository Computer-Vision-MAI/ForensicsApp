"""Core implementation of Eigenfaces (PCA), Fisherfaces (LDA), and face recognition.

Follows 'docs/Materials-CV-MAI-Set6.pdf' (Sets 6.2 & 6.3) and 'docs/face_detection_and_recognition.ipynb'.
Includes:
  - Dataset preparation: downloading/extracting LFW to folders with label correspondence.
  - Model training: PCA eigenfaces, projection coefficients, and LDA Fisherfaces.
  - Mean faces export: global and per-class mean faces saved to disk.
  - Visualizations: K-eigenfaces grid and composite test-vs-matches gallery.
  - Multiple classification strategies: Nearest Neighbor, Majority Voting, Distance-Weighted Voting, and Class Centroids.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

from forensics_app.paths import DATA_DIR, MODELS_DIR

DEFAULT_DATA_DIR = DATA_DIR / "lfw_faces"
DEFAULT_MODEL_DIR = MODELS_DIR / "eigenfaces"


@dataclass
class FaceModel:
    """Stores the trained face subspace parameters and training data projections."""

    mean_face: np.ndarray  # Shape: (H, W)
    eigenfaces: np.ndarray  # Shape: (n_components, H, W)
    components: np.ndarray  # Shape: (n_components, D), where D = H * W
    explained_variance_ratio: np.ndarray  # Shape: (n_components,)
    X_train_pca: np.ndarray  # Shape: (n_train, n_components)
    y_train: np.ndarray  # Shape: (n_train,)
    target_names: list[str]  # List of person names
    train_image_paths: list[str]  # File paths to train images
    img_shape: tuple[int, int]  # (H, W)


def prepare_lfw_dataset(
    data_dir: Path = DEFAULT_DATA_DIR, min_faces_per_person: int = 70, resize: float = 0.4
) -> dict[str, Any]:
    """Fetch LFW dataset, split into train/test, and save images to folders with correspondence.

    Creates:
      data_dir / train / {index:04d}_{Person_Name}.png
      data_dir / test  / {index:04d}_{Person_Name}.png
      data_dir / metadata.json (records paths, labels, and target names)

    Parameters
    ----------
    data_dir : Path
        Target directory to store dataset folders.
    min_faces_per_person : int
        Minimum number of images per person (default 70).
    resize : float
        Image scale factor (default 0.4 -> 50x37 pixels).

    Returns
    -------
    dict[str, Any]
        Metadata dictionary with train/test file info and class names.
    """
    data_dir = Path(data_dir)
    metadata_path = data_dir / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                return _relocate_metadata_paths(json.load(f), data_dir)
        except Exception:
            pass

    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    # If images already exist on disk in train/ and test/, index them directly without calling fetch_lfw_people
    if train_dir.exists() and any(train_dir.glob("*.png")):
        train_files = sorted(list(train_dir.glob("*.png")))
        test_files = sorted(list(test_dir.glob("*.png"))) if test_dir.exists() else []

        # Extract person names from filenames: train_XXXX_Person_Name.png
        target_names_set: set[str] = set()
        for f in train_files:
            parts = f.stem.split("_", 2)
            if len(parts) >= 3:
                target_names_set.add(parts[2].replace("_", " "))
        target_names = sorted(list(target_names_set))
        name_to_label = {name: idx for idx, name in enumerate(target_names)}

        with Image.open(train_files[0]) as sample_img:
            h, w = sample_img.height, sample_img.width

        def _index_files(files: list[Path]) -> list[dict[str, Any]]:
            """Build metadata entries (path, label, name) from exported image filenames."""
            items: list[dict[str, Any]] = []
            for f in files:
                parts = f.stem.split("_", 2)
                p_name = parts[2].replace("_", " ") if len(parts) >= 3 else "Unknown"
                items.append(
                    {
                        "path": str(f),
                        "filename": f.name,
                        "label": name_to_label.get(p_name, 0),
                        "name": p_name,
                    }
                )
            return items

        meta = {
            "target_names": target_names,
            "n_classes": len(target_names),
            "img_shape": [h, w],
            "train": _index_files(train_files),
            "test": _index_files(test_files),
        }
        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        except Exception:
            pass
        return meta

    # Initial fallback only if no dataset images exist on disk:
    # Fetch LFW via scikit-learn helper, partition into 75% train / 25% test, and export to PNG files
    lfw = fetch_lfw_people(min_faces_per_person=min_faces_per_person, resize=resize)
    x_all = lfw.data  # (N, D)
    y_all = lfw.target  # (N,)
    target_names = list(lfw.target_names)
    images = lfw.images  # (N, H, W)
    h, w = images.shape[1], images.shape[2]

    # Split into train (75%) and test (25%) using fixed random state for reproducibility
    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y_all, test_size=0.25, random_state=42
    )

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    def _save_set(
        x_data: np.ndarray, y_data: np.ndarray, folder: Path, prefix: str
    ) -> list[dict[str, Any]]:
        """Save one split as normalized 8-bit PNGs and return its metadata entries."""
        file_list: list[dict[str, Any]] = []
        for i in range(len(x_data)):
            label = int(y_data[i])
            name = target_names[label]
            safe_name = name.replace(" ", "_")
            filename = f"{prefix}_{i:04d}_{safe_name}.png"
            file_path = folder / filename

            # Normalize 0..255 and save as 8-bit grayscale PNG
            img_arr = x_data[i].reshape((h, w))
            img_norm = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min() + 1e-8)
            img_uint8 = (img_norm * 255.0).astype(np.uint8)
            Image.fromarray(img_uint8).save(file_path)

            file_list.append(
                {
                    "path": str(file_path),
                    "filename": filename,
                    "label": label,
                    "name": name,
                }
            )
        return file_list

    train_meta = _save_set(x_train, y_train, train_dir, "train")
    test_meta = _save_set(x_test, y_test, test_dir, "test")

    meta = {
        "target_names": target_names,
        "n_classes": len(target_names),
        "img_shape": [h, w],
        "train": train_meta,
        "test": test_meta,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


def _relocate_metadata_paths(meta: dict[str, Any], data_dir: Path) -> dict[str, Any]:
    """Point stored image paths at ``data_dir`` when they do not exist as written.

    Older metadata files store paths relative to the directory the app was started
    from; this keeps them usable when it is started from somewhere else.
    """
    for split in ("train", "test"):
        for item in meta.get(split, []):
            if not Path(item["path"]).exists():
                candidate = data_dir / split / item["filename"]
                if candidate.exists():
                    item["path"] = str(candidate)
    return meta


def train_eigenfaces_model(
    n_components: int = 100,
    data_dir: Path = DEFAULT_DATA_DIR,
    model_dir: Path = DEFAULT_MODEL_DIR,
) -> FaceModel:
    """Train PCA Eigenfaces model and save mean faces to disk as required by Set 6.2.

    Computes:
      - Global mean face and per-class mean faces.
      - Top n_components eigenfaces via SVD / PCA.
      - Training projection vectors (PCA coefficients).
      - Saves mean faces into 'models/eigenfaces/mean_faces/'.
      - Saves model arrays into 'models/eigenfaces/eigenfaces_model.npz'.

    Parameters
    ----------
    n_components : int
        Number of principal components (eigenfaces) to extract (default 100).
    data_dir : Path
        Directory containing prepared LFW face images.
    model_dir : Path
        Directory to save mean faces and serialized model.

    Returns
    -------
    FaceModel
        Fitted face recognition model.
    """
    meta = prepare_lfw_dataset(data_dir)
    h, w = meta["img_shape"]
    target_names = meta["target_names"]
    train_items = meta["train"]

    # Load all training images into float64 matrix (N, D)
    n_train = len(train_items)
    d = h * w
    x_train = np.zeros((n_train, d), dtype=np.float64)
    y_train = np.zeros(n_train, dtype=np.int64)
    train_paths = [item["path"] for item in train_items]

    for i, item in enumerate(train_items):
        with Image.open(item["path"]) as img:
            arr = np.array(img.convert("L"), dtype=np.float64)
        x_train[i] = arr.flatten()
        y_train[i] = item["label"]

    # 1. Compute global mean face
    mean_face_flat = np.mean(x_train, axis=0)
    mean_face = mean_face_flat.reshape((h, w))

    # 2. Save mean faces to folder
    mean_faces_dir = model_dir / "mean_faces"
    mean_faces_dir.mkdir(parents=True, exist_ok=True)

    # Global mean face
    global_mean_uint8 = np.clip(mean_face, 0, 255).astype(np.uint8)
    Image.fromarray(global_mean_uint8).save(mean_faces_dir / "mean_face_global.png")

    # Per-class mean faces
    for class_idx, name in enumerate(target_names):
        class_mask = y_train == class_idx
        if np.any(class_mask):
            class_mean = np.mean(x_train[class_mask], axis=0).reshape((h, w))
            class_mean_uint8 = np.clip(class_mean, 0, 255).astype(np.uint8)
            safe_name = name.replace(" ", "_")
            Image.fromarray(class_mean_uint8).save(
                mean_faces_dir / f"mean_face_{safe_name}.png"
            )

    # 3. Fit PCA: finds eigenvectors of covariance matrix that maximize variance
    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        whiten=False,
        random_state=42,
    )
    pca.fit(x_train)

    # Eigenfaces: components reshaped to (n_components, H, W)
    eigenfaces = pca.components_.reshape((n_components, h, w))
    # Project all training images to obtain coefficient vectors: w = (x - mean) * V^T
    x_train_pca = pca.transform(x_train)

    model = FaceModel(
        mean_face=mean_face,
        eigenfaces=eigenfaces,
        components=pca.components_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        X_train_pca=x_train_pca,
        y_train=y_train,
        target_names=target_names,
        train_image_paths=train_paths,
        img_shape=(h, w),
    )

    save_face_model(model, model_dir)
    return model


def save_face_model(model: FaceModel, model_dir: Path = DEFAULT_MODEL_DIR) -> None:
    """Save FaceModel arrays to a compressed .npz archive."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_dir / "eigenfaces_model.npz",
        mean_face=model.mean_face,
        eigenfaces=model.eigenfaces,
        components=model.components,
        explained_variance_ratio=model.explained_variance_ratio,
        X_train_pca=model.X_train_pca,
        y_train=model.y_train,
        target_names=np.array(model.target_names),
        train_image_paths=np.array(model.train_image_paths),
        img_shape=np.array(model.img_shape),
    )


def load_face_model(model_dir: Path = DEFAULT_MODEL_DIR) -> FaceModel | None:
    """Load FaceModel from disk; returns None if file does not exist."""
    npz_path = Path(model_dir) / "eigenfaces_model.npz"
    if not npz_path.exists():
        return None
    try:
        data = np.load(npz_path)
        return FaceModel(
            mean_face=data["mean_face"],
            eigenfaces=data["eigenfaces"],
            components=data["components"],
            explained_variance_ratio=data["explained_variance_ratio"],
            X_train_pca=data["X_train_pca"],
            y_train=data["y_train"],
            target_names=list(data["target_names"]),
            train_image_paths=list(data["train_image_paths"]),
            img_shape=tuple(data["img_shape"]),
        )
    except Exception:
        return None


def make_eigenfaces_grid(
    eigenfaces: np.ndarray, k: int, n_cols: int | None = None
) -> Image.Image:
    """Render the first K eigenfaces into a balanced, square-like grid gallery with labels.

    Parameters
    ----------
    eigenfaces : np.ndarray
        Array of shape (n_components, H, W).
    k : int
        Number of top eigenfaces to display.
    n_cols : int | None
        Columns in the grid. If None, automatically set to ceil(sqrt(k)) so the
        layout is balanced and nearly square (e.g. 2x2, 3x3, 4x4, 5x5).

    Returns
    -------
    Image.Image
        Annotated PIL Image containing the grid.
    """
    k = max(1, min(k, len(eigenfaces)))
    if n_cols is None or n_cols <= 0:
        n_cols = max(1, int(np.ceil(np.sqrt(k))))
    n_rows = int(np.ceil(k / n_cols))
    h, w = eigenfaces.shape[1], eigenfaces.shape[2]

    # Magnification for thumbnail display
    thumb_w, thumb_h = int(w * 2.8), int(h * 2.8)
    pad = 10
    header_h = 22
    cell_w = thumb_w + pad * 2
    cell_h = thumb_h + pad * 2 + header_h

    grid_img = Image.new("RGB", (n_cols * cell_w, n_rows * cell_h), color=(32, 35, 42))
    draw = ImageDraw.Draw(grid_img)

    for i in range(k):
        r = i // n_cols
        c = i % n_cols
        x = c * cell_w + pad
        y = r * cell_h + pad

        ef = eigenfaces[i]
        # Normalize eigenface intensity to 0..255 for visualization
        ef_norm = (ef - ef.min()) / (ef.max() - ef.min() + 1e-8)
        ef_uint8 = (ef_norm * 255.0).astype(np.uint8)

        ef_pil = Image.fromarray(ef_uint8).convert("RGB")
        ef_pil = ef_pil.resize((thumb_w, thumb_h), Image.Resampling.NEAREST)

        grid_img.paste(ef_pil, (x, y + header_h))
        title = f"Eigenface #{i + 1}"
        draw.text((x + 2, y + 2), title, fill=(230, 230, 230))
        draw.rectangle(
            [x - 1, y + header_h - 1, x + thumb_w, y + header_h + thumb_h],
            outline=(90, 100, 120),
        )

    return grid_img


def project_face_image(
    image: Image.Image | np.ndarray, model: FaceModel, k: int
) -> np.ndarray:
    """Center and project a query image onto the first k eigenfaces.

    Computes:
      w_test = (x_test - mean_face) * components[:k].T

    Parameters
    ----------
    image : Image.Image | np.ndarray
        Input test image.
    model : FaceModel
        Trained model with mean_face and principal components.
    k : int
        Number of eigenfaces to use (1 <= k <= n_components).

    Returns
    -------
    np.ndarray
        1D feature vector of length k.
    """
    h, w = model.img_shape
    if isinstance(image, Image.Image):
        gray = image.convert("L").resize((w, h), Image.Resampling.BILINEAR)
        arr = np.array(gray, dtype=np.float64).flatten()
    else:
        import cv2

        if image.ndim == 3 and image.shape[2] == 1:
            image = image[:, :, 0]
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        elif image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.shape != (h, w):
            image = cv2.resize(image, (w, h))
        arr = image.astype(np.float64).flatten()

    mean_flat = model.mean_face.flatten()
    centered = arr - mean_flat
    sub_components = model.components[:k]  # (k, D)
    return np.dot(centered, sub_components.T)  # (k,)


def classify_face(
    test_w: np.ndarray,
    model: FaceModel,
    k: int,
    m: int,
    strategy: str = "majority",
    use_lda: bool = False,
) -> dict[str, Any]:
    """Recognize a face by comparing its feature vector against training images.

    Parameters
    ----------
    test_w : np.ndarray
        Test image PCA weight vector of length k.
    model : FaceModel
        Trained model containing training projections and labels.
    k : int
        Number of eigenfaces used.
    m : int
        Number of nearest neighbors to consider.
    strategy : str
        Classification rule: 'majority', 'distance_weighted', or 'centroid'.
    use_lda : bool
        If True, applies Linear Discriminant Analysis (Fisherfaces) onto PCA weights.

    Returns
    -------
    dict[str, Any]
        Dictionary with predicted label, name, nearest neighbors list, and strategy details.
    """
    train_w = model.X_train_pca[:, :k]  # (n_train, k)
    y_train = model.y_train
    target_names = model.target_names

    # If FisherFaces (LDA) is selected, train LDA on train_w and project both train and test
    if use_lda:
        lda = LinearDiscriminantAnalysis()
        train_proj = lda.fit_transform(train_w, y_train)
        test_proj = lda.transform(test_w.reshape(1, -1))[0]
    else:
        train_proj = train_w
        test_proj = test_w

    # Euclidean distance from query vector to every training vector
    diffs = train_proj - test_proj.reshape(1, -1)
    distances = np.linalg.norm(diffs, axis=1)

    # Sort ascending
    sorted_indices = np.argsort(distances)
    top_m_indices = sorted_indices[:m]

    top_matches: list[dict[str, Any]] = []
    for rank, idx in enumerate(top_m_indices):
        lbl = int(y_train[idx])
        top_matches.append(
            {
                "rank": rank + 1,
                "train_index": int(idx),
                "distance": float(distances[idx]),
                "label": lbl,
                "name": target_names[lbl],
                "image_path": model.train_image_paths[idx],
            }
        )

    # -------------------------------------------------------------
    # Classification Strategies
    # -------------------------------------------------------------
    if strategy == "centroid":
        # Distance to class centroids (mean weight vector per class)
        n_classes = len(target_names)
        class_centroids = np.zeros((n_classes, train_proj.shape[1]), dtype=np.float64)
        for c in range(n_classes):
            mask = y_train == c
            if np.any(mask):
                class_centroids[c] = np.mean(train_proj[mask], axis=0)

        centroid_distances = np.linalg.norm(class_centroids - test_proj.reshape(1, -1), axis=1)
        predicted_label = int(np.argmin(centroid_distances))
        strategy_info = {
            "type": "Class Centroids",
            "centroid_distances": {
                target_names[c]: float(centroid_distances[c]) for c in range(n_classes)
            },
        }

    elif strategy == "distance_weighted":
        # Distance-weighted voting: weight = 1 / (distance + 1e-6)
        class_weights: dict[int, float] = {}
        for match in top_matches:
            c = match["label"]
            weight = 1.0 / (match["distance"] + 1e-6)
            class_weights[c] = class_weights.get(c, 0.0) + weight

        predicted_label = max(class_weights.items(), key=lambda item: item[1])[0]
        strategy_info = {
            "type": "Distance-Weighted Voting",
            "weights": {target_names[c]: float(w) for c, w in class_weights.items()},
        }

    else:
        # Default: Majority voting among m nearest neighbors
        class_counts: dict[int, int] = {}
        for match in top_matches:
            c = match["label"]
            class_counts[c] = class_counts.get(c, 0) + 1

        # Tie-breaker: neighbor with smallest distance
        best_count = -1
        predicted_label = top_matches[0]["label"]
        for match in top_matches:
            c = match["label"]
            if class_counts[c] > best_count:
                best_count = class_counts[c]
                predicted_label = c

        strategy_info = {
            "type": "Majority Voting (m-NN)",
            "vote_counts": {target_names[c]: class_counts[c] for c in class_counts},
        }

    return {
        "predicted_label": predicted_label,
        "predicted_name": target_names[predicted_label],
        "top_matches": top_matches,
        "strategy_info": strategy_info,
        "use_lda": use_lda,
        "k": k,
        "m": m,
    }


def make_recognition_gallery(
    test_image: Image.Image,
    top_matches: list[dict[str, Any]],
    predicted_name: str,
    strategy: str,
    true_name: str | None = None,
) -> Image.Image:
    """Build a side-by-side gallery showing the test image and its top-m matches.

    Parameters
    ----------
    test_image : Image.Image
        Query face image.
    top_matches : list[dict[str, Any]]
        List of match dictionaries containing name, distance, and image_path.
    predicted_name : str
        Predicted person name.
    strategy : str
        Classification strategy used.
    true_name : str | None
        Ground truth identity if known from test set filename.

    Returns
    -------
    Image.Image
        Composite comparison image with clean headers and labels.
    """
    m = len(top_matches)
    thumb_w, thumb_h = 130, 175
    banner_h = 40
    footer_h = 42
    pad = 12

    total_w = pad + (m + 1) * (thumb_w + pad)
    total_h = banner_h + pad + thumb_h + footer_h + pad

    gallery = Image.new("RGB", (total_w, total_h), color=(26, 28, 34))
    draw = ImageDraw.Draw(gallery)

    # Top banner with verdict
    is_correct = (true_name is not None) and (predicted_name == true_name)
    if true_name is not None:
        banner_bg = (35, 95, 45) if is_correct else (120, 40, 40)
        status_text = (
            f"PREDICTED: {predicted_name} | TRUE: {true_name} ({'CORRECT' if is_correct else 'MISMATCH'})"
        )
    else:
        banner_bg = (40, 50, 75)
        status_text = f"PREDICTED: {predicted_name}  [{strategy}]"

    draw.rectangle([0, 0, total_w, banner_h], fill=banner_bg)
    draw.text((pad, 12), status_text, fill=(255, 255, 255))

    # Helper to paste thumbnail with label
    def _paste_face(img: Image.Image, x_pos: int, title: str, subtitle: str, border_color: tuple[int, int, int]) -> None:
        """Paste one resized face thumbnail with its border and two caption lines."""
        resized = img.convert("RGB").resize((thumb_w, thumb_h), Image.Resampling.BILINEAR)
        y_pos = banner_h + pad
        gallery.paste(resized, (x_pos, y_pos))
        draw.rectangle([x_pos - 1, y_pos - 1, x_pos + thumb_w, y_pos + thumb_h], outline=border_color, width=2)
        draw.text((x_pos, y_pos + thumb_h + 4), title, fill=(240, 240, 240))
        draw.text((x_pos, y_pos + thumb_h + 20), subtitle, fill=(180, 190, 205))

    # Column 0: Query/Test Face
    query_title = "Query (Test)"
    query_sub = true_name[:16] if true_name else "Unknown Identity"
    _paste_face(test_image, pad, query_title, query_sub, border_color=(255, 215, 0))

    # Columns 1..m: Top Matched Faces
    for i, match in enumerate(top_matches):
        match_x = pad + (i + 1) * (thumb_w + pad)
        try:
            with Image.open(match["image_path"]) as m_img:
                m_pil = m_img.copy()
        except Exception:
            m_pil = Image.new("RGB", (thumb_w, thumb_h), color=(50, 50, 50))

        m_title = f"Match #{match['rank']}: {match['name'].split()[-1]}"
        m_sub = f"d = {match['distance']:.2f}"
        b_color = (80, 220, 100) if match["name"] == predicted_name else (120, 120, 130)
        _paste_face(m_pil, match_x, m_title, m_sub, border_color=b_color)

    return gallery

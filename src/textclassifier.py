#Okay this is our classifier document.# src/textclassifier.py
import os
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# folder names in your dataset root
SPLITS = {
    "train": "CVPR_2024_dataset_Train",
    "val":   "CVPR_2024_dataset_Val",
    "test":  "CVPR_2024_dataset_Test",
}

# class folders exactly as they appear
CLASSES = ["Black", "Blue", "Green", "TTR"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def filename_to_text(p: Path) -> str:
    # greasy_pizza_box.jpg -> "greasy pizza box"
    return p.stem.replace("_", " ").lower()

def load_split(data_root: Path, split: str):
    split_dir = data_root / SPLITS[split]
    X, y = [], []
    for cls in CLASSES:
        cls_dir = split_dir / cls
        if not cls_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {cls_dir}")
        for p in cls_dir.rglob("*"):
            if p.suffix.lower() in IMG_EXTS:
                X.append(filename_to_text(p))
                y.append(CLASS_TO_IDX[cls])
    return X, y

def main():
    data_root = Path(os.environ["GARBAGE_DATA_ROOT"]).expanduser()
    print("Using data root:", data_root)

    X_train, y_train = load_split(data_root, "train")
    X_val, y_val     = load_split(data_root, "val")
    X_test, y_test   = load_split(data_root, "test")

    print(f"Train={len(X_train)}  Val={len(X_val)}  Test={len(X_test)}")

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=3,
            max_features=200_000
        )),
        ("clf", SGDClassifier(
            loss="log_loss",
            alpha=1e-5,
            max_iter=30,
            n_jobs=-1,
            class_weight="balanced",
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    print("\nVAL accuracy:", accuracy_score(y_val, val_pred))
    print(classification_report(y_val, val_pred, target_names=CLASSES))

    test_pred = model.predict(X_test)
    print("\nTEST accuracy:", accuracy_score(y_test, test_pred))
    print(classification_report(y_test, test_pred, target_names=CLASSES))
    print("Confusion matrix:\n", confusion_matrix(y_test, test_pred))

if __name__ == "__main__":
    main()
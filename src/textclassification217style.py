import os, re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn


def main():
    #Configuring the same path as before and defining the needed classes
    DATA_ROOT = Path(os.environ["GARBAGE_DATA_ROOT"]).expanduser()

    SPLITS = {
        "train": "CVPR_2024_dataset_Train",
        "val":   "CVPR_2024_dataset_Val",
        "test":  "CVPR_2024_dataset_Test",
    }

    CLASSES = ["Black", "Blue", "Green", "TTR"]
    CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

    print("DATA_ROOT:", DATA_ROOT)

    # Working with ensuring that stripping of the text itself
    def filename_to_text(p: Path) -> str:
        s = p.stem.lower().replace("_", " ")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def load_split(split: str):
        split_dir = DATA_ROOT / SPLITS[split]
        X_text, y = [], []
        for cls in CLASSES:
            cls_dir = split_dir / cls
            for p in cls_dir.rglob("*"):
                if p.suffix.lower() in IMG_EXTS:
                    X_text.append(filename_to_text(p))
                    y.append(CLASS_TO_IDX[cls])
        return X_text, np.array(y, dtype=np.int64)

    # Time to load in the data and set with the labels for y
    X_train, y_train = load_split("train")
    X_val,   y_val   = load_split("val")
    X_test,  y_test  = load_split("test")

    print("Sizes:", len(X_train), len(X_val), len(X_test))
    print("Example:", X_train[0], y_train[0], CLASSES[y_train[0]])

    # Copying almost the same approac with the Tfidf vectorizer and playing with the parameters
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_features=200_000
    )

    Xtr = vectorizer.fit_transform(X_train)  # scipy sparse CSR
    Xva = vectorizer.transform(X_val)
    Xte = vectorizer.transform(X_test)

    print("Xtr:", Xtr.shape, "nnz:", Xtr.nnz)

    # Reference to chat CHATGPT (so helpful here) as we move scipy sparse to torch sparse
    def scipy_csr_to_torch_sparse(csr):
        coo = csr.tocoo()
        indices = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
        values  = torch.tensor(coo.data, dtype=torch.float32)
        shape   = torch.Size(coo.shape)
        return torch.sparse_coo_tensor(indices, values, shape).coalesce()

    Xtr_t = scipy_csr_to_torch_sparse(Xtr)
    Xva_t = scipy_csr_to_torch_sparse(Xva)
    Xte_t = scipy_csr_to_torch_sparse(Xte)

    ytr_t = torch.tensor(y_train, dtype=torch.long)
    yva_t = torch.tensor(y_val, dtype=torch.long)
    yte_t = torch.tensor(y_test, dtype=torch.long)

    print("Torch sparse:", Xtr_t.shape, "nnz:", Xtr_t._nnz())

    # Now building the PyTorch Linear Classifier
    num_features = Xtr_t.shape[1]
    num_classes = len(CLASSES)

    model = nn.Linear(num_features, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    def accuracy_from_logits(logits, y):
        return (logits.argmax(dim=1) == y).float().mean().item()

    # Training our model and setting our EPOCH that we wanted
    for epoch in range(1, 21):
        model.train()
        optimizer.zero_grad()

        logits = torch.sparse.mm(Xtr_t, model.weight.t()) + model.bias
        loss = loss_fn(logits, ytr_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = torch.sparse.mm(Xva_t, model.weight.t()) + model.bias
            val_loss = loss_fn(val_logits, yva_t).item()
            val_acc = accuracy_from_logits(val_logits, yva_t)

        if epoch in [1, 2, 3, 5, 10, 20]:
            print(f"Epoch {epoch:02d} | train loss {loss.item():.4f} | val loss {val_loss:.4f} | val acc {val_acc:.4f}")

    # Now testing our evaluation
    model.eval()
    with torch.no_grad():
        test_logits = torch.sparse.mm(Xte_t, model.weight.t()) + model.bias
        test_preds = test_logits.argmax(dim=1).cpu().numpy()

    print("\nTEST REPORT")
    print(classification_report(y_test, test_preds, target_names=CLASSES))
    print("Confusion matrix:\n", confusion_matrix(y_test, test_preds))


if __name__ == "__main__":
    main()
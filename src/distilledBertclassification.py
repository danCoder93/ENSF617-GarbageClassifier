import os, re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# Configuration of our path
DATA_ROOT = Path(os.environ["GARBAGE_DATA_ROOT"]).expanduser()

SPLITS = {
    "train": "CVPR_2024_dataset_Train",
    "val":   "CVPR_2024_dataset_Val",
    "test":  "CVPR_2024_dataset_Test",
}

CLASSES = ["Black", "Blue", "Green", "TTR"]
CLASS_TO_IDX = {c:i for i,c in enumerate(CLASSES)}
IDX_TO_CLASS = {i:c for c,i in CLASS_TO_IDX.items()}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def filename_to_text(p: Path) -> str:
    s = p.stem.lower().replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_split(split: str):
    split_dir = DATA_ROOT / SPLITS[split]
    texts, labels = [], []

    for cls in CLASSES:
        cls_dir = split_dir / cls
        for p in cls_dir.rglob("*"):
            if p.suffix.lower() in IMG_EXTS:
                texts.append(filename_to_text(p))
                labels.append(CLASS_TO_IDX[cls])

    return texts, np.array(labels, dtype=np.int64)


train_texts, train_labels = load_split("train")
val_texts, val_labels     = load_split("val")
test_texts, test_labels   = load_split("test")

print("Sizes:", len(train_texts), len(val_texts), len(test_texts))
print("Example:", train_texts[0], train_labels[0], IDX_TO_CLASS[int(train_labels[0])])


# Setting up our tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=32):
        self.enc = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=max_len
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        item["labels"] = self.labels[idx]
        return item


train_ds = TextDataset(train_texts, train_labels, tokenizer, max_len=32)
val_ds   = TextDataset(val_texts, val_labels, tokenizer, max_len=32)
test_ds  = TextDataset(test_texts, test_labels, tokenizer, max_len=32)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False)


# Model Set up
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(CLASSES)
).to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 3
total_steps = num_epochs * len(train_loader)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

def eval_loader(loader):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            total_loss += out.loss.item()
            preds = out.logits.argmax(dim=1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return total_loss / max(1, len(loader)), all_preds, all_labels


# Training our model
best_val_acc = -1.0
for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0

    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

    val_loss, val_preds, val_y = eval_loader(val_loader)
    val_acc = accuracy_score(val_y, val_preds)

    print(f"Epoch {epoch} | train loss {running_loss/len(train_loader):.4f} | val loss {val_loss:.4f} | val acc {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "outputs_text_distilbert_best.pth")
        print("  saved best checkpoint")


# Testing our model
model.load_state_dict(torch.load("outputs_text_distilbert_best.pth", map_location=device))
test_loss, test_preds, test_y = eval_loader(test_loader)

print("\nTEST acc:", accuracy_score(test_y, test_preds))
print(classification_report(test_y, test_preds, target_names=CLASSES))
print("Confusion matrix:\n", confusion_matrix(test_y, test_preds))
import torch

class DataCollate():

    @staticmethod
    def collate_image(batch):
        images = torch.stack([b["image"] for b in batch])
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        paths  = [b["path"] for b in batch]
        return {"image": images, "labels": labels, "paths": paths}

    @staticmethod
    def make_collate_text(tokenizer, max_length: int):
        def collate_text(batch):
            texts  = [b["text"] for b in batch]
            labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
            paths  = [b["path"] for b in batch]

            enc = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": labels,
                "paths": paths,
            }
        return collate_text

    @staticmethod
    def make_collate_multimodal(tokenizer, max_length: int):
        def collate_mm(batch):
            images = torch.stack([b["image"] for b in batch])
            texts  = [b["text"] for b in batch]
            labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
            paths  = [b["path"] for b in batch]

            enc = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            return {
                "image": images,
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": labels,
                "paths": paths,
            }
        return collate_mm
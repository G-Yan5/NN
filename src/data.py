from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import RobertaTokenizerFast
import torchvision.transforms as T


@dataclass
class Batch:
    pixel_values: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    label: torch.Tensor


class YelpMultimodalDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        images_dir: str,
        split: str,
        tokenizer_name: str,
        max_length: int,
        image_size: int = 224,
        image_col: str = "image_path",
        text_col: str = "text",
        label_col: str = "label",
        split_col: str = "split",
    ) -> None:
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df[split_col] == split].reset_index(drop=True)

        self.images_dir = images_dir
        self.image_col = image_col
        self.text_col = text_col
        self.label_col = label_col

        self.tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_name)
        self.max_length = int(max_length)

        # ImageNet normalize
        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        # Load image
        img_path = os.path.join(self.images_dir, str(row[self.image_col]))
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.transform(image)

        # Tokenize text
        text = str(row[self.text_col]) if self.text_col in row else ""
        tok = self.tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )

        # Read label
        label = int(row[self.label_col])

        return {
            "pixel_values": pixel_values,
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
            "label": label,
        }


class Collator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, batch: List[Dict[str, Any]]) -> Batch:
        # Stack images
        pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)

        # Pad text
        input_ids_list = [torch.tensor(b["input_ids"], dtype=torch.long) for b in batch]
        attn_list = [torch.tensor(b["attention_mask"], dtype=torch.long) for b in batch]
        max_len = max(x.size(0) for x in input_ids_list)

        input_ids = torch.full((len(batch), max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

        for i, (ids, attn) in enumerate(zip(input_ids_list, attn_list)):
            L = ids.size(0)
            input_ids[i, :L] = ids
            attention_mask[i, :L] = attn

        # Stack labels
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)

        return Batch(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, label=labels)

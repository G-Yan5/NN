from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm
from transformers import RobertaModel


class RobertaTextEncoder(nn.Module):
    def __init__(self, name: str, dropout: float = 0.1) -> None:
        super().__init__()
        self.model = RobertaModel.from_pretrained(name)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # CLS embedding
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        cls = out.last_hidden_state[:, 0, :]
        return self.dropout(cls)


class EfficientNetImageEncoder(nn.Module):
    def __init__(self, name: str = "efficientnet_b0", pretrained: bool = True, dropout: float = 0.1) -> None:
        super().__init__()
        if not hasattr(tvm, name):
            raise ValueError(f"Unknown EfficientNet: {name}")

        # Load weights (if available)
        weights = None
        if pretrained:
            weights_enum_name = f"{name.upper()}_Weights"
            if hasattr(tvm, weights_enum_name):
                weights = getattr(tvm, weights_enum_name).DEFAULT

        self.backbone = getattr(tvm, name)(weights=weights)

        # Remove classifier
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.out_dim = in_features
        self.dropout = nn.Dropout(dropout)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Image features
        feats = self.backbone(pixel_values)
        return self.dropout(feats)


class FusionClassifier(nn.Module):
    def __init__(
        self,
        text_encoder: RobertaTextEncoder,
        image_encoder: EfficientNetImageEncoder,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder

        # Text + image dims
        text_dim = self.text_encoder.model.config.hidden_size
        img_dim = self.image_encoder.out_dim
        in_dim = text_dim + img_dim

        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        # Encode + fuse
        t = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        v = self.image_encoder(pixel_values=pixel_values)
        x = torch.cat([t, v], dim=-1)

        logits = self.head(x)
        return logits.squeeze(1)

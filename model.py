"""
model.py

Encoder-only Transformer for isolated ASL sign classification.
Input:  landmark sequences of shape (B, T, F)
Output: class logits of shape (B, num_classes)

Architecture:
    Linear projection → Positional encoding → N × TransformerEncoderLayer
    → Mean pool (masked) → Dropout → Linear classifier
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """Standard fixed sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ASLTransformer(nn.Module):
    """
    Encoder-only Transformer for sign classification.

    Args:
        input_dim:   Feature dimension per frame (138 with our landmark layout)
        d_model:     Internal transformer dimension
        num_heads:   Attention heads (must divide d_model)
        num_layers:  Number of encoder layers
        ffn_dim:     Feed-forward hidden dimension
        num_classes: Number of sign classes
        max_len:     Maximum sequence length (for positional encoding)
        dropout:     Dropout rate
    """

    def __init__(
        self,
        input_dim: int = 138,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        ffn_dim: int = 512,
        num_classes: int = 100,
        max_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Project raw landmarks into model dimension
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
        )

        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,   # (B, T, F) convention
            norm_first=True,    # Pre-norm (more stable training)
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:                     (B, T, input_dim)
            src_key_padding_mask:  (B, T) bool, True = padding position

        Returns:
            logits: (B, num_classes)
        """
        x = self.input_proj(x)           # (B, T, d_model)
        x = self.pos_enc(x)              # (B, T, d_model)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # (B, T, d_model)

        # Masked mean pooling — don't average over padding tokens
        if src_key_padding_mask is not None:
            valid_mask = ~src_key_padding_mask  # (B, T), True = valid
            x = x * valid_mask.unsqueeze(-1).float()
            lengths = valid_mask.sum(dim=1, keepdim=True).clamp(min=1).float()
            pooled = x.sum(dim=1) / lengths    # (B, d_model)
        else:
            pooled = x.mean(dim=1)             # (B, d_model)

        return self.classifier(pooled)         # (B, num_classes)


def build_model(num_classes: int, **kwargs) -> ASLTransformer:
    """Convenience constructor. Pass any ASLTransformer kwargs to override defaults."""
    return ASLTransformer(num_classes=num_classes, **kwargs)


if __name__ == "__main__":
    # Quick sanity check
    B, T, F = 4, 128, 138
    model = build_model(num_classes=100)
    x = torch.randn(B, T, F)
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[:, 100:] = True  # pretend last 28 frames are padding

    logits = model(x, mask)
    print(f"Input:  {x.shape}")
    print(f"Output: {logits.shape}")   # should be (4, 100)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
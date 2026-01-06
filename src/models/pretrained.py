"""Lightweight wrapper to use a pretrained wav2vec2 encoder and a small classifier head.

Usage:
    from src.models.pretrained import Wav2Vec2Classifier
    model = Wav2Vec2Classifier(model_name='facebook/wav2vec2-base-960h', freeze=True)

The model expects waveform tensor of shape (batch, samples) and returns logits (batch, 1).
"""
from typing import Optional

import torch
import torch.nn as nn

try:
    from transformers import Wav2Vec2Model
except Exception as e:
    Wav2Vec2Model = None


class Wav2Vec2Classifier(nn.Module):
    def __init__(self, model_name: str = "facebook/wav2vec2-base-960h", freeze: bool = True, hidden_dim: int = 256):
        super().__init__()
        if Wav2Vec2Model is None:
            raise RuntimeError("transformers library is required for pretrained models. Install 'transformers'.")

        self.encoder = Wav2Vec2Model.from_pretrained(model_name)
        embed_dim = self.encoder.config.hidden_size

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.pool = lambda x: x.mean(dim=1)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor):
        """x: (batch, samples)
        returns logits (batch, 1)
        """
        # Wav2Vec2Model expects input values in shape (batch, seq)
        if x.dim() == 2:
            input_values = x
        else:
            # flatten leading dims to (batch, seq)
            input_values = x.view(x.size(0), -1)

        outputs = self.encoder(input_values).last_hidden_state  # (batch, seq, hidden)
        pooled = self.pool(outputs)  # (batch, hidden)
        logits = self.classifier(pooled)
        return logits

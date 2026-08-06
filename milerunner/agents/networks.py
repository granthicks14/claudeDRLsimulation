"""Neural-network feature extractors for architecture search.

Provides a family of interchangeable torch feature extractors compatible with
Stable-Baselines3 policies: a plain MLP, a 1D-CNN, a self-attention /
Transformer encoder, and a GRU "memory" encoder. The observation vector is
tokenised into feature groups so attention and recurrence have a sequence to
operate over. Combined with :mod:`sb3_contrib`'s recurrent policies (LSTM over
time), this gives the evolutionary search a real space of architectures to
explore.

The evolutionary layer selects among these by name and mutates their widths /
depths, so no single architecture is privileged by hand.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

try:
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
except Exception:  # pragma: no cover
    class BaseFeaturesExtractor(nn.Module):  # minimal stand-in for tests
        def __init__(self, observation_space, features_dim: int = 64):
            super().__init__()
            self._features_dim = features_dim

        @property
        def features_dim(self) -> int:
            return self._features_dim


def _obs_size(observation_space) -> int:
    return int(np.prod(observation_space.shape))


class MLPExtractor(BaseFeaturesExtractor):
    """Standard multilayer perceptron."""

    def __init__(self, observation_space, features_dim: int = 256,
                 hidden: int = 256, depth: int = 2, activation: str = "relu"):
        super().__init__(observation_space, features_dim)
        act = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU,
               "silu": nn.SiLU}.get(activation, nn.ReLU)
        layers = [nn.Linear(_obs_size(observation_space), hidden), act()]
        for _ in range(max(depth - 1, 0)):
            layers += [nn.Linear(hidden, hidden), act()]
        layers += [nn.Linear(hidden, features_dim), act()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNN1DExtractor(BaseFeaturesExtractor):
    """Treat the observation as a 1D signal and convolve over it."""

    def __init__(self, observation_space, features_dim: int = 256,
                 channels: int = 32, kernel: int = 5):
        super().__init__(observation_space, features_dim)
        n = _obs_size(observation_space)
        pad = kernel // 2
        self.conv = nn.Sequential(
            nn.Conv1d(1, channels, kernel, padding=pad), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel, padding=pad), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(channels * 8, features_dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)              # (B, 1, N)
        return self.head(self.conv(x))


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerExtractor(BaseFeaturesExtractor):
    """Self-attention encoder over tokenised observation groups."""

    def __init__(self, observation_space, features_dim: int = 256,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 token_size: int = 6):
        super().__init__(observation_space, features_dim)
        n = _obs_size(observation_space)
        self.token_size = token_size
        self.n_tokens = math.ceil(n / token_size)
        self.pad = self.n_tokens * token_size - n
        self.embed = nn.Linear(token_size, d_model)
        self.pos = _PositionalEncoding(d_model, self.n_tokens + 1)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model * 2,
                                           batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Sequential(nn.Linear(d_model, features_dim), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        if self.pad:
            x = torch.cat([x, x.new_zeros(b, self.pad)], dim=1)
        tokens = x.view(b, self.n_tokens, self.token_size)
        tokens = self.embed(tokens)
        cls = self.cls.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.pos(tokens)
        enc = self.encoder(tokens)
        return self.head(enc[:, 0])     # CLS token summary


class GRUMemoryExtractor(BaseFeaturesExtractor):
    """GRU applied across observation tokens as a lightweight memory module."""

    def __init__(self, observation_space, features_dim: int = 256,
                 hidden: int = 128, token_size: int = 6):
        super().__init__(observation_space, features_dim)
        n = _obs_size(observation_space)
        self.token_size = token_size
        self.n_tokens = math.ceil(n / token_size)
        self.pad = self.n_tokens * token_size - n
        self.gru = nn.GRU(token_size, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, features_dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        if self.pad:
            x = torch.cat([x, x.new_zeros(b, self.pad)], dim=1)
        tokens = x.view(b, self.n_tokens, self.token_size)
        out, h = self.gru(tokens)
        return self.head(h[-1])


EXTRACTORS: Dict[str, type] = {
    "mlp": MLPExtractor,
    "cnn": CNN1DExtractor,
    "transformer": TransformerExtractor,
    "attention": TransformerExtractor,
    "gru": GRUMemoryExtractor,
    "memory": GRUMemoryExtractor,
}


def make_extractor_kwargs(arch: str, features_dim: int, extra: Dict | None = None) -> Dict:
    """Build the ``policy_kwargs`` fragment selecting a feature extractor."""
    extra = extra or {}
    cls = EXTRACTORS.get(arch, MLPExtractor)
    kwargs = {"features_dim": features_dim}
    kwargs.update(extra)
    return {"features_extractor_class": cls, "features_extractor_kwargs": kwargs}

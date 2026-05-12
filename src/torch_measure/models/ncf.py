# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Neural Collaborative Filter (NCF) that predicts response matrix entries."""

import math

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

from torch_measure.models._network import MLP


class NCF(nn.Module):
    """Neural Collaborative Filter predictive model.

    A neural network model to predict response matrix entries.

    Architecture:
    - Sentence embeddings for both subject and item content
    - Small MLP head trained offline on training data

    Parameters
    ----------
    encoder : SentenceTransformer
        Pre-trained transformer model used to embed subject and item content.
    embedding_dim : int
        Output dimension of the encoder model.
    encode_batch_size : int
        Batch size used to embed subject and item content.
    hidden_dim : int
        Dimension of hidden layers.
    n_layers : int
        Number of layers (minimum 1).
    dropout : float
        Dropout rate between layers.
    device : str
        Device to place parameters on.
    """

    def __init__(
        self,
        encoder: SentenceTransformer,
        embedding_dim: int,
        encode_batch_size: int = 256,
        hidden_dim: int = 256,
        n_layers: int = 3,
        dropout: float = 0.1,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.encode_batch_size = encode_batch_size
        self._device = device

        self.encoder = encoder
        self.net = NCFHead(
            input_dim=embedding_dim * 2,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
        ).to(self._device)

    def _encode_single(self, subject: str, item: str) -> torch.Tensor:
        """Encode a subject-item pair."""
        u = self.encoder.encode(subject, convert_to_tensor=True, device=self._device)
        v = self.encoder.encode(item, convert_to_tensor=True, device=self._device)
        return u, v

    def _raw_prob(self, subject: str, item: str) -> float:
        """Forward pass through the NCF, returns probability in [0, 1]."""
        with torch.no_grad():
            u, v = self._encode_single(subject, item)
            x = torch.cat([u, v], dim=-1).unsqueeze(0)
            logit = self.net(x).squeeze(-1).item()
        return float(1.0 / (1.0 + math.exp(-logit)))

    def encode_batch(
        self, subjects: list[str], items: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of subject-item pairs."""
        u = self.encoder.encode(
            subjects,
            convert_to_tensor=True,
            batch_size=self.encode_batch_size,
            show_progress_bar=True,
            device=self._device,
        )
        v = self.encoder.encode(
            items,
            convert_to_tensor=True,
            batch_size=self.encode_batch_size,
            show_progress_bar=True,
            device=self._device,
        )
        return u, v

    def load_head(self, path: str) -> None:
        """Load pre-trained NCFHead weights from a state dict file."""
        state = torch.load(path, map_location=self._device, weights_only=True)
        self.net.load_state_dict(state)

    def load_embeddings(self, path: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Load pre-computed subject and item embeddings from a checkpoint file.

        Parameters
        ----------
        path : str
            Path to the embeddings checkpoint saved by ``torch.save`` with keys
            ``"subject_embeddings"`` and ``"item_embeddings"``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Subject embeddings and item embeddings, respectively.
        """
        data = torch.load(path, weights_only=True)
        return data["subject_embeddings"], data["item_embeddings"]

    def predict(self, data: dict, labeled: list[dict]) -> float:
        """Compute response probability P(subject passes item).

        Parameters
        ----------
        data : dict
            Dictionary with keys ``"subject_content"`` (str) and
            ``"item_content"`` (str) containing the raw text for the subject
            and item to score.
        labeled : list[dict]
            Previously observed subject-item-response records. Not used by
            this model; included for interface compatibility.

        Returns
        -------
        float
            Predicted probability that the subject passes the item, clipped to
            ``[1e-7, 1 - 1e-7]``.
        """
        probs = self._raw_prob(data["subject_content"], data["item_content"])
        probs = float(np.clip(probs, 1e-7, 1 - 1e-7))
        return probs


class NCFHead(nn.Module):
    """Neural Collaborative Filter Multi-Layer Perceptron head.

    Maps sentence embeddings to a unidimensional output.

    Parameters
    ----------
    input_dim : int
        Dimension of the input (concatenated subject and item embeddings).
    hidden_dim : int
        Dimension of hidden layers.
    n_layers : int
        Number of layers (minimum 1).
    dropout : float
        Dropout rate between layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.net = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            n_layers=n_layers,
            dropout=dropout,
        )

    def forward(self, x):
        """Forward pass."""
        return self.net(x).squeeze(-1)

# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Neural Collaborative Filter (NCF) that predicts response matrix entries."""

import math

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize
from scipy.special import expit as sigmoid
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

        # Calibration
        self._platt_a = 1.0
        self._platt_b = 0.0
        self._round_calibrated = False  # reset each round

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

    def _fit_platt(self, labeled: list[dict]) -> None:
        """
        Fit a one-parameter Platt scaler on revealed labels.
        Uses scipy to optimise log-loss of:
            p_calibrated = sigmoid(a * logit + b)
        where logit = logit(_raw_prob(...)).
        """
        if not labeled:
            return

        logits, ys = [], []
        for ex in labeled:
            p = self._raw_prob(ex["subject_content"], ex["item_content"])
            p = float(np.clip(p, 1e-7, 1 - 1e-7))
            logits.append(math.log(p / (1 - p)))
            ys.append(float(ex["label"]))
        logits = np.array(logits)
        ys = np.array(ys)

        def neg_log_loss(params):
            a, b = params
            probs = sigmoid(a * logits + b)
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            return -np.mean(ys * np.log(probs) + (1 - ys) * np.log(1 - probs))

        result = minimize(neg_log_loss, x0=[1.0, 0.0], method="L-BFGS-B")
        if result.success:
            self._platt_a, self._platt_b = result.x
        self._round_calibrated = True

    def encode_batch(self, subjects: list[str], items: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
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

        1. Compute raw NCF probability.
        2. On first call of a round with labels available, fit Platt scaler.
        3. Apply calibrated scaling and return.

        Parameters
        ----------
        data : dict
            Dictionary with keys ``"subject_content"`` (str) and
            ``"item_content"`` (str) containing the raw text for the subject
            and item to score.
        labeled : list[dict]
            Previously observed subject-item-response records.

        Returns
        -------
        float
            Predicted probability that the subject passes the item, clipped to
            ``[1e-7, 1 - 1e-7]``.
        """
        # Fit Platt scaler once per round (on first call with labeled data)
        if labeled and not self._round_calibrated:
            self._fit_platt(labeled)

        raw_p = self._raw_prob(data["subject_content"], data["item_content"])
        raw_p = float(np.clip(raw_p, 1e-7, 1 - 1e-7))

        if not self._round_calibrated:
            return raw_p

        # Apply Platt calibration in log-odds space
        raw_logit = math.log(raw_p / (1 - raw_p))
        cal_logit = self._platt_a * raw_logit + self._platt_b
        return float(1.0 / (1.0 + math.exp(-cal_logit)))


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

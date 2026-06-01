# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Per-category temperature + bias calibration from K revealed anchor labels.

Given a small set of revealed ``(logit, label)`` pairs grouped by category,
fit a per-category temperature ``T_c`` and bias ``b_c`` and apply

    p = sigmoid(z / T_c + bias_alpha * b_c)

to held-out logits. The per-category temperature is shrunk toward a globally
fit ``T_global`` by an informativeness gate that measures how well the K
anchors identify ``T_c``; when anchors are uninformative we fall back to the
pooled global fit. ``gate="global"`` skips the per-category fit entirely.
"""

from __future__ import annotations

from typing import Literal

import torch

Gate = Literal["global", "fisher", "label_var"]


class AnchorCalibrator:
    """Post-hoc calibrator wrapping any logit producer.

    Parameters
    ----------
    gate : {"global", "fisher", "label_var"}
        How to weight the per-category fit against the pooled global fit.

        - ``"global"``: always use the global T (default).
        - ``"fisher"``: leverage-normalised Fisher information of the T-fit
          at ``T_global``.
        - ``"label_var"``: ``ybar * (1 - ybar)`` over the category's anchors.

        Effective per-category shrinkage is
        ``lam_T_max * I / (I + kappa)``.
    lam_T_max : float
        Cap on the per-category shrinkage weight.
    kappa : float
        Half-saturation of the gate. Larger ``kappa`` shrinks harder toward
        ``T_global``. Ignored when ``gate="global"``.
    lam_bias : float
        Shrinkage of per-category bias toward the pooled global bias.
    bias_alpha : float
        Multiplier on the bias term in the final logit.
    clip_range : (float, float)
        Output probabilities clipped to this range.
    T_bounds : (float, float)
        Search bounds for the golden-section T fit.

    Examples
    --------
    >>> import torch
    >>> cal = AnchorCalibrator(gate="fisher")
    >>> _ = cal.fit(  # 5 anchors split across 2 categories
    ...     logits=torch.tensor([0.5, -0.2, 1.1, 0.0, -0.8]),
    ...     labels=torch.tensor([1, 0, 1, 1, 0]),
    ...     category=torch.tensor([0, 0, 0, 1, 1]),
    ... )
    >>> probs = cal.transform(
    ...     logits=torch.tensor([0.3, -0.5]),
    ...     category=torch.tensor([0, 1]),
    ... )
    """

    def __init__(
        self,
        gate: Gate = "global",
        lam_T_max: float = 0.7,
        kappa: float = 0.1,
        lam_bias: float = 0.7,
        bias_alpha: float = 0.5,
        clip_range: tuple[float, float] = (0.02, 0.98),
        T_bounds: tuple[float, float] = (0.1, 10.0),
    ) -> None:
        if gate not in ("global", "fisher", "label_var"):
            raise ValueError(f"Unknown gate: {gate!r}")
        self.gate = gate
        self.lam_T_max = lam_T_max
        self.kappa = kappa
        self.lam_bias = lam_bias
        self.bias_alpha = bias_alpha
        self.clip_range = clip_range
        self.T_bounds = T_bounds

        self.T_global: float = 1.0
        self.bias_global: float = 0.0
        self.T_by_cat: dict[int, float] = {}
        self.bias_by_cat: dict[int, float] = {}

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        category: torch.Tensor,
    ) -> AnchorCalibrator:
        """Fit T and bias from anchor pairs grouped by category.

        Parameters
        ----------
        logits : torch.Tensor
            Predicted logits at the anchors, shape ``(N,)``.
        labels : torch.Tensor
            Observed binary labels at the anchors, shape ``(N,)``.
        category : torch.Tensor
            Category index for each anchor, shape ``(N,)``.

        Returns
        -------
        self
        """
        z = logits.detach().flatten().to(torch.float64)
        y = labels.detach().flatten().to(torch.float64)
        c = category.detach().flatten().to(torch.long)
        if not (len(z) == len(y) == len(c)):
            raise ValueError("logits, labels, category must have equal length")

        if len(z) == 0:
            self.T_global, self.bias_global = 1.0, 0.0
            self.T_by_cat, self.bias_by_cat = {}, {}
            return self

        self.T_global = _fit_temperature(z, y, self.T_bounds)
        self.bias_global = _mean_residual(z, y, self.T_global)

        self.T_by_cat = {}
        self.bias_by_cat = {}
        for cat in torch.unique(c).tolist():
            m = c == cat
            z_c, y_c = z[m], y[m]
            T_fit = _fit_temperature(z_c, y_c, self.T_bounds)
            eff_lam = _effective_lam(z_c, y_c, self.T_global, self.gate, self.lam_T_max, self.kappa)
            self.T_by_cat[int(cat)] = eff_lam * T_fit + (1 - eff_lam) * self.T_global
            r_c = _mean_residual(z_c, y_c, self.T_by_cat[int(cat)])
            self.bias_by_cat[int(cat)] = self.lam_bias * r_c + (1 - self.lam_bias) * self.bias_global
        return self

    def transform(self, logits: torch.Tensor, category: torch.Tensor) -> torch.Tensor:
        """Apply the fitted calibration to held-out logits.

        Categories not seen during ``fit`` fall back to ``(T_global, 0)``.
        """
        z = logits.flatten().to(torch.float64)
        c = category.flatten().to(torch.long)
        T = torch.full_like(z, self.T_global)
        b = torch.zeros_like(z)
        for cat in torch.unique(c).tolist():
            cat = int(cat)
            if cat in self.T_by_cat:
                m = c == cat
                T[m] = self.T_by_cat[cat]
                b[m] = self.bias_by_cat[cat]
        p = torch.sigmoid(z / T + self.bias_alpha * b)
        lo, hi = self.clip_range
        return p.clamp(lo, hi).to(logits.dtype).reshape(logits.shape)

    def fit_transform(
        self,
        anchor_logits: torch.Tensor,
        anchor_labels: torch.Tensor,
        anchor_category: torch.Tensor,
        logits: torch.Tensor,
        category: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience: ``fit(anchors).transform(query)``."""
        return self.fit(anchor_logits, anchor_labels, anchor_category).transform(logits, category)


# ---------------------------------------------------------------------------
# Internal kernels
# ---------------------------------------------------------------------------


def _nll(z: torch.Tensor, y: torch.Tensor, T: float) -> float:
    p = torch.sigmoid(z / T).clamp(1e-9, 1 - 1e-9)
    return float(-(y * torch.log(p) + (1 - y) * torch.log(1 - p)).sum())


def _fit_temperature(z: torch.Tensor, y: torch.Tensor, bounds: tuple[float, float]) -> float:
    """Golden-section search on log-T to minimise Bernoulli NLL."""
    import math

    if len(z) == 0:
        return 1.0
    lo, hi = math.log(bounds[0]), math.log(bounds[1])
    phi = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = _nll(z, y, math.exp(c)), _nll(z, y, math.exp(d))
    for _ in range(40):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = _nll(z, y, math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = _nll(z, y, math.exp(d))
    return float(math.exp((a + b) / 2))


def _mean_residual(z: torch.Tensor, y: torch.Tensor, T: float) -> float:
    """Mean (observed_logit - predicted_logit) over the anchors.

    Labels are clipped to [0.05, 0.95] before logit to keep magnitudes bounded.
    """
    import math

    obs = torch.where(y > 0.5, torch.tensor(math.log(0.95 / 0.05)), torch.tensor(math.log(0.05 / 0.95))).to(z.dtype)
    return float((obs - z / T).mean())


def _effective_lam(
    z: torch.Tensor,
    y: torch.Tensor,
    T_global: float,
    gate: Gate,
    lam_T_max: float,
    kappa: float,
) -> float:
    """Effective per-category shrinkage weight ``lam_T_max * I / (I + kappa)``.

    - ``global``: returns 0 (always pick T_global).
    - ``fisher``: leverage-normalised Fisher information at T_global.
    - ``label_var``: ``ybar * (1 - ybar)``.
    """
    if gate == "global" or kappa <= 0 or len(z) == 0:
        return 0.0 if gate == "global" else lam_T_max
    if gate == "label_var":
        ybar = float(y.mean())
        info = ybar * (1.0 - ybar)
    else:  # fisher
        p = torch.sigmoid(z / T_global)
        leverage = (z / T_global) ** 2
        ref = float((leverage * 0.25).sum())
        info = float((leverage * p * (1 - p)).sum()) / ref if ref > 1e-12 else 0.0
    return lam_T_max * info / (info + kappa)

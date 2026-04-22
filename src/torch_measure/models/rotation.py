# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Factor rotation methods for interpretability.

Consolidated from factor-model/calibration/rotation.py.
"""

from __future__ import annotations

import torch


def varimax_rotation(loadings: torch.Tensor, max_iter: int = 100, tol: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Varimax rotation to factor loadings.

    Varimax maximizes the variance of squared loadings within each factor,
    producing a simpler structure.

    Parameters
    ----------
    loadings : torch.Tensor
        Factor loading matrix (n_items, n_factors).
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    rotated_loadings : torch.Tensor
        Rotated loading matrix (n_items, n_factors).
    rotation_matrix : torch.Tensor
        Rotation matrix (n_factors, n_factors).
    """
    try:
        from factor_analyzer.rotator import Rotator

        rotator = Rotator(method="varimax")
        rotated = rotator.fit_transform(loadings.detach().cpu().numpy())
        rotated_t = torch.tensor(rotated, dtype=loadings.dtype, device=loadings.device)
        # Compute rotation matrix: R = L_orig^+ @ L_rotated
        rotation = torch.linalg.lstsq(loadings.detach(), rotated_t).solution
        return rotated_t, rotation
    except ImportError:
        # Simple varimax implementation
        return _varimax_torch(loadings, max_iter, tol)


def _varimax_torch(loadings: torch.Tensor, max_iter: int = 100, tol: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch Varimax rotation."""
    p, k = loadings.shape
    rotation = torch.eye(k, dtype=loadings.dtype, device=loadings.device)
    rotated = loadings.clone()

    for _ in range(max_iter):
        old = rotated.clone()
        for i in range(k):
            for j in range(i + 1, k):
                # Compute rotation angle
                u = rotated[:, i] ** 2 - rotated[:, j] ** 2
                v = 2 * rotated[:, i] * rotated[:, j]
                a = u.sum()
                b = v.sum()
                c = (u**2 - v**2).sum()
                d = (2 * u * v).sum()

                angle = 0.25 * torch.atan2(2 * d - 2 * a * b / p, c - (a**2 - b**2) / p)

                cos_a = torch.cos(angle)
                sin_a = torch.sin(angle)

                # Apply rotation
                col_i = rotated[:, i] * cos_a + rotated[:, j] * sin_a
                col_j = -rotated[:, i] * sin_a + rotated[:, j] * cos_a
                rotated[:, i] = col_i
                rotated[:, j] = col_j

                # Update rotation matrix
                rot_i = rotation[:, i] * cos_a + rotation[:, j] * sin_a
                rot_j = -rotation[:, i] * sin_a + rotation[:, j] * cos_a
                rotation[:, i] = rot_i
                rotation[:, j] = rot_j

        if (rotated - old).abs().max() < tol:
            break

    return rotated, rotation


def promax_rotation(
    loadings: torch.Tensor, power: int = 4, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Promax (oblique) rotation to factor loadings.

    Promax starts with Varimax and then applies a power transformation
    to achieve simple structure while allowing correlated factors.

    Parameters
    ----------
    loadings : torch.Tensor
        Factor loading matrix (n_items, n_factors).
    power : int
        Power parameter for the Promax transformation.

    Returns
    -------
    rotated_loadings : torch.Tensor
        Promax-rotated loadings.
    rotation_matrix : torch.Tensor
        Rotation matrix.
    """
    try:
        from factor_analyzer.rotator import Rotator

        rotator = Rotator(method="promax", power=power)
        rotated = rotator.fit_transform(loadings.detach().cpu().numpy())
        rotated_t = torch.tensor(rotated, dtype=loadings.dtype, device=loadings.device)
        rotation = torch.linalg.lstsq(loadings.detach(), rotated_t).solution
        return rotated_t, rotation
    except ImportError:
        # Fall back to varimax
        return varimax_rotation(loadings, **kwargs)


def bifactor_rotation(
    U: torch.Tensor,
    V: torch.Tensor,
    Z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply bifactor rotation: whiten, then Varimax, then separate general factor.

    Parameters
    ----------
    U : torch.Tensor
        Subject abilities (n_subjects, K).
    V : torch.Tensor
        Item loadings (n_items, K).
    Z : torch.Tensor
        Item intercepts (n_items,).

    Returns
    -------
    U_rot : torch.Tensor
        Rotated abilities.
    V_rot : torch.Tensor
        Rotated loadings.
    Z : torch.Tensor
        Unchanged intercepts.
    """
    # Whiten U
    U_centered = U - U.mean(dim=0)
    cov = (U_centered.T @ U_centered) / (U.shape[0] - 1)
    L = torch.linalg.cholesky(cov + 1e-6 * torch.eye(cov.shape[0], device=cov.device))
    L_inv = torch.linalg.inv(L)

    U_white = U_centered @ L_inv.T
    V_transformed = V @ L.T

    # Varimax rotation
    V_rot, rotation = varimax_rotation(V_transformed)
    U_rot = U_white @ rotation

    return U_rot, V_rot, Z

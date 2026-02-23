# Copyright (c) 2026 AIMS Foundation. MIT License.

import torch

from torch_measure.models.rotation import bifactor_rotation, varimax_rotation


class TestVarimaxRotation:
    def test_output_shapes(self):
        loadings = torch.randn(20, 3)
        rotated, rotation = varimax_rotation(loadings)
        assert rotated.shape == (20, 3)
        assert rotation.shape == (3, 3)

    def test_preserves_norms(self):
        """Varimax rotation should approximately preserve row norms (communalities)."""
        loadings = torch.randn(20, 3)
        rotated, _ = varimax_rotation(loadings)
        orig_norms = (loadings**2).sum(dim=1)
        rot_norms = (rotated**2).sum(dim=1)
        assert torch.allclose(orig_norms, rot_norms, atol=0.1)

    def test_identity_for_single_factor(self):
        """With a single factor, rotation should be trivial."""
        loadings = torch.randn(10, 1)
        rotated, rotation = varimax_rotation(loadings)
        assert rotated.shape == (10, 1)


class TestBifactorRotation:
    def test_output_shapes(self):
        U = torch.randn(50, 3)
        V = torch.randn(20, 3)
        Z = torch.randn(20)
        U_rot, V_rot, Z_out = bifactor_rotation(U, V, Z)
        assert U_rot.shape == (50, 3)
        assert V_rot.shape == (20, 3)
        assert torch.equal(Z, Z_out)

    def test_intercepts_unchanged(self):
        """Bifactor rotation should not change intercepts."""
        U = torch.randn(50, 3)
        V = torch.randn(20, 3)
        Z = torch.randn(20)
        _, _, Z_out = bifactor_rotation(U, V, Z)
        assert torch.equal(Z, Z_out)

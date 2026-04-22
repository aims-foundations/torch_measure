# Copyright (c) 2026 AIMS Foundations. MIT License.

import torch

from torch_measure.models import MultiFacetRasch


class TestMultiFacetRasch:
    def test_init(self):
        model = MultiFacetRasch(n_subjects=10, n_items=20, n_facet_levels=3)
        assert model.n_subjects == 10
        assert model.n_items == 20
        assert model.n_facet_levels == 3
        assert model.gamma.shape == (3,)
        assert model.tau.shape == (20, 3)
        assert model.delta.shape == (10, 3)

    def test_predict_shape(self):
        model = MultiFacetRasch(n_subjects=5, n_items=10, n_facet_levels=3)
        probs = model.predict()
        assert probs.shape == (5, 10)
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_predict_with_facet_index(self):
        model = MultiFacetRasch(n_subjects=5, n_items=10, n_facet_levels=3)
        probs_0 = model.predict(facet_indices=torch.tensor([0]))
        probs_1 = model.predict(facet_indices=torch.tensor([1]))
        assert probs_0.shape == (5, 10)
        assert probs_1.shape == (5, 10)

    def test_set_reference_level(self):
        model = MultiFacetRasch(n_subjects=5, n_items=10, n_facet_levels=3)
        model.set_reference_level(0)
        assert model.gamma_mask[0] == 0.0
        assert model.gamma_mask[1] == 1.0
        assert model.tau_mask[:, 0].sum() == 0.0

    def test_reference_level_anchors_output(self):
        """Setting reference level should anchor gamma and tau to zero for that level."""
        model = MultiFacetRasch(n_subjects=5, n_items=10, n_facet_levels=3)
        model.set_reference_level(0)
        # Gamma for level 0 should have no effect
        with torch.no_grad():
            model.gamma.fill_(5.0)
        probs_ref = model.predict(facet_indices=torch.tensor([0]))
        # The gamma contribution for level 0 should be masked out
        gamma_eff = model.gamma * model.gamma_mask
        assert gamma_eff[0] == 0.0
        assert probs_ref.shape == (5, 10)

    def test_fit_reduces_loss(self, small_response_matrix):
        model = MultiFacetRasch(n_subjects=20, n_items=30, n_facet_levels=2)
        history = model.fit(small_response_matrix, max_epochs=50, verbose=False)
        assert history["losses"][-1] < history["losses"][0]

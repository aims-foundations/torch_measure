# Copyright (c) 2026 AIMS Foundations. MIT License.

import pandas as pd
import pytest
import torch

from torch_measure.datasets._long_form import LongFormData
from torch_measure.models import DemandAssessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_query(n_subjects: int, n_rows: int, item_feature_dim: int) -> dict:
    """Build a minimal valid query for DemandAssessor."""
    return {
        "subject_idx": torch.randint(0, n_subjects, (n_rows,)),
        "item_features": torch.randn(n_rows, item_feature_dim),
    }


def _synth_longform(
    n_subjects: int = 10,
    n_items: int = 15,
    seed: int = 0,
) -> LongFormData:
    """Dense synthetic LongFormData with zero-padded IDs so sorted() == insertion order."""
    torch.manual_seed(seed)
    ability = torch.randn(n_subjects)
    difficulty = torch.randn(n_items)
    probs = torch.sigmoid(ability.unsqueeze(1) - difficulty.unsqueeze(0))
    responses = torch.bernoulli(probs)

    rows = []
    for s in range(n_subjects):
        for i in range(n_items):
            rows.append(
                {
                    "subject_id": f"s{s:02d}",
                    "item_id": f"i{i:02d}",
                    "benchmark_id": "synthetic",
                    "trial": 0,
                    "test_condition": None,
                    "response": float(responses[s, i].item()),
                    "correct_answer": None,
                    "trace": None,
                }
            )
    df = pd.DataFrame(rows)
    items = pd.DataFrame(
        [{"item_id": f"i{i:02d}", "benchmark_id": "synthetic"} for i in range(n_items)]
    )
    subjects = pd.DataFrame([{"subject_id": f"s{s:02d}"} for s in range(n_subjects)])
    return LongFormData(
        name="synthetic",
        responses=df,
        items=items,
        subjects=subjects,
        traces=None,
        info={},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDemandAssessor:

    # --- construction -------------------------------------------------------

    def test_init(self):
        model = DemandAssessor(n_subjects=10, item_feature_dim=8)
        assert model.n_subjects == 10
        assert model.item_feature_dim == 8
        assert model.subject_embedding_dim == 16  # default
        assert model.n_items == 0
        assert model.subject_embedding.weight.shape == (10, 16)

    def test_expected_keys(self):
        assert DemandAssessor.expected_keys == ("subject_idx", "item_features")

    def test_n_items_zero_before_fit(self):
        model = DemandAssessor(n_subjects=5, item_feature_dim=4)
        assert model.n_items == 0

    # --- predict() ----------------------------------------------------------

    def test_predict_shape_before_fit(self):
        """predict() is valid before fit() — parameters are randomly initialised."""
        model = DemandAssessor(n_subjects=5, item_feature_dim=4)
        query = _make_query(5, 12, 4)
        probs = model.predict(query)
        assert probs.shape == (12,)
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_predict_validates_feature_dim(self):
        model = DemandAssessor(n_subjects=5, item_feature_dim=8)
        bad_query = {
            "subject_idx": torch.zeros(3, dtype=torch.long),
            "item_features": torch.randn(3, 4),  # wrong dim
        }
        try:
            model.predict(bad_query)
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    def test_predict_known_values(self):
        """Verify forward computation with manually set weights (n_layers=1)."""
        # n_layers=1: net.net = Sequential([Linear(E+F, 1)]) — no activation.
        E, F = 2, 2
        model = DemandAssessor(
            n_subjects=2, item_feature_dim=F, subject_embedding_dim=E, n_layers=1
        )
        with torch.no_grad():
            model.subject_embedding.weight.copy_(
                torch.tensor([[1.0, 0.0], [0.0, 0.0]])
            )
            # net.net[0] is the sole Linear(E+F, 1); weight shape (1, 4)
            model.net.net[0].weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
            model.net.net[0].bias.zero_()

        # Subject 0: e_s=[1,0], f=[0,0] → x=[1,0,0,0] → logit=1 → sigmoid(1)
        # Subject 1: e_s=[0,0], f=[0,0] → x=[0,0,0,0] → logit=0 → sigmoid(0)=0.5
        query = {
            "subject_idx": torch.tensor([0, 1]),
            "item_features": torch.zeros(2, F),
        }
        probs = model.predict(query)
        expected = torch.sigmoid(torch.tensor([1.0, 0.0]))
        assert torch.allclose(probs, expected, atol=1e-5)

    def test_predict_subjects_differ(self, seed):
        """Different subjects produce different probabilities for the same item."""
        model = DemandAssessor(n_subjects=5, item_feature_dim=4)
        f = torch.randn(1, 4)
        p0 = model.predict({"subject_idx": torch.tensor([0]), "item_features": f})
        p1 = model.predict({"subject_idx": torch.tensor([1]), "item_features": f})
        assert not torch.allclose(p0, p1)

    def test_forward_equals_predict(self):
        model = DemandAssessor(n_subjects=5, item_feature_dim=4)
        query = _make_query(5, 10, 4)
        assert torch.allclose(model(query), model.predict(query))

    # --- fit() — basic ------------------------------------------------------

    def test_fit_reduces_loss(self, small_response_matrix):
        n_subjects, n_items = small_response_matrix.shape
        torch.manual_seed(0)
        features = torch.randn(n_items, 8)
        model = DemandAssessor(n_subjects=n_subjects, item_feature_dim=8)
        history = model.fit(
            small_response_matrix, features, max_epochs=100, verbose=False
        )
        assert len(history["losses"]) > 0
        assert history["losses"][-1] < history["losses"][0]

    def test_fit_updates_n_items(self, small_response_matrix):
        n_subjects, n_items = small_response_matrix.shape
        features = torch.randn(n_items, 4)
        model = DemandAssessor(n_subjects=n_subjects, item_feature_dim=4)
        model.fit(small_response_matrix, features, max_epochs=5, verbose=False)
        assert model.n_items == n_items

    def test_fit_refit_updates_n_items(self, small_response_matrix):
        """Re-fitting with a different item count correctly updates n_items."""
        n_subjects, n_items = small_response_matrix.shape
        features_30 = torch.randn(n_items, 4)
        model = DemandAssessor(n_subjects=n_subjects, item_feature_dim=4)
        model.fit(small_response_matrix, features_30, max_epochs=5, verbose=False)
        assert model.n_items == 30

        # Re-fit on a smaller matrix
        small_matrix = small_response_matrix[:, :10]
        features_10 = features_30[:10]
        model.fit(small_matrix, features_10, max_epochs=5, verbose=False)
        assert model.n_items == 10

    # --- fit() — validation -------------------------------------------------

    def test_fit_validates_feature_dim_wrong_width(self, small_response_matrix):
        n_subjects, n_items = small_response_matrix.shape
        bad_features = torch.randn(n_items, 99)  # wrong second dim
        model = DemandAssessor(n_subjects=n_subjects, item_feature_dim=8)
        with pytest.raises(ValueError):
            model.fit(small_response_matrix, bad_features, verbose=False)

    def test_fit_validates_feature_rank_1d(self, small_response_matrix):
        """1-D item_features must raise ValueError, not IndexError."""
        n_subjects, _ = small_response_matrix.shape
        features_1d = torch.randn(8)  # 1-D, not 2-D
        model = DemandAssessor(n_subjects=n_subjects, item_feature_dim=8)
        with pytest.raises(ValueError):
            model.fit(small_response_matrix, features_1d, verbose=False)

    def test_fit_validates_empty_observations_all_nan(self):
        """All-NaN response matrix must raise ValueError before training."""
        n_subjects, n_items = 5, 10
        nan_matrix = torch.full((n_subjects, n_items), float("nan"))
        features = torch.randn(n_items, 4)
        model = DemandAssessor(n_subjects=n_subjects, item_feature_dim=4)
        with pytest.raises(ValueError, match="No observed responses"):
            model.fit(nan_matrix, features, verbose=False)

    def test_fit_validates_empty_observations_all_false_mask(self):
        """All-False mask must raise ValueError before training."""
        n_subjects, n_items = 5, 10
        matrix = torch.rand(n_subjects, n_items)
        mask = torch.zeros(n_subjects, n_items, dtype=torch.bool)
        features = torch.randn(n_items, 4)
        model = DemandAssessor(n_subjects=n_subjects, item_feature_dim=4)
        with pytest.raises(ValueError, match="No observed responses"):
            model.fit(matrix, features, mask=mask, verbose=False)

    def test_fit_validates_insufficient_feature_rows(self):
        """item_features with fewer rows than items referenced raises ValueError."""
        n_subjects, n_items = 5, 10
        matrix = torch.rand(n_subjects, n_items)
        too_few = torch.randn(5, 4)  # only 5 rows, but data has 10 items
        model = DemandAssessor(n_subjects=n_subjects, item_feature_dim=4)
        with pytest.raises(ValueError):
            model.fit(matrix, too_few, verbose=False)

    def test_fit_with_explicit_mask(self, small_response_matrix):
        """Explicit boolean mask selects the correct subset of observations."""
        n_subjects, n_items = small_response_matrix.shape
        mask = torch.ones(n_subjects, n_items, dtype=torch.bool)
        mask[:, n_items // 2 :] = False  # hide second half of items
        features = torch.randn(n_items, 4)
        model = DemandAssessor(n_subjects=n_subjects, item_feature_dim=4)
        history = model.fit(
            small_response_matrix, features, mask=mask, max_epochs=10, verbose=False
        )
        assert len(history["losses"]) > 0

    def test_fit_with_nan_entries(self, response_matrix_with_nans):
        """NaN entries in the response matrix are correctly excluded."""
        n_subjects, n_items = response_matrix_with_nans.shape
        features = torch.randn(n_items, 4)
        model = DemandAssessor(n_subjects=n_subjects, item_feature_dim=4)
        history = model.fit(
            response_matrix_with_nans, features, max_epochs=10, verbose=False
        )
        assert len(history["losses"]) > 0

    def test_fit_convergence_tol(self, small_response_matrix):
        """convergence_tol stops training early when loss plateaus."""
        n_subjects, n_items = small_response_matrix.shape
        features = torch.randn(n_items, 4)
        model = DemandAssessor(n_subjects=n_subjects, item_feature_dim=4)
        history = model.fit(
            small_response_matrix,
            features,
            max_epochs=10_000,
            convergence_tol=1e-3,
            verbose=False,
        )
        assert len(history["losses"]) < 10_000

    # --- fit() — LongFormData -----------------------------------------------

    def test_fit_accepts_longform_data(self):
        """LongFormData input produces valid training history."""
        data = _synth_longform(n_subjects=10, n_items=15)
        # item_ids from to_fit_tensors() are sorted: "i00", "i01", ..., "i14"
        # which matches insertion order for zero-padded IDs.
        item_ids = data.to_fit_tensors()["item_ids"]
        n_items = len(item_ids)
        features = torch.randn(n_items, 4)
        model = DemandAssessor(n_subjects=10, item_feature_dim=4)
        history = model.fit(data, features, max_epochs=10, verbose=False)
        assert len(history["losses"]) > 0
        assert model.n_items == n_items

    def test_fit_longform_item_ordering_is_sorted(self):
        """Verify to_fit_tensors() item ordering so callers can align features."""
        data = _synth_longform(n_subjects=5, n_items=6)
        item_ids = data.to_fit_tensors()["item_ids"]
        # Zero-padded IDs: sorted() == ["i00", "i01", "i02", "i03", "i04", "i05"]
        assert item_ids == sorted(item_ids), (
            "to_fit_tensors() must return item_ids in sorted order; "
            "item_features rows must be aligned accordingly."
        )

    def test_fit_longform_unsorted_item_ids(self):
        """fit() maps item_idx via sorted() even when IDs appear out of order.

        Constructs LongFormData where item IDs are inserted in non-alphabetical
        order ("item_b", "item_a", "item_c") and verifies that to_fit_tensors()
        returns them sorted ("item_a", "item_b", "item_c"). Features must be
        supplied in that sorted order for correct training.
        """
        torch.manual_seed(0)
        n_subjects = 4
        insertion_order = ["item_b", "item_a", "item_c"]
        responses = torch.bernoulli(torch.rand(n_subjects, 3))

        rows = []
        for s in range(n_subjects):
            for j, item_id in enumerate(insertion_order):
                rows.append(
                    {
                        "subject_id": f"s{s:02d}",
                        "item_id": item_id,
                        "benchmark_id": "test",
                        "trial": 0,
                        "test_condition": None,
                        "response": float(responses[s, j].item()),
                        "correct_answer": None,
                        "trace": None,
                    }
                )
        df = pd.DataFrame(rows)
        items_df = pd.DataFrame(
            [{"item_id": iid, "benchmark_id": "test"} for iid in insertion_order]
        )
        subjects_df = pd.DataFrame([{"subject_id": f"s{s:02d}"} for s in range(n_subjects)])
        data = LongFormData(
            name="test",
            responses=df,
            items=items_df,
            subjects=subjects_df,
            traces=None,
            info={},
        )

        # to_fit_tensors() must sort: ["item_a", "item_b", "item_c"]
        fit_inputs = data.to_fit_tensors()
        assert fit_inputs["item_ids"] == ["item_a", "item_b", "item_c"]

        # Features aligned to sorted order — fit must succeed
        F = 4
        features = torch.randn(3, F)
        model = DemandAssessor(n_subjects=n_subjects, item_feature_dim=F)
        history = model.fit(data, features, max_epochs=5, verbose=False)
        assert len(history["losses"]) > 0

    # --- predict() — generalisation -----------------------------------------

    def test_predict_unseen_items(self, small_response_matrix):
        """After fitting, predict on feature vectors not present in training data."""
        n_subjects, n_items = small_response_matrix.shape
        torch.manual_seed(0)
        training_features = torch.randn(n_items, 4)
        model = DemandAssessor(n_subjects=n_subjects, item_feature_dim=4)
        model.fit(
            small_response_matrix, training_features, max_epochs=10, verbose=False
        )

        # New items: feature vectors never seen during training
        new_features = torch.randn(5, 4)
        query = {
            "subject_idx": torch.zeros(5, dtype=torch.long),
            "item_features": new_features,
        }
        probs = model.predict(query)
        assert probs.shape == (5,)
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    # --- serialisation ------------------------------------------------------

    def test_serialization(self, small_response_matrix):
        """load_state_dict() fully restores the model; no re-supply step needed."""
        n_subjects, n_items = small_response_matrix.shape
        torch.manual_seed(0)
        features = torch.randn(n_items, 4)

        model = DemandAssessor(n_subjects=n_subjects, item_feature_dim=4)
        model.fit(small_response_matrix, features, max_epochs=10, verbose=False)

        query = {
            "subject_idx": torch.tensor([0, 1, 2]),
            "item_features": features[:3],
        }
        probs_before = model.predict(query).detach()

        # Save and restore — no set_embeddings() or set_item_features() call.
        state = model.state_dict()
        model2 = DemandAssessor(n_subjects=n_subjects, item_feature_dim=4)
        model2.load_state_dict(state)

        probs_after = model2.predict(query).detach()
        assert torch.allclose(probs_before, probs_after)
        assert model2.n_items == n_items  # persisted via n_items_buf

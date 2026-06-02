# Copyright (c) 2026 AIMS Foundations. MIT License.

import pytest
import torch
from torch import nn

from torch_measure.models import CAIMIRA
from torch_measure.models._predictor import predict_dense


class TestCAIMIRA:
    def test_init(self):
        model = CAIMIRA(n_subjects=10, n_items=20, embedding_dim=64, latent_dim=3)
        assert model.n_subjects == 10
        assert model.n_items == 20
        assert model.embedding_dim == 64
        assert model.latent_dim == 3
        assert model.skill.shape == (10, 3)

    def test_default_latent_dim_matches_paper(self):
        """Default latent_dim mirrors the value used in CAIMIRA paper experiments (m=5)."""
        model = CAIMIRA(n_subjects=4, n_items=6, embedding_dim=8)
        assert model.latent_dim == 5
        assert model.skill.shape == (4, 5)

    def test_init_rejects_nonpositive_dims(self):
        try:
            CAIMIRA(n_subjects=5, n_items=10, embedding_dim=0, latent_dim=3)
            raise AssertionError("Expected ValueError for embedding_dim=0")
        except ValueError:
            pass
        try:
            CAIMIRA(n_subjects=5, n_items=10, embedding_dim=8, latent_dim=0)
            raise AssertionError("Expected ValueError for latent_dim=0")
        except ValueError:
            pass

    def test_difficulty_head_is_linear_and_biasless(self):
        """Architectural invariant: difficulty head is bias-free (centering absorbs the bias)."""
        model = CAIMIRA(n_subjects=5, n_items=10, embedding_dim=8, latent_dim=4)
        assert isinstance(model.difficulty_head, nn.Linear)
        assert model.difficulty_head.bias is None
        assert model.difficulty_head.weight.shape == (4, 8)

    def test_relevance_head_is_linear_with_bias(self):
        """Architectural invariant: relevance head is linear with bias (informative under softmax)."""
        model = CAIMIRA(n_subjects=5, n_items=10, embedding_dim=8, latent_dim=4)
        assert isinstance(model.relevance_head, nn.Linear)
        assert model.relevance_head.bias is not None
        assert model.relevance_head.weight.shape == (4, 8)
        assert model.relevance_head.bias.shape == (4,)

    def test_set_embeddings_wrong_n_items(self):
        model = CAIMIRA(n_subjects=5, n_items=10, embedding_dim=8, latent_dim=3)
        try:
            model.set_embeddings(torch.randn(7, 8))  # wrong n_items
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    def test_set_embeddings_wrong_embedding_dim(self):
        model = CAIMIRA(n_subjects=5, n_items=10, embedding_dim=8, latent_dim=3)
        try:
            model.set_embeddings(torch.randn(10, 16))  # wrong embedding_dim
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    def test_predict_requires_embeddings(self):
        model = CAIMIRA(n_subjects=5, n_items=10, embedding_dim=8, latent_dim=3)
        try:
            predict_dense(model)
            raise AssertionError("Should have raised RuntimeError")
        except RuntimeError:
            pass

    def test_predict_shape(self):
        model = CAIMIRA(n_subjects=5, n_items=10, embedding_dim=8, latent_dim=3)
        model.set_embeddings(torch.randn(10, 8))
        probs = predict_dense(model)
        assert probs.shape == (5, 10)
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_relevance_rows_sum_to_one(self):
        """Softmax invariant: each item's relevance vector lies on the simplex."""
        model = CAIMIRA(n_subjects=5, n_items=10, embedding_dim=8, latent_dim=4)
        model.set_embeddings(torch.randn(10, 8))
        relevance, _ = model.compute_item_params()
        assert relevance.shape == (10, 4)
        assert torch.allclose(relevance.sum(dim=-1), torch.ones(10), atol=1e-6)

    def test_relevance_is_nonnegative(self):
        """Softmax invariant: relevance entries are non-negative."""
        model = CAIMIRA(n_subjects=5, n_items=10, embedding_dim=8, latent_dim=4)
        model.set_embeddings(torch.randn(10, 8))
        relevance, _ = model.compute_item_params()
        assert (relevance >= 0).all()

    def test_difficulty_zero_centered_over_training_bank_dynamic(self):
        """In dynamic centering mode, difficulty across the training bank has zero mean."""
        model = CAIMIRA(n_subjects=5, n_items=10, embedding_dim=8, latent_dim=4)
        model.set_embeddings(torch.randn(10, 8))
        _, difficulty = model.compute_item_params(center="dynamic")
        assert difficulty.shape == (10, 4)
        assert torch.allclose(difficulty.mean(dim=0), torch.zeros(4), atol=1e-6)

    def test_frozen_difficulty_mean_buffer_used_at_inference(self):
        """After fit() the snapshot is taken; frozen-mode centering reads the buffer."""
        torch.manual_seed(0)
        n_subjects, n_items, embed = 6, 12, 8
        model = CAIMIRA(n_subjects=n_subjects, n_items=n_items, embedding_dim=embed, latent_dim=3)
        embeddings = torch.randn(n_items, embed)
        responses = torch.bernoulli(torch.full((n_subjects, n_items), 0.5))
        model.fit(responses, embeddings, max_epochs=5, verbose=False)
        # Buffer should equal the training-bank d_raw mean.
        with torch.no_grad():
            d_raw_train = model.difficulty_head(embeddings)
            assert torch.allclose(model._difficulty_mean, d_raw_train.mean(dim=0), atol=1e-6)
        # Cold-start: new items should be centered by the SAME buffer, not their own mean.
        new_embeddings = torch.randn(5, embed)
        _, d_new = model.compute_item_params(new_embeddings, center="frozen")
        with torch.no_grad():
            d_raw_new = model.difficulty_head(new_embeddings)
            expected = d_raw_new - model._difficulty_mean
        assert torch.allclose(d_new, expected, atol=1e-6)

    def test_predict_matches_manual_caimira_equation(self):
        """End-to-end check: predict() matches sigmoid((s_i - d_j) · r_j) row by row."""
        torch.manual_seed(0)
        n_subjects, n_items, embed, latent = 4, 6, 8, 3
        model = CAIMIRA(n_subjects=n_subjects, n_items=n_items, embedding_dim=embed, latent_dim=latent)
        model.eval()  # use frozen centering (buffer is still zero → no centering effect)
        embeddings = torch.randn(n_items, embed)
        model.set_embeddings(embeddings)

        query = {
            "subject_idx": torch.tensor([0, 1, 2, 3, 0]),
            "item_idx": torch.tensor([1, 2, 3, 4, 5]),
        }
        with torch.no_grad():
            probs = model.predict(query)
            relevance, difficulty = model.compute_item_params(center="frozen")
            manual = torch.sigmoid(
                (
                    (model.skill[query["subject_idx"]] - difficulty[query["item_idx"]]) * relevance[query["item_idx"]]
                ).sum(dim=-1)
            )
        assert probs.shape == (5,)
        assert torch.allclose(probs, manual, atol=1e-6)

    def test_predict_embeddings_scores_cold_start_items_with_frozen_centering(self):
        """Cold-start item embeddings should score without mutating the training bank."""
        torch.manual_seed(0)
        n_subjects, n_items, embed, latent = 4, 6, 8, 3
        model = CAIMIRA(n_subjects=n_subjects, n_items=n_items, embedding_dim=embed, latent_dim=latent)
        train_embeddings = torch.randn(n_items, embed)
        model.set_embeddings(train_embeddings)
        with torch.no_grad():
            model._difficulty_mean.fill_(0.25)
        model.eval()

        subject_idx = torch.tensor([0, 1, 2, 3])
        cold_embeddings = torch.randn(4, embed)
        probs = model.predict_embeddings(subject_idx, cold_embeddings, center="frozen")

        relevance, difficulty = model.compute_item_params(cold_embeddings, center="frozen")
        manual = torch.sigmoid(((model.skill[subject_idx] - difficulty) * relevance).sum(dim=-1))
        assert probs.shape == (4,)
        assert torch.allclose(probs, manual, atol=1e-6)
        assert model._embeddings is not None
        assert torch.allclose(model._embeddings, train_embeddings)

    def test_difficulty_mean_buffer_survives_state_dict_roundtrip(self):
        """The frozen centering buffer is part of the deployable state_dict."""
        model = CAIMIRA(n_subjects=3, n_items=5, embedding_dim=7, latent_dim=2)
        with torch.no_grad():
            model._difficulty_mean.copy_(torch.tensor([0.3, -0.2]))
        clone = CAIMIRA(n_subjects=3, n_items=5, embedding_dim=7, latent_dim=2)
        clone.load_state_dict(model.state_dict())
        assert torch.allclose(clone._difficulty_mean, torch.tensor([0.3, -0.2]))

    def test_embedding_paths_follow_parameter_device_after_to(self):
        """Embedding inputs should follow the module's actual parameter device after .to()."""
        model = CAIMIRA(n_subjects=3, n_items=5, embedding_dim=7, latent_dim=2)
        model.to("meta")
        embeddings = torch.empty(5, 7)
        model.set_embeddings(embeddings)
        assert model._embeddings is not None
        assert model._embeddings.device.type == "meta"

    def test_fit_reduces_loss(self, small_response_matrix):
        torch.manual_seed(0)
        n_subjects, n_items = small_response_matrix.shape
        embeddings = torch.randn(n_items, 16)
        model = CAIMIRA(n_subjects=n_subjects, n_items=n_items, embedding_dim=16, latent_dim=3)
        history = model.fit(small_response_matrix, embeddings, max_epochs=50, verbose=False)
        assert history["losses"][-1] < history["losses"][0]

    def test_eval_mode_uses_frozen_centering(self):
        """In eval mode, compute_item_params uses the buffer (not dynamic recompute)."""
        model = CAIMIRA(n_subjects=5, n_items=10, embedding_dim=8, latent_dim=3)
        model.set_embeddings(torch.randn(10, 8))
        # Manually set the buffer to a known non-zero value
        with torch.no_grad():
            model._difficulty_mean.fill_(0.5)
        model.eval()
        _, d_frozen = model.compute_item_params(center="auto")
        with torch.no_grad():
            d_raw = model.difficulty_head(model._embeddings)
            expected = d_raw - 0.5
        assert torch.allclose(d_frozen, expected, atol=1e-6)

    def test_exported_from_models_init(self):
        """CAIMIRA must be importable from torch_measure.models top-level."""
        from torch_measure.models import CAIMIRA as ExportedCAIMIRA

        assert ExportedCAIMIRA is CAIMIRA


class TestCAIMIRADeviceAndFitMethod:
    """Regression tests for the Lane A patches.

    Pin the contracts that emerged from the PR #2 v2 multi-wave code review:

    * A1 — ``_refresh_difficulty_mean`` must move ``self._embeddings`` to
      the parameter device before calling ``difficulty_head``. The
      attribute is plain Python state, not a registered buffer, so
      ``model.to(device)`` does not carry it along.
    * A2 — ``predict`` must move CPU query indices to the parameter device.
    * A3 — ``fit(method=...)`` must validate the algorithm name with a
      clear ``ValueError`` rather than letting unknown values slip into
      ``**kwargs`` and surface as a confusing ``TypeError`` from
      ``mle_fit``.
    * A4 — ``predict_embeddings`` with the default ``center='frozen'``
      must equal the paper equation ``sigmoid((s - d) . r)`` after a
      ``state_dict()`` round-trip.
    """

    def test_model_to_device_after_set_embeddings_refreshes_difficulty_mean(self):
        """``model.to(device)`` then ``_refresh_difficulty_mean()`` must not
        raise a cross-device ``RuntimeError``.

        Regression test for A1: ``self._embeddings`` is a plain Python
        attribute set by :meth:`CAIMIRA.set_embeddings`, NOT a buffer
        registered via ``register_buffer``. PyTorch's ``model.to(device)``
        only carries parameters and registered buffers, so without the
        patched ``.to(self._parameter_device())`` inside
        ``_refresh_difficulty_mean`` the embeddings would lag behind on
        the original device after a ``model.to(...)`` call.

        The CPU branch always runs and exercises the new ``.to()`` line.
        The CUDA branch is the strongest cross-device assertion: it would
        raise without the A1 fix.
        """
        torch.manual_seed(0)
        model = CAIMIRA(n_subjects=3, n_items=5, embedding_dim=4, latent_dim=2)
        model.set_embeddings(torch.randn(5, 4))
        model.to("cpu")
        model._refresh_difficulty_mean()
        assert model._difficulty_mean.device.type == "cpu"
        assert not torch.allclose(model._difficulty_mean, torch.zeros(2))

        if torch.cuda.is_available():
            torch.manual_seed(0)
            cuda_model = CAIMIRA(n_subjects=3, n_items=5, embedding_dim=4, latent_dim=2)
            cuda_model.set_embeddings(torch.randn(5, 4))
            cuda_model.to("cuda")
            cuda_model._refresh_difficulty_mean()
            assert cuda_model._difficulty_mean.device.type == "cuda"

    def test_predict_accepts_cpu_query_indices_after_model_to_device(self):
        """``predict`` must move CPU-built query indices to the parameter device.

        Regression test for A2: callers building tensors on CPU should
        not need to know whether the model has been moved to CUDA. The
        in-method ``device = self._parameter_device(); .to(device)`` calls
        on ``query["subject_idx"]`` and ``query["item_idx"]`` are what
        make ``predict`` device-agnostic for users.
        """
        torch.manual_seed(0)
        model = CAIMIRA(n_subjects=4, n_items=6, embedding_dim=8, latent_dim=3)
        model.set_embeddings(torch.randn(6, 8))
        model.to("cpu")
        query_cpu = {
            "subject_idx": torch.tensor([0, 1, 2, 3], dtype=torch.long),
            "item_idx": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        }
        probs = model.predict(query_cpu)
        assert probs.shape == (4,)
        assert ((probs >= 0) & (probs <= 1)).all()

        if torch.cuda.is_available():
            torch.manual_seed(0)
            cuda_model = CAIMIRA(n_subjects=4, n_items=6, embedding_dim=8, latent_dim=3)
            cuda_model.set_embeddings(torch.randn(6, 8))
            cuda_model.to("cuda")
            cuda_probs = cuda_model.predict(query_cpu)
            assert cuda_probs.device.type == "cuda"
            assert cuda_probs.shape == (4,)

    def test_fit_method_em_raises_value_error(self):
        """``fit(method='em')`` (or any non-``'mle'`` value) must raise a
        clear ``ValueError``.

        Regression test for A3: before the patch, an unknown ``method``
        slipped into ``**kwargs`` and was forwarded to
        :func:`mle_fit`, surfacing as a cryptic ``TypeError`` about an
        unexpected keyword argument far from the caller's intent.
        """
        torch.manual_seed(0)
        model = CAIMIRA(n_subjects=4, n_items=6, embedding_dim=8, latent_dim=2)
        embeddings = torch.randn(6, 8)
        responses = torch.bernoulli(torch.full((4, 6), 0.5))
        with pytest.raises(ValueError, match="only supports method='mle'"):
            model.fit(responses, embeddings, max_epochs=2, verbose=False, method="em")

    def test_predict_embeddings_default_center_frozen_matches_manual_equation_after_state_dict_load(self):
        """``predict_embeddings`` with default ``center='frozen'`` must
        equal the paper equation after a ``state_dict()`` round-trip.

        Pins the cold-start probability surface across serialize /
        deserialize, which is the realistic deployment path: train on one
        host, ship ``state_dict()`` to a fresh process, and score new
        items with ``predict_embeddings``. The frozen-mean buffer is the
        only piece of training-time centering state that the fresh
        instance carries, so it must reproduce the paper equation
        ``sigmoid((s - d) . r)`` exactly.
        """
        torch.manual_seed(0)
        n_subjects, n_items, embed, latent = 4, 6, 8, 3
        model = CAIMIRA(n_subjects=n_subjects, n_items=n_items, embedding_dim=embed, latent_dim=latent)
        train_embeddings = torch.randn(n_items, embed)
        model.set_embeddings(train_embeddings)
        with torch.no_grad():
            model._difficulty_mean.copy_(model.difficulty_head(train_embeddings).mean(dim=0))

        state = model.state_dict()
        clone = CAIMIRA(n_subjects=n_subjects, n_items=n_items, embedding_dim=embed, latent_dim=latent)
        clone.load_state_dict(state, strict=True)
        clone.eval()

        subject_idx = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        cold_embeddings = torch.randn(4, embed)
        p_from_method = clone.predict_embeddings(subject_idx, cold_embeddings)

        with torch.no_grad():
            relevance = torch.softmax(clone.relevance_head(cold_embeddings), dim=-1)
            d_raw = clone.difficulty_head(cold_embeddings)
            difficulty = d_raw - clone._difficulty_mean
            p_manual = torch.sigmoid(((clone.skill[subject_idx] - difficulty) * relevance).sum(dim=-1))
        assert p_from_method.shape == (4,)
        assert torch.allclose(p_from_method, p_manual, atol=1e-6)

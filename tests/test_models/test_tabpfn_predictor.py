# Copyright (c) 2026 AIMS Foundations. MIT License.

import pytest
import torch

from torch_measure.models import TabPFNPredictor
from torch_measure.models._predictor import cartesian_query, predict_dense


@pytest.mark.slow
class TestTabPFNPredictor:
    def test_init(self):
        model = TabPFNPredictor(n_subjects=10, n_items=20, n_features=8)
        assert model.n_subjects == 10
        assert model.n_items == 20
        assert model.n_features == 8

    def test_fit_and_predict_shape(self, small_response_matrix):
        n_subjects, n_items = small_response_matrix.shape
        n_features = 8
        torch.manual_seed(0)
        features = torch.randn(n_items, n_features)
        model = TabPFNPredictor(
            n_subjects=n_subjects,
            n_items=n_items,
            n_features=n_features,
        )
        history = model.fit(small_response_matrix, features)
        assert history["n_train"] == n_subjects * n_items
        assert history["n_observed"] == n_subjects * n_items

        probs = predict_dense(model)
        assert probs.shape == (n_subjects, n_items)
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_predict_requires_fit(self):
        model = TabPFNPredictor(n_subjects=10, n_items=20, n_features=8)
        with pytest.raises(RuntimeError):
            model.predict(cartesian_query(10, 20))

    def test_fit_validates_feature_shape(self, small_response_matrix):
        n_subjects, n_items = small_response_matrix.shape
        torch.manual_seed(0)
        bad_features = torch.randn(n_items, 99)  # wrong n_features
        model = TabPFNPredictor(
            n_subjects=n_subjects,
            n_items=n_items,
            n_features=8,
        )
        with pytest.raises(ValueError):
            model.fit(small_response_matrix, bad_features)

    def test_max_train_cap_warns(self, small_response_matrix):
        n_subjects, n_items = small_response_matrix.shape
        n_features = 4
        torch.manual_seed(0)
        features = torch.randn(n_items, n_features)
        model = TabPFNPredictor(
            n_subjects=n_subjects,
            n_items=n_items,
            n_features=n_features,
            max_train=100,
        )
        with pytest.warns(UserWarning, match="max_train"):
            history = model.fit(small_response_matrix, features)
        assert history["n_train"] == 100
        assert history["n_observed"] == n_subjects * n_items

    def test_single_class_raises(self):
        n_subjects, n_items, n_features = 5, 10, 4
        response = torch.zeros(n_subjects, n_items)
        torch.manual_seed(0)
        features = torch.randn(n_items, n_features)
        model = TabPFNPredictor(
            n_subjects=n_subjects,
            n_items=n_items,
            n_features=n_features,
        )
        with pytest.raises(RuntimeError, match="one class"):
            model.fit(response, features)

    def test_mask_excludes_cells(self, small_response_matrix):
        n_subjects, n_items = small_response_matrix.shape
        n_features = 4
        torch.manual_seed(0)
        features = torch.randn(n_items, n_features)
        mask = torch.ones(n_subjects, n_items, dtype=torch.bool)
        mask[:, n_items // 2 :] = False
        model = TabPFNPredictor(
            n_subjects=n_subjects,
            n_items=n_items,
            n_features=n_features,
        )
        history = model.fit(small_response_matrix, features, mask=mask)
        assert history["n_train"] == int(mask.sum())

    def test_forward_returns_predict(self, small_response_matrix):
        n_subjects, n_items = small_response_matrix.shape
        n_features = 4
        torch.manual_seed(0)
        features = torch.randn(n_items, n_features)
        model = TabPFNPredictor(
            n_subjects=n_subjects,
            n_items=n_items,
            n_features=n_features,
        )
        model.fit(small_response_matrix, features)
        query = cartesian_query(n_subjects, n_items)
        torch.testing.assert_close(model(query), model.predict(query))

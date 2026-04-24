# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Public-API tests for long-form input to ``IRTModel.fit()``.

Exercises the canonical end-to-end path: construct a
:class:`~torch_measure.datasets.LongFormData`, pass it to
``model.fit(data)``, verify parameters are estimated sensibly, and
verify numerical equivalence against the wide-form path.
"""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from torch_measure.datasets._long_form import LongFormData
from torch_measure.models import Rasch, ThreePL, TwoPL


def _synth_longform(n_subjects: int = 20, n_items: int = 30, seed: int = 42) -> LongFormData:
    """Dense synthetic LongFormData: one observation per (subject, item)."""
    torch.manual_seed(seed)
    ability = torch.randn(n_subjects)
    difficulty = torch.randn(n_items)
    probs = torch.sigmoid(ability.unsqueeze(1) - difficulty.unsqueeze(0))
    responses = torch.bernoulli(probs)

    rows = []
    for s in range(n_subjects):
        for i in range(n_items):
            rows.append({
                "subject_id": f"s{s:02d}",
                "item_id": f"i{i:02d}",
                "benchmark_id": "synthetic",
                "trial": 0,
                "test_condition": None,
                "response": float(responses[s, i].item()),
                "correct_answer": None,
                "trace": None,
            })
    df = pd.DataFrame(rows)
    items = pd.DataFrame([
        {"item_id": f"i{i:02d}", "benchmark_id": "synthetic"} for i in range(n_items)
    ])
    subjects = pd.DataFrame([
        {"subject_id": f"s{s:02d}"} for s in range(n_subjects)
    ])
    return LongFormData(
        name="synthetic",
        responses=df,
        items=items,
        subjects=subjects,
        traces=None,
        info={},
    )


class TestLongFormFit:
    def test_rasch_long_form(self):
        data = _synth_longform()
        model = Rasch(n_subjects=20, n_items=30)
        history = model.fit(data, method="mle", max_epochs=100, verbose=False)
        assert len(history["losses"]) > 0
        assert history["losses"][-1] < history["losses"][0]
        assert model.ability.shape == (20,)
        assert model.difficulty.shape == (30,)

    def test_twopl_long_form(self):
        data = _synth_longform()
        model = TwoPL(n_subjects=20, n_items=30)
        history = model.fit(data, method="jml", max_epochs=50, verbose=False)
        assert history["losses"][-1] < history["losses"][0]

    def test_long_and_wide_agree(self):
        """Same seed + data → MLE loss should match between long and wide paths."""
        data = _synth_longform()

        # Wide-form path
        matrix = data.to_response_matrix().data
        torch.manual_seed(0)
        model_w = Rasch(n_subjects=20, n_items=30)
        hist_w = model_w.fit(matrix, method="mle", max_epochs=200, verbose=False)

        # Long-form path (same seed so Rasch's Parameter init matches)
        torch.manual_seed(0)
        model_l = Rasch(n_subjects=20, n_items=30)
        hist_l = model_l.fit(data, method="mle", max_epochs=200, verbose=False)

        # Both should converge; final losses should be very close
        assert abs(hist_w["losses"][-1] - hist_l["losses"][-1]) < 1e-4

    def test_long_form_sparse_observations(self):
        """LongFormData with missing observations — no rows emitted for the missing cells."""
        data = _synth_longform(n_subjects=15, n_items=20)
        # Drop 20% of observations at random
        torch.manual_seed(123)
        keep = torch.rand(len(data.responses)) > 0.2
        data = LongFormData(
            name=data.name,
            responses=data.responses[keep.numpy()].reset_index(drop=True),
            items=data.items,
            subjects=data.subjects,
            traces=None,
            info={},
        )
        model = Rasch(n_subjects=15, n_items=20)
        history = model.fit(data, method="mle", max_epochs=100, verbose=False)
        assert history["losses"][-1] < history["losses"][0]


class TestLongFormFitPyro:
    def test_svi_long_form(self):
        pytest.importorskip("pyro", reason="pyro-ppl required for SVI tests")
        data = _synth_longform()
        model = Rasch(n_subjects=20, n_items=30)
        history = model.fit(data, method="svi", max_epochs=50, verbose=False)
        assert len(history["losses"]) == 50
        assert history["losses"][-1] < history["losses"][0]


class TestLongFormFitNetwork:
    def test_ising_long_form(self):
        from torch_measure.models import IsingModel

        data = _synth_longform(n_subjects=25, n_items=10)
        model = IsingModel(n_items=10)
        history = model.fit(data, max_epochs=50, verbose=False)
        assert len(history["losses"]) > 0


class TestPredictAt:
    def test_predict_at_matches_predict(self):
        model = TwoPL(n_subjects=20, n_items=30)
        # Random ability/diff/disc
        with torch.no_grad():
            model.ability.normal_()
            model.difficulty.normal_()
        probs_full = model.predict()

        s_idx = torch.tensor([0, 5, 19, 10], dtype=torch.long)
        i_idx = torch.tensor([0, 12, 29, 7], dtype=torch.long)
        probs_at = model.predict_at(s_idx, i_idx)

        assert torch.allclose(probs_at, probs_full[s_idx, i_idx])

    def test_predict_at_threepl(self):
        model = ThreePL(n_subjects=10, n_items=15)
        probs_full = model.predict()
        s_idx = torch.tensor([0, 3, 9], dtype=torch.long)
        i_idx = torch.tensor([0, 7, 14], dtype=torch.long)
        assert torch.allclose(model.predict_at(s_idx, i_idx), probs_full[s_idx, i_idx])

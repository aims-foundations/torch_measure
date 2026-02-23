# Copyright (c) 2026 AIMS Foundation. MIT License.

import torch

from torch_measure.cat.runner import AdaptiveTester
from torch_measure.models import Rasch


class TestAdaptiveTester:
    def test_run_basic(self):
        model = Rasch(n_subjects=1, n_items=20)
        with torch.no_grad():
            model.difficulty.copy_(torch.linspace(-2, 2, 20))
        tester = AdaptiveTester(model, strategy="fisher")
        responses = torch.bernoulli(torch.full((20,), 0.5))
        result = tester.run(responses, budget=5)
        assert "ability" in result
        assert len(result["administered"]) == 5
        assert len(result["responses"]) == 5
        assert len(result["ability_trajectory"]) == 5

    def test_budget_limits_items(self):
        model = Rasch(n_subjects=1, n_items=20)
        tester = AdaptiveTester(model, strategy="fisher")
        responses = torch.bernoulli(torch.full((20,), 0.5))
        result = tester.run(responses, budget=3)
        assert len(result["administered"]) == 3

    def test_spanning_strategy(self):
        model = Rasch(n_subjects=1, n_items=20)
        with torch.no_grad():
            model.difficulty.copy_(torch.linspace(-3, 3, 20))
        tester = AdaptiveTester(model, strategy="spanning", n_spanning=5)
        responses = torch.bernoulli(torch.full((20,), 0.5))
        result = tester.run(responses, budget=10)
        assert len(result["administered"]) == 10

    def test_no_duplicate_items(self):
        model = Rasch(n_subjects=1, n_items=20)
        tester = AdaptiveTester(model, strategy="fisher")
        responses = torch.bernoulli(torch.full((20,), 0.5))
        result = tester.run(responses, budget=10)
        assert len(set(result["administered"])) == 10

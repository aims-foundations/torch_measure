# Copyright (c) 2026 AIMS Foundation. MIT License.

import torch

from torch_measure.cat.strategies import MaxInfoStrategy, RandomStrategy, SpanningStrategy


class TestMaxInfoStrategy:
    def test_selects_best_item(self):
        strategy = MaxInfoStrategy()
        ability = torch.tensor(0.0)
        difficulty = torch.tensor([-2.0, 0.0, 2.0])
        administered = torch.tensor([False, False, False])
        # Should select item 1 (difficulty closest to ability)
        item = strategy.select(ability, difficulty, None, administered)
        assert item == 1

    def test_skips_administered(self):
        strategy = MaxInfoStrategy()
        ability = torch.tensor(0.0)
        difficulty = torch.tensor([-2.0, 0.0, 2.0])
        administered = torch.tensor([False, True, False])
        item = strategy.select(ability, difficulty, None, administered)
        assert item != 1


class TestSpanningStrategy:
    def test_spanning_phase(self):
        strategy = SpanningStrategy(n_spanning=3)
        ability = torch.tensor(0.0)
        difficulty = torch.linspace(-3, 3, 10)
        administered = torch.zeros(10, dtype=torch.bool)
        selected = []
        for _ in range(3):
            item = strategy.select(ability, difficulty, None, administered)
            administered[item] = True
            selected.append(item)
        # Should select diverse items across difficulty range
        assert len(set(selected)) == 3

    def test_reset(self):
        strategy = SpanningStrategy(n_spanning=2)
        strategy._spanning_count = 5
        strategy.reset()
        assert strategy._spanning_count == 0


class TestRandomStrategy:
    def test_selects_available(self):
        strategy = RandomStrategy()
        ability = torch.tensor(0.0)
        difficulty = torch.randn(10)
        administered = torch.zeros(10, dtype=torch.bool)
        administered[:8] = True  # only 2 items available
        item = strategy.select(ability, difficulty, None, administered)
        assert not administered[item]

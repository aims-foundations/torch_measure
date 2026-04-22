# Copyright (c) 2026 AIMS Foundations. MIT License.

"""PairwiseComparisons data structure for pairwise preference data."""

from __future__ import annotations

import torch


class PairwiseComparisons:
    """Pairwise comparison data (e.g., Chatbot Arena).

    Each observation records subject_a vs subject_b with an outcome.

    Parameters
    ----------
    subject_a : torch.LongTensor
        Indices into ``subject_ids`` for the first subject in each comparison.
        Shape: ``(n_comparisons,)``.
    subject_b : torch.LongTensor
        Indices into ``subject_ids`` for the second subject in each comparison.
        Shape: ``(n_comparisons,)``.
    outcome : torch.Tensor
        Comparison outcome. ``1.0`` = subject_a wins, ``0.0`` = subject_b wins,
        ``0.5`` = tie. Shape: ``(n_comparisons,)``.
    subject_ids : list[str]
        Unique subject identifiers (e.g., model names).
    item_ids : list[str] | None
        Unique item/prompt identifiers (e.g., question IDs).
    item_contents : list[str] | None
        Text content for each item (one per entry in ``item_ids``).
    item_idx : torch.LongTensor | None
        Per-comparison index into ``item_ids``, shape ``(n_comparisons,)``.
        Maps each comparison to the item/prompt it was evaluated on.
    subject_metadata : list[dict] | None
        Structured metadata per subject (one dict per entry in ``subject_ids``).
    comparison_metadata : list[dict] | None
        Structured metadata per comparison (one dict per row).
    """

    def __init__(
        self,
        subject_a: torch.Tensor,
        subject_b: torch.Tensor,
        outcome: torch.Tensor,
        subject_ids: list[str],
        item_ids: list[str] | None = None,
        item_contents: list[str] | None = None,
        item_idx: torch.Tensor | None = None,
        subject_metadata: list[dict] | None = None,
        comparison_metadata: list[dict] | None = None,
    ) -> None:
        if subject_a.ndim != 1 or subject_b.ndim != 1 or outcome.ndim != 1:
            raise ValueError("subject_a, subject_b, and outcome must be 1-D tensors")
        n = subject_a.shape[0]
        if subject_b.shape[0] != n or outcome.shape[0] != n:
            raise ValueError(
                f"Length mismatch: subject_a={subject_a.shape[0]}, "
                f"subject_b={subject_b.shape[0]}, outcome={outcome.shape[0]}"
            )
        self.subject_a = subject_a.long()
        self.subject_b = subject_b.long()
        self.outcome = outcome.float()
        self.subject_ids = subject_ids
        self.item_ids = item_ids
        self.item_contents = item_contents
        self.item_idx = item_idx.long() if item_idx is not None else None
        self.subject_metadata = subject_metadata
        self.comparison_metadata = comparison_metadata

    @property
    def n_comparisons(self) -> int:
        """Number of pairwise comparisons."""
        return self.outcome.shape[0]

    @property
    def n_subjects(self) -> int:
        """Number of unique subjects."""
        return len(self.subject_ids)

    @property
    def n_items(self) -> int:
        """Number of unique items/prompts."""
        if self.item_ids is None:
            return 0
        return len(self.item_ids)

    @property
    def shape(self) -> tuple[int, int]:
        """(n_comparisons, n_subjects)."""
        return (self.n_comparisons, self.n_subjects)

    @property
    def density(self) -> float:
        """Fraction of all possible ordered pairs that are observed.

        Computed as ``n_comparisons / (n_subjects * (n_subjects - 1) / 2)``.
        """
        n = self.n_subjects
        total_pairs = n * (n - 1) / 2
        if total_pairs == 0:
            return 0.0
        return self.n_comparisons / total_pairs

    def win_rates(self) -> torch.Tensor:
        """Per-subject overall win rate.

        Returns
        -------
        torch.Tensor
            Win rate for each subject, shape ``(n_subjects,)``.
            Ties count as 0.5 wins and 0.5 losses.
        """
        wins = torch.zeros(self.n_subjects)
        counts = torch.zeros(self.n_subjects)

        wins.scatter_add_(0, self.subject_a, self.outcome)
        wins.scatter_add_(0, self.subject_b, 1.0 - self.outcome)
        counts.scatter_add_(0, self.subject_a, torch.ones(self.n_comparisons))
        counts.scatter_add_(0, self.subject_b, torch.ones(self.n_comparisons))

        return wins / counts.clamp(min=1)

    def to_win_matrix(self) -> torch.Tensor:
        """Aggregate into a pairwise win-rate matrix.

        Returns
        -------
        torch.Tensor
            Square matrix of shape ``(n_subjects, n_subjects)`` where entry
            ``(i, j)`` is the win rate of subject *i* against subject *j*.
            Diagonal is NaN. Unobserved pairs are NaN.
        """
        n = self.n_subjects
        wins = torch.zeros(n, n)
        counts = torch.zeros(n, n)

        a = self.subject_a
        b = self.subject_b
        wins[a, b] += self.outcome
        wins[b, a] += 1.0 - self.outcome
        counts[a, b] += 1
        counts[b, a] += 1

        mat = wins / counts.clamp(min=1)
        mat[counts == 0] = float("nan")
        mat.fill_diagonal_(float("nan"))
        return mat

    def to(self, device: torch.device | str) -> PairwiseComparisons:
        """Move tensors to a device."""
        return PairwiseComparisons(
            subject_a=self.subject_a.to(device),
            subject_b=self.subject_b.to(device),
            outcome=self.outcome.to(device),
            subject_ids=self.subject_ids,
            item_ids=self.item_ids,
            item_contents=self.item_contents,
            item_idx=self.item_idx.to(device) if self.item_idx is not None else None,
            subject_metadata=self.subject_metadata,
            comparison_metadata=self.comparison_metadata,
        )

    @classmethod
    def from_dataframe(
        cls,
        df,
        subject_a_col: str = "model_a",
        subject_b_col: str = "model_b",
        outcome_col: str = "outcome",
    ) -> PairwiseComparisons:
        """Create from a pandas DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with at least subject_a, subject_b, and outcome columns.
        subject_a_col : str
            Column name for the first subject.
        subject_b_col : str
            Column name for the second subject.
        outcome_col : str
            Column name for the outcome (1.0 = a wins, 0.0 = b wins, 0.5 = tie).
        """
        all_subjects = sorted(set(df[subject_a_col]) | set(df[subject_b_col]))
        sid_to_idx = {s: i for i, s in enumerate(all_subjects)}

        subject_a = torch.tensor([sid_to_idx[s] for s in df[subject_a_col]], dtype=torch.long)
        subject_b = torch.tensor([sid_to_idx[s] for s in df[subject_b_col]], dtype=torch.long)
        outcome = torch.tensor(df[outcome_col].values, dtype=torch.float32)

        return cls(
            subject_a=subject_a,
            subject_b=subject_b,
            outcome=outcome,
            subject_ids=all_subjects,
        )

    def __repr__(self) -> str:
        return (
            f"PairwiseComparisons(n_comparisons={self.n_comparisons}, "
            f"n_subjects={self.n_subjects}, density={self.density:.2%})"
        )

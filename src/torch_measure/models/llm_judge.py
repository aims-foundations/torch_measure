# Copyright (c) 2026 AIMS Foundations. MIT License.

"""LLM-as-judge predictive model using next-token yes/no probabilities."""

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_JUDGE_TEMPLATE = (
    "You will see a description of an AI subject and an"
    " evaluation item. Decide whether the subject would answer the item"
    " correctly. Reply with a single token: yes or no."
    "\n    Benchmark: {benchmark}"
    "\n    Condition: {condition}"
    "\n    Subject: {subject_content}"
    "\n    Item: {item_content}"
    "\n    Answer:"
)


class LLMJudge:
    """LLM-as-judge predictive model.

    Uses the next-token yes/no log-probability ratio from a causal language
    model to predict whether a subject would answer an item correctly.
    Optionally prepends same-subject in-context examples from ``labeled``.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier.
    max_icl : int
        Maximum number of same-subject labeled examples to prepend as
        in-context demonstrations.
    batch_size : int
        Batch size for LLM inference.
    device : str
        Device passed to ``device_map``. Use ``"auto"`` for multi-GPU.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-7B-Instruct",
        max_icl: int = 5,
        batch_size: int = 32,
        device: str = "auto",
    ) -> None:
        self.max_icl = max_icl
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="sdpa",
        )
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self._yes_id = self.tokenizer.encode(" yes", add_special_tokens=False)[-1]
        self._no_id = self.tokenizer.encode(" no", add_special_tokens=False)[-1]

    def _build_prompt(self, data: dict, labeled: list[dict] | None = None) -> str:
        """Build the judge prompt, optionally with same-subject ICL examples."""
        if labeled:
            same_subj = [
                ex for ex in labeled
                if ex["subject_content"] == data["subject_content"]
            ][-self.max_icl:]
            if same_subj:
                icl = "\n\n".join(
                    _JUDGE_TEMPLATE.format(
                        benchmark=ex.get("benchmark", ""),
                        condition=ex.get("condition", ""),
                        subject_content=ex["subject_content"],
                        item_content=ex["item_content"],
                    ) + (" yes" if ex["label"] >= 0.5 else " no")
                    for ex in same_subj
                )
                return icl + "\n\n" + _JUDGE_TEMPLATE.format(**data)
        return _JUDGE_TEMPLATE.format(**data)

    def _batch_probs(self, prompts: list[str]) -> list[float]:
        """Run a batch of prompts; return p_yes / (p_yes + p_no) for each."""
        ids = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)
        with torch.no_grad():
            logits = self.model(**ids).logits[:, -1, :]
        lp = torch.log_softmax(logits, dim=-1)
        p_yes = lp[:, self._yes_id].exp()
        p_no = lp[:, self._no_id].exp()
        return (p_yes / (p_yes + p_no)).tolist()

    def predict(self, data: dict, labeled: list[dict] | None = None) -> float:
        """Compute response probability P(subject passes item).

        Parameters
        ----------
        data : dict
            Dictionary with keys ``"subject_content"``, ``"item_content"``,
            ``"benchmark"``, and ``"condition"``.
        labeled : list[dict] or None
            Previously observed subject-item-response records with keys
            ``"subject_content"``, ``"item_content"``, ``"benchmark"``,
            ``"condition"``, and ``"label"`` (float in [0, 1]).  Same-subject
            records are prepended as in-context examples.

        Returns
        -------
        float
            Predicted probability that the subject passes the item, clipped to
            ``[1e-7, 1 - 1e-7]``.
        """
        prompt = self._build_prompt(data, labeled)
        prob = self._batch_probs([prompt])[0]
        return float(np.clip(prob, 1e-7, 1 - 1e-7))

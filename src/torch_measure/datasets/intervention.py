# Copyright (c) 2026 AIMS Foundation. MIT License.

"""Intervention / treatment-response dataset definitions.

These datasets differ from standard benchmarks: the subjects are *humans*
(not LLMs), and each response matrix is paired with a treatment condition.
The core structure is:

    (human participants) x (items) x {treatment, control} -> outcome

This enables measurement of AI's causal effect on human performance.
"""

from __future__ import annotations

from torch_measure.datasets._info import DatasetInfo

_REPO = "aims-foundation/torch-measure-data"

# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------

_COLLAB_CXR_CITATION = (
    "@article{yu2025collabcxr,\n"
    "  title={Heterogeneity and predictors of the effects of AI assistance on radiologists},\n"
    "  author={Yu, Feiyang and Moehring, Alex and Banerjee, Oishi and others},\n"
    "  journal={Nature Medicine},\n"
    "  year={2024},\n"
    "  url={https://www.nature.com/articles/s41591-024-02850-w}\n"
    "}"
)

_METR_PRODUCTIVITY_CITATION = (
    "@article{metr2025productivity,\n"
    "  title={Measuring the Impact of Early-2025 AI on Experienced Open-Source Developer Productivity},\n"
    "  author={{METR}},\n"
    "  year={2025},\n"
    "  url={https://arxiv.org/abs/2507.09089}\n"
    "}"
)

_HAIID_CITATION = (
    "@article{vodrahalli2022haiid,\n"
    "  title={Do Humans Trust Advice More if it Comes from AI? An Analysis of Human-AI Interactions},\n"
    "  author={Vodrahalli, Kailas and Daneshjou, Roxana and Gerstenberg, Tobias and Zou, James},\n"
    "  year={2022},\n"
    "  url={https://github.com/kailas-v/human-ai-interactions}\n"
    "}"
)

_GENAI_LEARNING_CITATION = (
    "@article{bastani2025genai,\n"
    "  title={Generative AI without guardrails can harm learning},\n"
    "  author={Bastani, Hamsa and Bastani, Osbert and Sungu, Alp and others},\n"
    "  journal={Proceedings of the National Academy of Sciences},\n"
    "  year={2025},\n"
    "  url={https://www.pnas.org/doi/10.1073/pnas.2422633122}\n"
    "}"
)


def _register_intervention_datasets() -> dict[str, DatasetInfo]:
    """Return registry entries for intervention / treatment-response datasets.

    These datasets measure human performance with and without AI assistance.
    Each dataset provides paired response matrices (one per condition), where:

    - **Rows**: Human participants (radiologists, developers, students, etc.)
    - **Columns**: Items (cases, tasks, problems)
    - **Values**: Outcomes (probability, time, binary correct)
    """
    d: dict[str, DatasetInfo] = {}

    # ── Collab-CXR (Radiology) ────────────────────────────────────────────

    d["intervention/collab_cxr_image_only"] = DatasetInfo(
        name="intervention/collab_cxr_image_only",
        family="intervention",
        description="Collab-CXR: 326 radiologists x 324 CXR cases, image only (no AI)",
        response_type="continuous",
        n_subjects=326,
        n_items=324,
        subject_entity="radiologist",
        item_entity="case",
        filename="intervention/collab_cxr_image_only.pt",
        citation=_COLLAB_CXR_CITATION,
        url="https://osf.io/z7apq/",
        license="CC-BY-4.0",
        tags=["intervention", "radiology", "no-ai", "continuous"],
    )

    d["intervention/collab_cxr_image_ai"] = DatasetInfo(
        name="intervention/collab_cxr_image_ai",
        family="intervention",
        description="Collab-CXR: 326 radiologists x 324 CXR cases, with AI predictions",
        response_type="continuous",
        n_subjects=326,
        n_items=324,
        subject_entity="radiologist",
        item_entity="case",
        filename="intervention/collab_cxr_image_ai.pt",
        citation=_COLLAB_CXR_CITATION,
        url="https://osf.io/z7apq/",
        license="CC-BY-4.0",
        tags=["intervention", "radiology", "with-ai", "continuous"],
    )

    d["intervention/collab_cxr_accuracy_no_ai"] = DatasetInfo(
        name="intervention/collab_cxr_accuracy_no_ai",
        family="intervention",
        description="Collab-CXR: binary accuracy, image only (no AI)",
        response_type="continuous",
        n_subjects=326,
        n_items=324,
        subject_entity="radiologist",
        item_entity="case",
        filename="intervention/collab_cxr_accuracy_no_ai.pt",
        citation=_COLLAB_CXR_CITATION,
        url="https://osf.io/z7apq/",
        license="CC-BY-4.0",
        tags=["intervention", "radiology", "no-ai", "accuracy"],
    )

    d["intervention/collab_cxr_accuracy_with_ai"] = DatasetInfo(
        name="intervention/collab_cxr_accuracy_with_ai",
        family="intervention",
        description="Collab-CXR: binary accuracy, with AI predictions",
        response_type="continuous",
        n_subjects=326,
        n_items=324,
        subject_entity="radiologist",
        item_entity="case",
        filename="intervention/collab_cxr_accuracy_with_ai.pt",
        citation=_COLLAB_CXR_CITATION,
        url="https://osf.io/z7apq/",
        license="CC-BY-4.0",
        tags=["intervention", "radiology", "with-ai", "accuracy"],
    )

    # ── METR Developer Productivity RCT ────────────────────────────────────

    d["intervention/metr_dev_early_ai"] = DatasetInfo(
        name="intervention/metr_dev_early_ai",
        family="intervention",
        description="METR Early-2025: 16 devs x 136 issues, AI-allowed (completion time in min)",
        response_type="continuous",
        n_subjects=16,
        n_items=136,
        subject_entity="developer",
        item_entity="issue",
        filename="intervention/metr_dev_early_ai.pt",
        citation=_METR_PRODUCTIVITY_CITATION,
        url="https://github.com/METR/Measuring-Early-2025-AI-on-Exp-OSS-Devs",
        license="MIT",
        tags=["intervention", "coding", "ai-allowed", "time"],
    )

    d["intervention/metr_dev_early_no_ai"] = DatasetInfo(
        name="intervention/metr_dev_early_no_ai",
        family="intervention",
        description="METR Early-2025: 16 devs x 110 issues, AI-disallowed (completion time in min)",
        response_type="continuous",
        n_subjects=16,
        n_items=110,
        subject_entity="developer",
        item_entity="issue",
        filename="intervention/metr_dev_early_no_ai.pt",
        citation=_METR_PRODUCTIVITY_CITATION,
        url="https://github.com/METR/Measuring-Early-2025-AI-on-Exp-OSS-Devs",
        license="MIT",
        tags=["intervention", "coding", "ai-disallowed", "time"],
    )

    d["intervention/metr_dev_late_ai"] = DatasetInfo(
        name="intervention/metr_dev_late_ai",
        family="intervention",
        description="METR Late-2025: 52 devs x 500 issues, AI-allowed (completion time in min)",
        response_type="continuous",
        n_subjects=52,
        n_items=500,
        subject_entity="developer",
        item_entity="issue",
        filename="intervention/metr_dev_late_ai.pt",
        citation=_METR_PRODUCTIVITY_CITATION,
        url="https://github.com/METR/Measuring-Late-2025-AI-on-OSS-Devs",
        license="MIT",
        tags=["intervention", "coding", "ai-allowed", "time"],
    )

    d["intervention/metr_dev_late_no_ai"] = DatasetInfo(
        name="intervention/metr_dev_late_no_ai",
        family="intervention",
        description="METR Late-2025: 53 devs x 415 issues, AI-disallowed (completion time in min)",
        response_type="continuous",
        n_subjects=53,
        n_items=415,
        subject_entity="developer",
        item_entity="issue",
        filename="intervention/metr_dev_late_no_ai.pt",
        citation=_METR_PRODUCTIVITY_CITATION,
        url="https://github.com/METR/Measuring-Late-2025-AI-on-OSS-Devs",
        license="MIT",
        tags=["intervention", "coding", "ai-disallowed", "time"],
    )

    # ── HAIID (Human-AI Interactions Dataset) ──────────────────────────────

    for domain in ["art", "census", "cities", "dermatology", "sarcasm"]:
        d[f"intervention/haiid_{domain}_pre"] = DatasetInfo(
            name=f"intervention/haiid_{domain}_pre",
            family="intervention",
            description=f"HAIID {domain}: participant x item, pre-advice binary accuracy",
            response_type="binary",
            n_subjects={"art": 558, "census": 92, "cities": 142, "dermatology": 37, "sarcasm": 296}[domain],
            n_items={"art": 32, "census": 32, "cities": 32, "dermatology": 24, "sarcasm": 32}[domain],
            subject_entity="participant",
            item_entity="classification_item",
            filename=f"intervention/haiid_{domain}_pre.pt",
            citation=_HAIID_CITATION,
            url="https://github.com/kailas-v/human-ai-interactions",
            license="MIT",
            tags=["intervention", "classification", domain, "pre-advice"],
        )

        for source in ["ai", "human"]:
            n_subj = {
                ("art", "ai"): 278, ("art", "human"): 280,
                ("census", "ai"): 43, ("census", "human"): 49,
                ("cities", "ai"): 79, ("cities", "human"): 63,
                ("dermatology", "ai"): 20, ("dermatology", "human"): 17,
                ("sarcasm", "ai"): 147, ("sarcasm", "human"): 149,
            }[(domain, source)]

            d[f"intervention/haiid_{domain}_post_{source}"] = DatasetInfo(
                name=f"intervention/haiid_{domain}_post_{source}",
                family="intervention",
                description=f"HAIID {domain}: post-{source}-advice binary accuracy",
                response_type="binary",
                n_subjects=n_subj,
                n_items={"art": 32, "census": 32, "cities": 32, "dermatology": 24, "sarcasm": 32}[domain],
                subject_entity="participant",
                item_entity="classification_item",
                filename=f"intervention/haiid_{domain}_post_{source}.pt",
                citation=_HAIID_CITATION,
                url="https://github.com/kailas-v/human-ai-interactions",
                license="MIT",
                tags=["intervention", "classification", domain, f"post-{source}-advice"],
            )

    # ── GenAI Learning (Education RCT) ─────────────────────────────────────

    for phase, n_items in [("practice", 57), ("exam", 48)]:
        for arm, n_students in [("control", 349), ("augmented", 312), ("vanilla", 282)]:
            d[f"intervention/genai_learning_{phase}_{arm}"] = DatasetInfo(
                name=f"intervention/genai_learning_{phase}_{arm}",
                family="intervention",
                description=f"GenAI Learning: {phase} phase, {arm} arm ({n_students} students x {n_items} problems)",
                response_type="binary",
                n_subjects=n_students,
                n_items=n_items,
                subject_entity="student",
                item_entity="math_problem",
                filename=f"intervention/genai_learning_{phase}_{arm}.pt",
                citation=_GENAI_LEARNING_CITATION,
                url="https://github.com/obastani/GenAICanHarmLearning",
                license="MIT",
                tags=["intervention", "education", "math", phase, arm],
            )

    return d

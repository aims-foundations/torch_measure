# Copyright (c) 2026 AIMS Foundation. MIT License.

"""HELM benchmark dataset definitions."""

from __future__ import annotations

from torch_measure.datasets._info import DatasetInfo

_HELM_URL = "https://crfm.stanford.edu/helm/"
_N_SUBJECTS = 183  # Number of LLMs evaluated across HELM benchmarks

# HELM citation used for benchmarks without a specific paper.
_HELM_CITATION = (
    "@article{liang2023holistic,\n"
    "  title={Holistic Evaluation of Language Models},\n"
    "  author={Liang, Percy and Bommasani, Rishi and Lee, Tony and Tsipras, Dimitris and "
    "Soylu, Dilara and Yasunaga, Michihiro and Zhang, Yian and Narang, Deepak and others},\n"
    "  journal={Transactions on Machine Learning Research},\n"
    "  year={2023}\n"
    "}"
)


def _register_helm_datasets() -> dict[str, DatasetInfo]:
    """Return registry entries for HELM benchmark response matrices.

    HELM (Holistic Evaluation of Language Models) by Stanford CRFM evaluates
    LLMs on a broad set of benchmarks.  Each response matrix has:

    - **Rows**: 183 LLMs evaluated in HELM (subjects).
    - **Columns**: Individual questions / tasks from each benchmark (items).
    - **Values**: Binary {0, 1} for correct / incorrect (NaN for missing).

    Source data: ``stair-lab/reeval`` on HuggingFace Hub.
    """
    datasets: dict[str, DatasetInfo] = {}

    # --- Knowledge & Reasoning ---

    datasets["helm/mmlu"] = DatasetInfo(
        name="helm/mmlu",
        family="helm",
        description="Massive Multitask Language Understanding (57 subjects, multiple-choice)",
        response_type="binary",
        n_subjects=_N_SUBJECTS,
        n_items=13223,
        filename="helm/mmlu.pt",
        citation=(
            "@article{hendrycks2021measuring,\n"
            "  title={Measuring Massive Multitask Language Understanding},\n"
            "  author={Hendrycks, Dan and Burns, Collin and Basart, Steven and Zou, Andy and "
            "Mazeika, Mantas and Song, Dawn and Steinhardt, Jacob},\n"
            "  journal={Proceedings of ICLR},\n"
            "  year={2021}\n"
            "}"
        ),
        url=_HELM_URL,
        license="MIT",
        tags=["nlp", "multiple-choice", "knowledge", "reasoning"],
    )

    datasets["helm/gsm8k"] = DatasetInfo(
        name="helm/gsm8k",
        family="helm",
        description="Grade School Math 8K (math word problems)",
        response_type="binary",
        n_subjects=_N_SUBJECTS,
        n_items=997,
        filename="helm/gsm8k.pt",
        citation=(
            "@article{cobbe2021training,\n"
            "  title={Training Verifiers to Solve Math Word Problems},\n"
            "  author={Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and Chen, Mark and "
            "Jun, Heewoo and Kaiser, Lukasz and Plappert, Matthias and Tworek, Jerry and Hilton, Jacob and "
            "Nakano, Reiichiro and Hesse, Christopher and Schulman, John},\n"
            "  journal={arXiv preprint arXiv:2110.14168},\n"
            "  year={2021}\n"
            "}"
        ),
        url=_HELM_URL,
        license="MIT",
        tags=["nlp", "math", "reasoning"],
    )

    datasets["helm/math"] = DatasetInfo(
        name="helm/math",
        family="helm",
        description="MATH (competition mathematics problems)",
        response_type="binary",
        n_subjects=_N_SUBJECTS,
        n_items=436,
        filename="helm/math.pt",
        citation=(
            "@article{hendrycks2021measuring_math,\n"
            "  title={Measuring Mathematical Problem Solving With the MATH Dataset},\n"
            "  author={Hendrycks, Dan and Burns, Collin and Kadavath, Saurav and Arora, Akul and "
            "Basart, Steven and Tang, Eric and Song, Dawn and Steinhardt, Jacob},\n"
            "  journal={NeurIPS},\n"
            "  year={2021}\n"
            "}"
        ),
        url=_HELM_URL,
        license="MIT",
        tags=["nlp", "math", "reasoning"],
    )

    datasets["helm/truthfulqa"] = DatasetInfo(
        name="helm/truthfulqa",
        family="helm",
        description="TruthfulQA (measuring truthfulness of LLM generations)",
        response_type="binary",
        n_subjects=_N_SUBJECTS,
        n_items=1888,
        filename="helm/truthfulqa.pt",
        citation=(
            "@article{lin2022truthfulqa,\n"
            "  title={TruthfulQA: Measuring How Models Mimic Human Falsehoods},\n"
            "  author={Lin, Stephanie and Hilton, Jacob and Evans, Owain},\n"
            "  journal={Proceedings of ACL},\n"
            "  year={2022}\n"
            "}"
        ),
        url=_HELM_URL,
        license="Apache-2.0",
        tags=["nlp", "truthfulness", "factuality"],
    )

    datasets["helm/commonsense"] = DatasetInfo(
        name="helm/commonsense",
        family="helm",
        description="CommonsenseQA (commonsense reasoning, multiple-choice)",
        response_type="binary",
        n_subjects=_N_SUBJECTS,
        n_items=498,
        filename="helm/commonsense.pt",
        citation=_HELM_CITATION,
        url=_HELM_URL,
        license="MIT",
        tags=["nlp", "commonsense", "reasoning", "multiple-choice"],
    )

    datasets["helm/synthetic_reasoning"] = DatasetInfo(
        name="helm/synthetic_reasoning",
        family="helm",
        description="Synthetic reasoning (rule-following and pattern matching)",
        response_type="binary",
        n_subjects=_N_SUBJECTS,
        n_items=2234,
        filename="helm/synthetic_reasoning.pt",
        citation=_HELM_CITATION,
        url=_HELM_URL,
        license="MIT",
        tags=["nlp", "reasoning", "synthetic"],
    )

    datasets["helm/dyck_language"] = DatasetInfo(
        name="helm/dyck_language",
        family="helm",
        description="Dyck language (bracket matching, formal language understanding)",
        response_type="binary",
        n_subjects=_N_SUBJECTS,
        n_items=500,
        filename="helm/dyck_language.pt",
        citation=_HELM_CITATION,
        url=_HELM_URL,
        license="MIT",
        tags=["nlp", "formal-language", "reasoning"],
    )

    # --- Reading Comprehension & NLU ---

    datasets["helm/boolq"] = DatasetInfo(
        name="helm/boolq",
        family="helm",
        description="BoolQ (yes/no reading comprehension)",
        response_type="binary",
        n_subjects=_N_SUBJECTS,
        n_items=3316,
        filename="helm/boolq.pt",
        citation=(
            "@article{clark2019boolq,\n"
            "  title={BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions},\n"
            "  author={Clark, Christopher and Lee, Kenton and Chang, Ming-Wei and Kwiatkowski, Tom and "
            "Collins, Michael and Toutanova, Kristina},\n"
            "  journal={Proceedings of NAACL},\n"
            "  year={2019}\n"
            "}"
        ),
        url=_HELM_URL,
        license="CC-BY-SA-3.0",
        tags=["nlp", "reading-comprehension", "boolean"],
    )

    datasets["helm/imdb"] = DatasetInfo(
        name="helm/imdb",
        family="helm",
        description="IMDB (movie review sentiment classification)",
        response_type="binary",
        n_subjects=_N_SUBJECTS,
        n_items=3530,
        filename="helm/imdb.pt",
        citation=_HELM_CITATION,
        url=_HELM_URL,
        license="MIT",
        tags=["nlp", "sentiment", "classification"],
    )

    datasets["helm/civil_comments"] = DatasetInfo(
        name="helm/civil_comments",
        family="helm",
        description="CivilComments (toxicity classification)",
        response_type="binary",
        n_subjects=_N_SUBJECTS,
        n_items=29407,
        filename="helm/civil_comments.pt",
        citation=_HELM_CITATION,
        url=_HELM_URL,
        license="CC-BY-4.0",
        tags=["nlp", "toxicity", "classification"],
    )

    # --- Law & Medicine ---

    datasets["helm/lsat_qa"] = DatasetInfo(
        name="helm/lsat_qa",
        family="helm",
        description="LSAT QA (law school analytical reasoning questions)",
        response_type="binary",
        n_subjects=_N_SUBJECTS,
        n_items=454,
        filename="helm/lsat_qa.pt",
        citation=_HELM_CITATION,
        url=_HELM_URL,
        license="MIT",
        tags=["nlp", "law", "reasoning", "multiple-choice"],
    )

    datasets["helm/legalbench"] = DatasetInfo(
        name="helm/legalbench",
        family="helm",
        description="LegalBench (legal reasoning benchmark)",
        response_type="binary",
        n_subjects=_N_SUBJECTS,
        n_items=1997,
        filename="helm/legalbench.pt",
        citation=_HELM_CITATION,
        url=_HELM_URL,
        license="MIT",
        tags=["nlp", "law", "reasoning"],
    )

    datasets["helm/legal_support"] = DatasetInfo(
        name="helm/legal_support",
        family="helm",
        description="Legal Support (legal case outcome prediction)",
        response_type="binary",
        n_subjects=_N_SUBJECTS,
        n_items=594,
        filename="helm/legal_support.pt",
        citation=_HELM_CITATION,
        url=_HELM_URL,
        license="MIT",
        tags=["nlp", "law", "classification"],
    )

    datasets["helm/med_qa"] = DatasetInfo(
        name="helm/med_qa",
        family="helm",
        description="MedQA (medical question answering, USMLE-style)",
        response_type="binary",
        n_subjects=_N_SUBJECTS,
        n_items=998,
        filename="helm/med_qa.pt",
        citation=_HELM_CITATION,
        url=_HELM_URL,
        license="MIT",
        tags=["nlp", "medical", "multiple-choice"],
    )

    # --- Bias & Safety ---

    datasets["helm/bbq"] = DatasetInfo(
        name="helm/bbq",
        family="helm",
        description="BBQ (bias benchmark for question answering)",
        response_type="binary",
        n_subjects=_N_SUBJECTS,
        n_items=999,
        filename="helm/bbq.pt",
        citation=_HELM_CITATION,
        url=_HELM_URL,
        license="CC-BY-4.0",
        tags=["nlp", "bias", "fairness"],
    )

    datasets["helm/air_bench_2024"] = DatasetInfo(
        name="helm/air_bench_2024",
        family="helm",
        description="AIR-Bench 2024 (safety and responsibility evaluation)",
        response_type="binary",
        n_subjects=_N_SUBJECTS,
        n_items=4985,
        filename="helm/air_bench_2024.pt",
        citation=_HELM_CITATION,
        url=_HELM_URL,
        license="MIT",
        tags=["nlp", "safety", "responsibility"],
    )

    # --- QA & Information ---

    datasets["helm/babi_qa"] = DatasetInfo(
        name="helm/babi_qa",
        family="helm",
        description="bAbI QA (synthetic question answering, reasoning tasks)",
        response_type="binary",
        n_subjects=_N_SUBJECTS,
        n_items=3461,
        filename="helm/babi_qa.pt",
        citation=_HELM_CITATION,
        url=_HELM_URL,
        license="MIT",
        tags=["nlp", "qa", "reasoning", "synthetic"],
    )

    datasets["helm/wikifact"] = DatasetInfo(
        name="helm/wikifact",
        family="helm",
        description="WikiFact (factual knowledge from Wikipedia)",
        response_type="binary",
        n_subjects=_N_SUBJECTS,
        n_items=5511,
        filename="helm/wikifact.pt",
        citation=_HELM_CITATION,
        url=_HELM_URL,
        license="CC-BY-SA-3.0",
        tags=["nlp", "knowledge", "factual"],
    )

    # --- Data Tasks ---

    datasets["helm/entity_matching"] = DatasetInfo(
        name="helm/entity_matching",
        family="helm",
        description="Entity Matching (record linkage / deduplication)",
        response_type="binary",
        n_subjects=_N_SUBJECTS,
        n_items=1396,
        filename="helm/entity_matching.pt",
        citation=_HELM_CITATION,
        url=_HELM_URL,
        license="MIT",
        tags=["nlp", "data", "entity-resolution"],
    )

    datasets["helm/entity_data_imputation"] = DatasetInfo(
        name="helm/entity_data_imputation",
        family="helm",
        description="Entity Data Imputation (missing value prediction)",
        response_type="binary",
        n_subjects=_N_SUBJECTS,
        n_items=395,
        filename="helm/entity_data_imputation.pt",
        citation=_HELM_CITATION,
        url=_HELM_URL,
        license="MIT",
        tags=["nlp", "data", "imputation"],
    )

    datasets["helm/raft"] = DatasetInfo(
        name="helm/raft",
        family="helm",
        description="RAFT (Real-world Annotated Few-shot Tasks)",
        response_type="binary",
        n_subjects=_N_SUBJECTS,
        n_items=1336,
        filename="helm/raft.pt",
        citation=_HELM_CITATION,
        url=_HELM_URL,
        license="MIT",
        tags=["nlp", "few-shot", "classification"],
    )

    # --- Regional ---

    datasets["helm/thai_exam"] = DatasetInfo(
        name="helm/thai_exam",
        family="helm",
        description="Thai Exam (Thai-language national exam questions)",
        response_type="binary",
        n_subjects=_N_SUBJECTS,
        n_items=557,
        filename="helm/thai_exam.pt",
        citation=_HELM_CITATION,
        url=_HELM_URL,
        license="MIT",
        tags=["nlp", "multilingual", "thai", "exam"],
    )

    # --- Aggregate ---

    datasets["helm/all"] = DatasetInfo(
        name="helm/all",
        family="helm",
        description="All HELM benchmarks concatenated (183 models x 78,712 items across 22 benchmarks)",
        response_type="binary",
        n_subjects=_N_SUBJECTS,
        n_items=78712,
        filename="helm/all.pt",
        citation=_HELM_CITATION,
        url=_HELM_URL,
        license="MIT",
        tags=["nlp", "aggregate", "multi-benchmark"],
    )

    return datasets

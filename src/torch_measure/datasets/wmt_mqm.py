# Copyright (c) 2026 AIMS Foundations. MIT License.

"""WMT MQM (Multidimensional Quality Metrics) dataset definitions."""

from __future__ import annotations

from torch_measure.datasets._info import DatasetInfo

_WMT_MQM_URL = "https://huggingface.co/datasets/RicardoRei/wmt-mqm-human-evaluation"

_WMT_MQM_CITATION = (
    "@inproceedings{freitag2021experts,\n"
    "  title={Experts, Errors, and Context: A Large-Scale Study of Human\n"
    "         Evaluation for Machine Translation},\n"
    "  author={Freitag, Markus and Foster, George and Grangier, David and\n"
    "          Ratnakar, Viresh and Tan, Qijun and Macherey, Wolfgang},\n"
    "  booktitle={Transactions of the Association for Computational Linguistics},\n"
    "  volume={9},\n"
    "  pages={1460--1474},\n"
    "  year={2021},\n"
    "  publisher={MIT Press},\n"
    "  url={https://doi.org/10.1162/tacl_a_00437}\n"
    "}"
)


def _register_wmt_mqm_datasets() -> dict[str, DatasetInfo]:
    """Return registry entries for WMT MQM response matrices.

    WMT MQM provides expert human evaluation of machine translation quality
    from WMT shared tasks (2020-2022).  Multiple human annotators rate
    translations using fine-grained error categories, producing segment-level
    quality scores.

    Each response matrix has:

    - **Rows**: MT systems (translation engines / models).
    - **Columns**: Source segments to be translated.
    - **Values**: Continuous MQM scores (mean across annotators per segment).
      Lower (more negative) scores indicate more errors; 0.0 is a perfect
      translation.

    Source data: ``RicardoRei/wmt-mqm-human-evaluation`` on HuggingFace Hub.
    """
    datasets: dict[str, DatasetInfo] = {}

    # ------------------------------------------------------------------
    # Per year + language-pair splits
    # ------------------------------------------------------------------

    # --- 2020 ---

    datasets["2020_en_de"] = DatasetInfo(
        name="2020_en_de",
        family="wmt_mqm",
        description="WMT 2020 MQM en-de: 10 systems x 1418 segments",
        response_type="continuous",
        n_subjects=10,
        n_items=1418,
        subject_entity="system",
        item_entity="segment",
        filename="2020_en_de.pt",
        citation=_WMT_MQM_CITATION,
        url=_WMT_MQM_URL,
        license="CC-BY-4.0",
        tags=["mt", "mqm", "en-de", "2020", "human-eval"],
    )

    datasets["2020_zh_en"] = DatasetInfo(
        name="2020_zh_en",
        family="wmt_mqm",
        description="WMT 2020 MQM zh-en: 10 systems x 2000 segments",
        response_type="continuous",
        n_subjects=10,
        n_items=2000,
        subject_entity="system",
        item_entity="segment",
        filename="2020_zh_en.pt",
        citation=_WMT_MQM_CITATION,
        url=_WMT_MQM_URL,
        license="CC-BY-4.0",
        tags=["mt", "mqm", "zh-en", "2020", "human-eval"],
    )

    # --- 2021 ---

    datasets["2021_en_de"] = DatasetInfo(
        name="2021_en_de",
        family="wmt_mqm",
        description="WMT 2021 MQM en-de: 17 systems x 1050 segments",
        response_type="continuous",
        n_subjects=17,
        n_items=1050,
        subject_entity="system",
        item_entity="segment",
        filename="2021_en_de.pt",
        citation=_WMT_MQM_CITATION,
        url=_WMT_MQM_URL,
        license="CC-BY-4.0",
        tags=["mt", "mqm", "en-de", "2021", "human-eval"],
    )

    datasets["2021_en_ru"] = DatasetInfo(
        name="2021_en_ru",
        family="wmt_mqm",
        description="WMT 2021 MQM en-ru: 16 systems x 1031 segments",
        response_type="continuous",
        n_subjects=16,
        n_items=1031,
        subject_entity="system",
        item_entity="segment",
        filename="2021_en_ru.pt",
        citation=_WMT_MQM_CITATION,
        url=_WMT_MQM_URL,
        license="CC-BY-4.0",
        tags=["mt", "mqm", "en-ru", "2021", "human-eval"],
    )

    datasets["2021_zh_en"] = DatasetInfo(
        name="2021_zh_en",
        family="wmt_mqm",
        description="WMT 2021 MQM zh-en: 16 systems x 1173 segments",
        response_type="continuous",
        n_subjects=16,
        n_items=1173,
        subject_entity="system",
        item_entity="segment",
        filename="2021_zh_en.pt",
        citation=_WMT_MQM_CITATION,
        url=_WMT_MQM_URL,
        license="CC-BY-4.0",
        tags=["mt", "mqm", "zh-en", "2021", "human-eval"],
    )

    # --- 2022 ---

    datasets["2022_en_de"] = DatasetInfo(
        name="2022_en_de",
        family="wmt_mqm",
        description="WMT 2022 MQM en-de: 15 systems x 1215 segments",
        response_type="continuous",
        n_subjects=15,
        n_items=1215,
        subject_entity="system",
        item_entity="segment",
        filename="2022_en_de.pt",
        citation=_WMT_MQM_CITATION,
        url=_WMT_MQM_URL,
        license="CC-BY-4.0",
        tags=["mt", "mqm", "en-de", "2022", "human-eval"],
    )

    datasets["2022_en_ru"] = DatasetInfo(
        name="2022_en_ru",
        family="wmt_mqm",
        description="WMT 2022 MQM en-ru: 15 systems x 1215 segments",
        response_type="continuous",
        n_subjects=15,
        n_items=1215,
        subject_entity="system",
        item_entity="segment",
        filename="2022_en_ru.pt",
        citation=_WMT_MQM_CITATION,
        url=_WMT_MQM_URL,
        license="CC-BY-4.0",
        tags=["mt", "mqm", "en-ru", "2022", "human-eval"],
    )

    datasets["2022_zh_en"] = DatasetInfo(
        name="2022_zh_en",
        family="wmt_mqm",
        description="WMT 2022 MQM zh-en: 15 systems x 1871 segments",
        response_type="continuous",
        n_subjects=15,
        n_items=1871,
        subject_entity="system",
        item_entity="segment",
        filename="2022_zh_en.pt",
        citation=_WMT_MQM_CITATION,
        url=_WMT_MQM_URL,
        license="CC-BY-4.0",
        tags=["mt", "mqm", "zh-en", "2022", "human-eval"],
    )

    return datasets

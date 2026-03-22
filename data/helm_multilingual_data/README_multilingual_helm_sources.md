# HELM Multilingual Per-Instance Evaluation Data Sources

Last updated: 2026-03-21

## Overview

This document catalogs all available sources of HELM per-instance evaluation data
for non-English benchmarks. These are critical for validity analysis when porting
benchmarks across cultures and regions.

---

## 1. stair-lab/reeval (HuggingFace)

**URL:** https://huggingface.co/datasets/stair-lab/reeval
**Paper:** [Reliable and Efficient Amortized Model-based Evaluation](https://arxiv.org/abs/2503.13335)
**Source file:** `resmat.pkl` (pickled pandas DataFrame, ~507 MB)
**Existing pipeline:** `/lfs/skampere1/0/sttruong/torch_measure/scripts/migrate_helm_data.py`

### Benchmarks in stair-lab/reeval (22 total):
| Benchmark | English? | Multilingual? |
|-----------|----------|---------------|
| mmlu | Yes | No |
| gsm | Yes | No |
| truthful_qa | Yes | No |
| boolq | Yes | No |
| commonsense | Yes | No |
| math | Yes | No |
| lsat_qa | Yes | No |
| med_qa | Yes | No |
| legalbench | Yes | No |
| imdb | Yes | No |
| civil_comments | Yes | No |
| bbq | Yes | No |
| air_bench_2024 | Yes | No |
| babi_qa | Yes | No |
| raft | Yes | No |
| wikifact | Yes | No |
| synthetic_reasoning | Yes | No |
| entity_matching | Yes | No |
| entity_data_imputation | Yes | No |
| dyck_language | Yes | No |
| legal_support | Yes | No |
| **thai_exam** | **No** | **Thai** |

**Key finding:** Only `thai_exam` is non-English. The response matrix has 183 models
and 5.69M data points. The `resmat.pkl` contains per-instance binary correctness
scores in a (models x items) matrix.

### Related datasets in stair-lab:
- `stair-lab/reeval-sft` - SFT training data derived from HELM
- `stair-lab/reeval_fa` - Factor analysis results (not raw data)
- `stair-lab/reeval-difficulty-for-helm` - Item difficulty parameters (217k items)

---

## 2. HELM GCS Bucket: Non-English Projects

**Bucket:** `gs://crfm-helm-public/`
**Access:** Public, unauthenticated
**Download tool:** `gcloud storage rsync`

### 2a. ThaiExam (`gs://crfm-helm-public/thaiexam/benchmark_output/`)

**Leaderboard:** https://crfm.stanford.edu/helm/thaiexam/latest/
**Blog post:** https://crfm.stanford.edu/2024/09/04/thaiexam.html
**Language:** Thai
**Versions:** v1.0.0, v1.1.0, v1.2.0-preview, v1.2.0, v1.3.0-preview

**Exam types (5):**
- A-Level (Thai university entrance)
- IC (Investment Consultant license)
- ONET (Ordinary National Educational Test)
- TGAT (Thai General Aptitude Test)
- TPAT1 (Thai Professional Aptitude Test 1)

**Models in v1.2.0:** Only 2 (nectec_OpenThaiLLM-Prebuilt-7B, nectec_Pathumma-llm-text-1.0.0)
- Note: The leaderboard shows 34+ models but not all have per-instance data in GCS.
  The remaining model data is in the stair-lab/reeval dataset.

**Per-instance files per run:**
- `instances.json` - Questions with Thai text (162 items for ONET)
- `per_instance_stats.json` - Per-question metrics
- `scenario_state.json` - Full request/response pairs
- `display_predictions.json` - Model predictions
- `display_requests.json` - Prompts sent to models
- `run_spec.json`, `scenario.json`, `stats.json`

**Download example:**
```bash
gcloud storage rsync -r \
  gs://crfm-helm-public/thaiexam/benchmark_output/ \
  ./helm_multilingual_data/thaiexam/
```

### 2b. CLEVA - Chinese Language Evaluation (`gs://crfm-helm-public/cleva/benchmark_output/`)

**Language:** Chinese (Mandarin, Classical Chinese)
**Version:** v1.0.0
**Total runs:** 80 (21 tasks x 4 models)
**Paper:** [CLEVA: Chinese Language Models EVAluation Platform](https://github.com/LaVi-Lab/CLEVA) (EMNLP 2023)

**Tasks (21):**
1. Classical Chinese understanding
2. Commonsense reasoning (textual entailment)
3. Coreference resolution
4. Cultural knowledge (idiom)
5. Mathematical calculation (add)
6. Mathematical calculation (multiply)
7. Mathematical calculation (subtract)
8. Mathematical reasoning (word problems)
9. Paraphrase generation
10. Paraphrase identification (financial)
11. Paraphrase identification (short utterance)
12. Pinyin transliteration (pinyin to Chinese)
13. Pinyin transliteration (Chinese to pinyin)
14. Reading comprehension
15. Sentiment analysis
16. Summarization (dialogue)
17. Text classification (news)
18. Text classification (humor)
19. Toxicity detection
20. Translation (English to Chinese)
21. Translation (Chinese to English)

**Models (4):** vicuna-13b-v1.3, llama-65b, gpt-3.5-turbo-0613, gpt-4-0613

**Download:**
```bash
gcloud storage rsync -r \
  gs://crfm-helm-public/cleva/benchmark_output/ \
  ./helm_multilingual_data/cleva/
```

### 2c. African Languages MMLU & Winogrande (`gs://crfm-helm-public/mmlu-winogrande-afr/benchmark_output/`)

**Languages (11):**
- Afrikaans (af), Amharic (am), Bambara (bm), Igbo (ig),
  Sepedi (nso), Shona (sn), Sesotho (st), Setswana (tn),
  Tsonga (ts), Xhosa (xh), Zulu (zu)

**Versions:** v1.0.0, v1.1.0-preview, v1.1.0
**Total runs in v1.1.0:** 308

**Benchmarks (2):**
- `mmlu_clinical_afr` - Clinical knowledge, College medicine, Virology (translated to African languages)
- `winogrande_afr` - Winogrande commonsense reasoning (translated to African languages)

**Models (7):**
- anthropic/claude-3-5-haiku-20241022
- anthropic/claude-3-7-sonnet-20250219
- google/gemini-2.0-flash-001
- google/gemini-2.0-flash-lite-001
- meta/llama-3.1-70b-instruct-turbo
- meta/llama-3.1-8b-instruct-turbo
- qwen/qwen2.5-72b-instruct-turbo

**Download:**
```bash
gcloud storage rsync -r \
  gs://crfm-helm-public/mmlu-winogrande-afr/benchmark_output/ \
  ./helm_multilingual_data/mmlu_winogrande_afr/
```

---

## 3. HELM Arabic (Announced but NOT yet in GCS)

**Leaderboard:** https://crfm.stanford.edu/helm/arabic/latest/
**Blog post:** https://crfm.stanford.edu/2025/12/18/helm-arabic.html
**Status:** Leaderboard exists, but NO data found in GCS bucket (no `arabic/` prefix)

**Benchmarks (7):**
1. AlGhafa - Arabic NLP evaluation
2. ArabicMMLU - Native Arabic school exams
3. Arabic EXAMS - High school exams
4. MadinahQA - Arabic grammar/language
5. AraTrust - Arabic safety evaluation
6. ALRAGE - Arabic RAG evaluation
7. ArbMMLU-HT - Human-translated MMLU to Arabic

**Action needed:** Contact Stanford CRFM to request GCS upload, or run HELM
Arabic evaluations locally using the helm framework config files:
- `run_entries_arabic.conf`
- `schema_arabic.yaml`

---

## 4. SEA-HELM (Separate from HELM GCS)

**Leaderboard:** https://crfm.stanford.edu/helm/seahelm/latest/ and https://leaderboard.sea-lion.ai/
**Paper:** https://arxiv.org/abs/2502.14301 (Findings of ACL 2025)
**GitHub:** https://github.com/aisingapore/SEA-HELM
**Status:** NOT in GCS bucket (no `seahelm/` prefix). Data available via HuggingFace.

**Languages:** Filipino, Indonesian, Tamil, Thai, Vietnamese

**HuggingFace datasets (13):**
https://huggingface.co/collections/aisingapore/sea-helm-evaluation-datasets-67593d0bb8c9f17f9f6b0fcb

| Dataset | Rows |
|---------|------|
| Instruction-Following-IFEval | 1,140 |
| MultiTurn-Chat-MT-Bench | 151 |
| Cultural-Evaluation-Kalahi | 332 |
| Linguistic-Diagnostics-Pragmatics | 584 |
| Linguistic-Diagnostics-Syntax | 555 |
| NLR-Causal-Reasoning | 1,100 |
| NLR-NLI | 1,180 |
| NLG-Abstractive-Summarization | 1,110 |
| NLG-Machine-Translation | 1,360 |
| NLU-Metaphor | 295 |
| NLU-Question-Answering | 804 |
| NLU-Sentiment-Analysis | 1,290 |
| Safety-Toxicity-Detection | 1,220 |

**Note:** These are INPUT datasets, not per-instance evaluation results.
To get per-instance results, you would need to run SEA-HELM evaluations locally.

---

## 5. IberBench (Independent, not HELM)

**Leaderboard:** https://huggingface.co/spaces/iberbench/leaderboard
**Paper:** https://arxiv.org/abs/2504.16921
**GitHub:** https://github.com/IberBench/
**Languages:** Spanish (multiple varieties), Portuguese, Catalan, Basque, Galician

**Key facts:**
- 101 datasets across 22 task categories
- Test splits are kept PRIVATE to prevent leakage
- One public dataset: `iberbench/iberbench_all` (388k items)
- Per-instance evaluation data NOT publicly available

---

## 6. Complete GCS Bucket Directory (gs://crfm-helm-public/)

All 26 top-level directories:

| Prefix | Content | Non-English? |
|--------|---------|-------------|
| air-bench/ | AI safety | No |
| audio/ | Audio HELM | No (mostly English) |
| benchmark_output/ | Classic HELM | No (English only) |
| capabilities/ | HELM Capabilities | No |
| **cleva/** | **Chinese CLEVA** | **Yes - Chinese** |
| config/ | Configuration | N/A |
| ewok/ | EWoK benchmark | No |
| finance/ | Finance HELM | Unknown |
| gzip/ | Compressed data | Unknown |
| heim/ | Image generation | No |
| image2struct/ | Image to structure | No |
| instruct/ | HELM Instruct | No |
| lite/ | HELM Lite | No |
| long-context/ | Long context eval | No |
| medhelm/ | Medical HELM | No |
| **mmlu-winogrande-afr/** | **African languages** | **Yes - 11 African langs** |
| mmlu/ | MMLU-focused | No |
| prod_env/ | Production env | N/A |
| reasoning/ | Reasoning evals | No |
| robo-reward-bench/ | Reward modeling | No |
| safety/ | Safety evals | No |
| source_datasets/ | Source data | N/A |
| speech/ | Speech evals | Unknown |
| **thaiexam/** | **Thai exams** | **Yes - Thai** |
| torr/ | ToRR benchmark | No |
| vhelm/ | Visual HELM | No |

---

## Summary: Available Non-English Per-Instance HELM Data

| Source | Languages | Models | Items | Per-Instance? | Access |
|--------|-----------|--------|-------|---------------|--------|
| stair-lab/reeval (thai_exam) | Thai | 183 | varies | Yes (response matrix) | HuggingFace (private) |
| GCS thaiexam/ | Thai | 2-5 | 162+/exam | Yes (JSON) | Public GCS |
| GCS cleva/ | Chinese | 4 | varies/task | Yes (JSON) | Public GCS |
| GCS mmlu-winogrande-afr/ | 11 African langs | 7 | varies | Yes (JSON) | Public GCS |
| HELM Arabic | Arabic | TBD | TBD | NOT YET | Not in GCS |
| SEA-HELM | 5 SEA langs | TBD | TBD | Input data only | HuggingFace |
| IberBench | 6 Iberian langs | TBD | TBD | Private | Not available |

---

## Recommended Download Script

```bash
# Install gcloud if not available
# pip install google-cloud-storage

# 1. Thai Exam (all versions)
gcloud storage rsync -r \
  gs://crfm-helm-public/thaiexam/benchmark_output/ \
  /lfs/skampere1/0/sttruong/torch_measure/data/helm_multilingual_data/thaiexam/

# 2. Chinese CLEVA
gcloud storage rsync -r \
  gs://crfm-helm-public/cleva/benchmark_output/ \
  /lfs/skampere1/0/sttruong/torch_measure/data/helm_multilingual_data/cleva/

# 3. African Languages MMLU + Winogrande
gcloud storage rsync -r \
  gs://crfm-helm-public/mmlu-winogrande-afr/benchmark_output/ \
  /lfs/skampere1/0/sttruong/torch_measure/data/helm_multilingual_data/mmlu_winogrande_afr/

# 4. stair-lab/reeval (requires HF token with access)
# Already handled by migrate_helm_data.py
```

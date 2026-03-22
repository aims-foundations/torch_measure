#!/usr/bin/env python3
"""
Build response matrices for legal benchmarks from per-item model predictions.

Data sources:
  1. LawBench (Chinese Legal) — github.com/open-compass/LawBench
     - 51 models, 18 tasks (20 in paper, 18 with prediction data), 500 items/task
     - EMNLP 2024. Tasks span memorization, understanding, application of Chinese law.
     - Prediction format: JSON dict keyed by str(index), each with
       origin_prompt, prediction, refr fields.
     - Evaluation: accuracy for MC tasks, F1 for multi-label, ROUGE-L for generation,
       F0.5 for proofreading, normalized log-distance for sentencing.

  2. LexEval (Chinese Legal) — github.com/CSHaitao/LexEval
     - 38 models, 23 tasks, 14,150 questions total (varying items/task)
     - NeurIPS 2024 Datasets & Benchmarks Track.
     - Prediction format: JSONL with input, output, answer fields.
     - Evaluation: accuracy for MC (categories 1-4,6), ROUGE-L for generation (category 5).

For the response matrix we compute per-item scores:
  - MC tasks (accuracy): binary {0, 1} — exact match of extracted answer vs reference.
  - MC tasks (F1): continuous [0, 1] — per-item F1 for multi-label predictions.
  - Generation tasks (ROUGE-L): continuous [0, 1] — per-item ROUGE-L F-score.
  - Sentencing tasks: continuous [0, 1] — normalized log-distance score.
  - Proofreading (task 2-1 in LawBench): omitted (requires external tools).

Output files in processed/:
  - lawbench_response_matrix.csv    : model x item response matrix (rows=models, cols=items)
  - lexeval_response_matrix.csv     : model x item response matrix
  - response_matrix.csv             : combined (both benchmarks, prefixed item IDs)
  - task_metadata.csv               : item_id, task_id, benchmark, language, category, metric
  - model_summary.csv               : model, benchmark, num_items, mean_score

All paths use Path(__file__).resolve().parent.parent
"""

import json
import os
import re
import sys
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = _BENCHMARK_DIR / "raw"
PROCESSED_DIR = _BENCHMARK_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

LAWBENCH_DIR = RAW_DIR / "LawBench"
LEXEVAL_DIR = RAW_DIR / "LexEval"

# ──────────────────────────────────────────────────────────────────────
# LawBench task metadata
# ──────────────────────────────────────────────────────────────────────
LAWBENCH_TASKS = {
    "1-1": {"name": "Article Recitation", "category": "memorization",
            "metric": "rouge-l", "type": "generation"},
    "1-2": {"name": "Knowledge QA", "category": "memorization",
            "metric": "accuracy", "type": "mc_single"},
    "2-1": {"name": "Document Proofread", "category": "understanding",
            "metric": "f0.5", "type": "generation"},
    "2-2": {"name": "Dispute Focus ID", "category": "understanding",
            "metric": "accuracy", "type": "mc_single"},
    "2-3": {"name": "Marital Disputes ID", "category": "understanding",
            "metric": "f1_multilabel", "type": "mc_multi"},
    "2-4": {"name": "Issue Topic ID", "category": "understanding",
            "metric": "accuracy", "type": "mc_single"},
    "2-5": {"name": "Reading Comprehension", "category": "understanding",
            "metric": "rc-f1", "type": "extraction"},
    "2-6": {"name": "Named Entity Recognition", "category": "understanding",
            "metric": "soft-f1", "type": "extraction"},
    "2-7": {"name": "Opinion Summarization", "category": "understanding",
            "metric": "rouge-l", "type": "generation"},
    "2-8": {"name": "Argument Mining", "category": "understanding",
            "metric": "accuracy", "type": "mc_single"},
    "2-9": {"name": "Event Detection", "category": "understanding",
            "metric": "f1_multilabel", "type": "mc_multi"},
    "2-10": {"name": "Trigger Word Extraction", "category": "understanding",
             "metric": "soft-f1", "type": "extraction"},
    "3-1": {"name": "Article Prediction (fact)", "category": "application",
            "metric": "f1_multilabel", "type": "mc_multi"},
    "3-2": {"name": "Article Prediction (scene)", "category": "application",
            "metric": "rouge-l", "type": "generation"},
    "3-3": {"name": "Charge Prediction", "category": "application",
            "metric": "f1_multilabel", "type": "mc_multi"},
    "3-4": {"name": "Prison Term Prediction (w/o art.)", "category": "application",
            "metric": "log-distance", "type": "regression"},
    "3-5": {"name": "Prison Term Prediction (w/ art.)", "category": "application",
            "metric": "log-distance", "type": "regression"},
    "3-6": {"name": "Case Analysis", "category": "application",
            "metric": "accuracy", "type": "mc_single"},
    "3-7": {"name": "Criminal Damages Calculation", "category": "application",
            "metric": "accuracy", "type": "extraction"},
    "3-8": {"name": "Legal Consultation", "category": "application",
            "metric": "rouge-l", "type": "generation"},
}

# LexEval task metadata — categories from the paper's taxonomy
LEXEVAL_TASKS = {
    "1_1": {"name": "Legal Knowledge MC (single)", "category": "legal_knowledge",
            "metric": "accuracy", "type": "mc"},
    "1_2": {"name": "Legal Knowledge MC (multi)", "category": "legal_knowledge",
            "metric": "accuracy", "type": "mc"},
    "1_3": {"name": "Legal Knowledge Judgment", "category": "legal_knowledge",
            "metric": "accuracy", "type": "mc"},
    "2_1": {"name": "Legal NLU Sentiment", "category": "legal_nlu",
            "metric": "accuracy", "type": "mc"},
    "2_2": {"name": "Legal NLU Event Detection", "category": "legal_nlu",
            "metric": "accuracy", "type": "mc"},
    "2_3": {"name": "Legal NLU Reading Comp.", "category": "legal_nlu",
            "metric": "accuracy", "type": "mc"},
    "2_4": {"name": "Legal NLU Relation Extraction", "category": "legal_nlu",
            "metric": "accuracy", "type": "mc"},
    "2_5": {"name": "Legal NLU NER", "category": "legal_nlu",
            "metric": "accuracy", "type": "mc"},
    "3_1": {"name": "Legal Discrimination Article", "category": "legal_discrimination",
            "metric": "accuracy", "type": "mc"},
    "3_2": {"name": "Legal Discrimination Charge", "category": "legal_discrimination",
            "metric": "accuracy", "type": "mc"},
    "3_3": {"name": "Legal Discrimination Prison Term", "category": "legal_discrimination",
            "metric": "accuracy", "type": "mc"},
    "3_4": {"name": "Legal Discrimination Similar Case", "category": "legal_discrimination",
            "metric": "accuracy", "type": "mc"},
    "3_5": {"name": "Legal Discrimination Dispute Focus", "category": "legal_discrimination",
            "metric": "accuracy", "type": "mc"},
    "3_6": {"name": "Legal Discrimination Consultation", "category": "legal_discrimination",
            "metric": "accuracy", "type": "mc"},
    "4_1": {"name": "Legal Reasoning Judgment Pred.", "category": "legal_reasoning",
            "metric": "accuracy", "type": "mc"},
    "4_2": {"name": "Legal Reasoning Crime Amount", "category": "legal_reasoning",
            "metric": "accuracy", "type": "mc"},
    "5_1": {"name": "Legal Generation Summary", "category": "legal_generation",
            "metric": "rouge-l", "type": "generation"},
    "5_2": {"name": "Legal Generation Consultation", "category": "legal_generation",
            "metric": "rouge-l", "type": "generation"},
    "5_3": {"name": "Legal Generation Doc Writing", "category": "legal_generation",
            "metric": "rouge-l", "type": "generation"},
    "5_4": {"name": "Legal Generation Argument", "category": "legal_generation",
            "metric": "rouge-l", "type": "generation"},
    "6_1": {"name": "Legal Ethics Morality", "category": "legal_ethics",
            "metric": "accuracy", "type": "mc"},
    "6_2": {"name": "Legal Ethics Prof. Responsibility", "category": "legal_ethics",
            "metric": "accuracy", "type": "mc"},
    "6_3": {"name": "Legal Ethics Fairness", "category": "legal_ethics",
            "metric": "accuracy", "type": "mc"},
}


# ══════════════════════════════════════════════════════════════════════
# LawBench per-item scoring functions
# ══════════════════════════════════════════════════════════════════════

def _lawbench_multi_choice_judge(prediction, option_list, answer_token):
    """Replicate LawBench's multi_choice_judge: binary 0/1."""
    count_dict = {}
    for option in option_list:
        count_dict[option] = 1 if prediction.count(option) > 0 else 0
    if sum(count_dict.values()) == 0:
        return np.nan  # abstention
    elif count_dict[answer_token] == 1 and sum(count_dict.values()) == 1:
        return 1.0
    return 0.0


def _lawbench_f1_multilabel(prediction, reference, option_list):
    """Per-item F1 for multi-label classification tasks."""
    pred_set = set()
    for opt in option_list:
        if opt in prediction:
            pred_set.add(opt)
    ref_set = set()
    for opt in option_list:
        if opt in reference:
            ref_set.add(opt)
    if len(pred_set) == 0 and len(ref_set) == 0:
        return 1.0
    if len(pred_set) == 0 or len(ref_set) == 0:
        return 0.0
    precision = len(pred_set & ref_set) / len(pred_set)
    recall = len(pred_set & ref_set) / len(ref_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _lawbench_rouge_l(prediction, reference):
    """Per-item ROUGE-L using simple character-level LCS (no jieba dependency)."""
    # Simple character-level ROUGE-L for Chinese text
    pred_chars = list(prediction.strip())
    ref_chars = list(reference.strip())
    if not pred_chars or not ref_chars:
        return 0.0

    # LCS length
    m, n = len(pred_chars), len(ref_chars)
    # Optimize: limit to first 2000 chars to avoid huge computations
    pred_chars = pred_chars[:2000]
    ref_chars = ref_chars[:2000]
    m, n = len(pred_chars), len(ref_chars)

    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if pred_chars[i - 1] == ref_chars[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr
    lcs_len = prev[n]

    precision = lcs_len / m if m > 0 else 0
    recall = lcs_len / n if n > 0 else 0
    beta = 1.0  # ROUGE-L standard beta
    if precision + recall == 0:
        return 0.0
    f1 = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
    return f1


def score_lawbench_item(task_id, prediction, reference):
    """Score a single LawBench item. Returns float in [0,1] or NaN."""
    meta = LAWBENCH_TASKS.get(task_id)
    if meta is None:
        return np.nan

    metric = meta["metric"]

    if metric == "accuracy" and task_id == "1-2":
        # Knowledge QA: ABCD options, answer in "正确答案：X。"
        option_list = ["A", "B", "C", "D"]
        if "正确答案：" in reference and len(reference) > 5:
            answer_letter = reference[5]
        elif "正确答案:" in reference and len(reference) > 5:
            answer_letter = reference[5]
        else:
            return np.nan
        return _lawbench_multi_choice_judge(prediction, option_list, answer_letter)

    elif metric == "accuracy" and task_id == "2-2":
        # Dispute focus: 16 category options
        option_list = ["诉讼主体", "租金情况", "利息", "本金争议", "责任认定", "责任划分",
                       "损失认定及处理", "原审判决是否适当", "合同效力", "财产分割",
                       "责任承担", "鉴定结论采信问题", "诉讼时效", "违约", "合同解除", "肇事逃逸"]
        for opt in option_list:
            if opt in reference:
                return 1.0 if opt in prediction and sum(1 for o in option_list if o in prediction) == 1 else 0.0
        return np.nan

    elif metric == "accuracy" and task_id == "2-4":
        # Issue topic classification: 20 categories
        option_list = ['婚姻家庭', '劳动纠纷', '交通事故', '债权债务', '刑事辩护',
                       '合同纠纷', '房产纠纷', '侵权', '公司法', '医疗纠纷',
                       '拆迁安置', '行政诉讼', '建设工程', '知识产权', '综合咨询',
                       '人身损害', '涉外法律', '海事海商', '消费权益', '抵押担保']
        ref_cat = reference.strip()
        # Extract predicted category
        pred_cats = [o for o in option_list if o in prediction]
        if len(pred_cats) == 0:
            return np.nan  # abstention
        return 1.0 if ref_cat in pred_cats and len(pred_cats) == 1 else 0.0

    elif metric == "accuracy" and task_id == "2-8":
        # Argument mining: ABCDE
        option_list = ["A", "B", "C", "D", "E"]
        if "[正确答案]" in reference and len(reference) > 6:
            answer_letter = reference[6]
        else:
            return np.nan
        return _lawbench_multi_choice_judge(prediction, option_list, answer_letter)

    elif metric == "accuracy" and task_id == "3-6":
        # Case analysis: ABCD
        option_list = ["A", "B", "C", "D"]
        if "正确答案:" in reference and len(reference) > 5:
            answer_letter = reference[5]
        elif "正确答案：" in reference and len(reference) > 5:
            answer_letter = reference[5]
        else:
            return np.nan
        return _lawbench_multi_choice_judge(prediction, option_list, answer_letter)

    elif metric == "accuracy" and task_id == "3-7":
        # Criminal damages: extract number
        if "上文涉及到的犯罪金额:" in reference and "元。" in reference:
            amount_str = reference.replace("上文涉及到的犯罪金额:", "").replace("元。", "").strip()
            return 1.0 if amount_str in prediction else 0.0
        return np.nan

    elif metric in ("rouge-l",):
        ref_clean = reference.replace("答案:", "").strip()
        return _lawbench_rouge_l(prediction, ref_clean)

    elif metric == "f1_multilabel" and task_id == "2-3":
        # Marital disputes multi-label
        option_list = ["婚后有子女", "限制行为能力子女抚养", "有夫妻共同财产", "支付抚养费",
                       "不动产分割", "婚后分局", "二次起诉离婚", "按月给付抚养费",
                       "准予离婚", "有夫妻共同债务", "婚前个人财产", "法定离婚",
                       "不履行家庭义务", "存在非婚生子", "适当帮助", "不履行离婚协议",
                       "损害赔偿", "感情不和分居满二年", "子女随非抚养权人生活", "婚后个人财产"]
        return _lawbench_f1_multilabel(prediction, reference, option_list)

    elif metric == "f1_multilabel" and task_id == "2-9":
        # Event detection
        option_list = ["支付/给付", "欺骗", "搜查/扣押", "要求/请求", "卖出", "买入",
                       "获利", "拘捕", "鉴定", "同意/接受", "供述", "联络", "帮助/救助",
                       "租用/借用", "受伤", "伪造", "卖淫", "伤害人身", "赔偿", "归还/偿还"]
        return _lawbench_f1_multilabel(prediction, reference, option_list)

    elif metric == "f1_multilabel" and task_id in ("3-1", "3-3"):
        # Article/charge prediction — these have large option sets
        # For simplicity, use character-overlap F1 on the predicted answer tokens
        pred_tokens = set(re.findall(r'[\u4e00-\u9fff]+', prediction))
        ref_tokens = set(re.findall(r'[\u4e00-\u9fff]+', reference))
        if not pred_tokens and not ref_tokens:
            return 1.0
        if not pred_tokens or not ref_tokens:
            return 0.0
        precision = len(pred_tokens & ref_tokens) / len(pred_tokens)
        recall = len(pred_tokens & ref_tokens) / len(ref_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    elif metric in ("rc-f1", "soft-f1"):
        # Reading comprehension / NER — character-level F1
        pred_chars = set(prediction.strip())
        ref_chars = set(reference.strip())
        if not pred_chars and not ref_chars:
            return 1.0
        if not pred_chars or not ref_chars:
            return 0.0
        precision = len(pred_chars & ref_chars) / len(pred_chars)
        recall = len(pred_chars & ref_chars) / len(ref_chars)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    elif metric == "log-distance":
        # Prison term: normalized log-distance
        try:
            if "死刑" in reference or "无期" in reference:
                return np.nan
            ref_str = reference.replace("刑期:", "").replace("个月", "").strip()
            ref_months = int(ref_str)
            # Try to extract a number from prediction
            nums = re.findall(r'\d+', prediction)
            if not nums:
                return np.nan
            pred_months = int(nums[0])
            if ref_months == 0:
                return 1.0 if pred_months == 0 else 0.0
            import math
            dist = abs(math.log(pred_months + 1) - math.log(ref_months + 1))
            # Normalize: score = max(0, 1 - dist / log(max_months+1))
            # Use 360 months (30 years) as max
            max_dist = math.log(361)
            score = max(0.0, 1.0 - dist / max_dist)
            return score
        except (ValueError, ZeroDivisionError):
            return np.nan

    elif metric == "f0.5":
        # Document proofreading — skip (requires external ChERRANT tool)
        return np.nan

    return np.nan


# ══════════════════════════════════════════════════════════════════════
# LexEval per-item scoring functions
# ══════════════════════════════════════════════════════════════════════

def _lexeval_find_valid_substrings(s):
    """Replicate LexEval's answer extraction for MC tasks."""
    s = s.split('解析')[0].split('分析')[0]
    s = s.replace("、", "").replace(".", "").replace(",", "").replace(";", "")
    s = s.replace("，", "").replace("和", "").replace(", ", "")
    pattern = r'[ABCDE]{1,5}'
    substrings = re.findall(pattern, s)
    valid_substrings = [sub for sub in substrings if len(sub) == len(set(sub))]
    valid_substrings = "".join(valid_substrings)
    valid_substrings = ''.join(OrderedDict.fromkeys(valid_substrings))
    return valid_substrings


def score_lexeval_item(task_id, output_str, answer_str):
    """Score a single LexEval item. Returns float in [0,1] or NaN."""
    meta = LEXEVAL_TASKS.get(task_id)
    if meta is None:
        return np.nan

    if meta["type"] == "mc":
        pred = _lexeval_find_valid_substrings(output_str)
        return 1.0 if pred == answer_str else 0.0

    elif meta["type"] == "generation":
        return _lawbench_rouge_l(output_str, answer_str)

    return np.nan


# ══════════════════════════════════════════════════════════════════════
# Build LawBench response matrix
# ══════════════════════════════════════════════════════════════════════

def build_lawbench():
    """Build LawBench response matrix from prediction files."""
    pred_dir = LAWBENCH_DIR / "predictions" / "zero_shot"
    if not pred_dir.exists():
        print(f"  [SKIP] LawBench predictions not found at {pred_dir}")
        return None, None

    # Discover models
    model_dirs = sorted([
        d for d in pred_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])
    print(f"  Found {len(model_dirs)} models")

    # Discover task files from first model to get item count
    ref_model = None
    for md in model_dirs:
        task_files = sorted(md.glob("*.json"))
        if len(task_files) >= 15:
            ref_model = md
            break
    if ref_model is None:
        ref_model = model_dirs[0]

    # Build item IDs: lawbench_{task_id}_item{idx}
    task_items = {}
    for tf in sorted(ref_model.glob("*.json")):
        task_id = tf.stem  # e.g., "1-2"
        with open(tf) as f:
            data = json.load(f)
        n_items = len(data)
        task_items[task_id] = n_items
        print(f"    Task {task_id}: {n_items} items")

    # Create item_id list
    all_item_ids = []
    for task_id in sorted(task_items.keys()):
        for idx in range(task_items[task_id]):
            all_item_ids.append(f"lawbench_{task_id}_item{idx}")

    print(f"  Total items: {len(all_item_ids)}")

    # Build response matrix
    records = {}  # model_name -> {item_id: score}

    for model_dir in model_dirs:
        model_name = model_dir.name
        print(f"  Processing {model_name}...", end=" ", flush=True)
        scores = {}
        task_files = sorted(model_dir.glob("*.json"))

        for tf in task_files:
            task_id = tf.stem
            if task_id not in LAWBENCH_TASKS:
                continue
            try:
                with open(tf) as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[ERROR reading {tf}: {e}]", end=" ")
                continue

            for idx_str in sorted(data.keys(), key=lambda x: int(x)):
                idx = int(idx_str)
                entry = data[idx_str]
                prediction = entry.get("prediction", "")
                reference = entry.get("refr", "")
                item_id = f"lawbench_{task_id}_item{idx}"
                score = score_lawbench_item(task_id, prediction, reference)
                scores[item_id] = score

        records[model_name] = scores
        n_scored = sum(1 for v in scores.values() if not np.isnan(v))
        print(f"{n_scored}/{len(scores)} scored")

    # Create DataFrame
    df = pd.DataFrame(records).T  # rows=models, cols=items
    df.index.name = "model"
    # Reorder columns
    valid_cols = [c for c in all_item_ids if c in df.columns]
    df = df[valid_cols]
    print(f"  LawBench matrix: {df.shape[0]} models x {df.shape[1]} items")

    # Build task metadata
    meta_rows = []
    for item_id in valid_cols:
        parts = item_id.split("_")
        task_id = parts[1]  # e.g., "1-2"
        task_meta = LAWBENCH_TASKS.get(task_id, {})
        meta_rows.append({
            "item_id": item_id,
            "task_id": task_id,
            "task_name": task_meta.get("name", ""),
            "benchmark": "LawBench",
            "language": "zh",
            "category": task_meta.get("category", ""),
            "metric": task_meta.get("metric", ""),
            "task_type": task_meta.get("type", ""),
        })
    meta_df = pd.DataFrame(meta_rows)

    return df, meta_df


# ══════════════════════════════════════════════════════════════════════
# Build LexEval response matrix
# ══════════════════════════════════════════════════════════════════════

def build_lexeval():
    """Build LexEval response matrix from prediction files."""
    pred_dir = LEXEVAL_DIR / "model_output" / "zero_shot"
    if not pred_dir.exists():
        print(f"  [SKIP] LexEval predictions not found at {pred_dir}")
        return None, None

    # Discover models
    model_dirs = sorted([
        d for d in pred_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])
    print(f"  Found {len(model_dirs)} models")

    # Discover task files and item counts from reference model (gpt4)
    ref_model = pred_dir / "gpt4"
    if not ref_model.exists():
        ref_model = model_dirs[0]

    task_items = {}
    for tf in sorted(ref_model.glob("*.jsonl")):
        # filename: gpt4_1_1.jsonl -> task_id = "1_1"
        parts = tf.stem.split("_")
        task_id = f"{parts[-2]}_{parts[-1]}"
        n_items = sum(1 for _ in open(tf))
        task_items[task_id] = n_items
        print(f"    Task {task_id}: {n_items} items")

    # Create item_id list
    all_item_ids = []
    for task_id in sorted(task_items.keys()):
        for idx in range(task_items[task_id]):
            all_item_ids.append(f"lexeval_{task_id}_item{idx}")

    print(f"  Total items: {len(all_item_ids)}")

    # Build response matrix
    records = {}

    for model_dir in model_dirs:
        model_name = model_dir.name
        print(f"  Processing {model_name}...", end=" ", flush=True)
        scores = {}
        task_files = sorted(model_dir.glob("*.jsonl"))

        for tf in task_files:
            parts = tf.stem.split("_")
            task_id = f"{parts[-2]}_{parts[-1]}"
            if task_id not in LEXEVAL_TASKS:
                continue

            try:
                with open(tf) as f:
                    for idx, line in enumerate(f):
                        entry = json.loads(line.strip())
                        item_id = f"lexeval_{task_id}_item{idx}"
                        output_str = entry.get("output", "")
                        answer_str = entry.get("answer", "")
                        score = score_lexeval_item(task_id, output_str, answer_str)
                        scores[item_id] = score
            except Exception as e:
                print(f"[ERROR reading {tf}: {e}]", end=" ")
                continue

        records[model_name] = scores
        n_scored = sum(1 for v in scores.values() if not np.isnan(v))
        print(f"{n_scored}/{len(scores)} scored")

    # Create DataFrame
    df = pd.DataFrame(records).T
    df.index.name = "model"
    valid_cols = [c for c in all_item_ids if c in df.columns]
    df = df[valid_cols]
    print(f"  LexEval matrix: {df.shape[0]} models x {df.shape[1]} items")

    # Build task metadata
    meta_rows = []
    for item_id in valid_cols:
        parts = item_id.replace("lexeval_", "").split("_item")
        task_id = parts[0]
        task_meta = LEXEVAL_TASKS.get(task_id, {})
        meta_rows.append({
            "item_id": item_id,
            "task_id": task_id,
            "task_name": task_meta.get("name", ""),
            "benchmark": "LexEval",
            "language": "zh",
            "category": task_meta.get("category", ""),
            "metric": task_meta.get("metric", ""),
            "task_type": task_meta.get("type", ""),
        })
    meta_df = pd.DataFrame(meta_rows)

    return df, meta_df


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Building Legal Benchmark Response Matrices")
    print("=" * 70)

    all_matrices = []
    all_metadata = []
    all_model_summaries = []

    # ── LawBench ─────────────────────────────────────────────────────
    print("\n[1/2] LawBench (Chinese Legal, EMNLP 2024)")
    print("-" * 50)
    lb_matrix, lb_meta = build_lawbench()
    if lb_matrix is not None:
        lb_matrix.to_csv(PROCESSED_DIR / "lawbench_response_matrix.csv")
        print(f"  Saved: lawbench_response_matrix.csv")
        all_matrices.append(lb_matrix)
        all_metadata.append(lb_meta)

        # Model summary
        for model in lb_matrix.index:
            row = lb_matrix.loc[model]
            valid = row.dropna()
            all_model_summaries.append({
                "model": model,
                "benchmark": "LawBench",
                "num_items_scored": len(valid),
                "num_items_total": len(row),
                "mean_score": valid.mean() if len(valid) > 0 else np.nan,
            })

    # ── LexEval ──────────────────────────────────────────────────────
    print("\n[2/2] LexEval (Chinese Legal, NeurIPS 2024)")
    print("-" * 50)
    le_matrix, le_meta = build_lexeval()
    if le_matrix is not None:
        le_matrix.to_csv(PROCESSED_DIR / "lexeval_response_matrix.csv")
        print(f"  Saved: lexeval_response_matrix.csv")
        all_matrices.append(le_matrix)
        all_metadata.append(le_meta)

        for model in le_matrix.index:
            row = le_matrix.loc[model]
            valid = row.dropna()
            all_model_summaries.append({
                "model": model,
                "benchmark": "LexEval",
                "num_items_scored": len(valid),
                "num_items_total": len(row),
                "mean_score": valid.mean() if len(valid) > 0 else np.nan,
            })

    # ── Combined ─────────────────────────────────────────────────────
    if all_matrices:
        print("\n" + "=" * 70)
        print("Combining matrices...")

        # Combine: align by model name where possible, but keep separate rows
        combined = pd.concat(all_matrices, axis=1)
        combined.index.name = "model"
        combined.to_csv(PROCESSED_DIR / "response_matrix.csv")
        print(f"  Combined matrix: {combined.shape[0]} models x {combined.shape[1]} items")
        print(f"  Saved: response_matrix.csv")

        # Task metadata
        meta_combined = pd.concat(all_metadata, ignore_index=True)
        meta_combined.to_csv(PROCESSED_DIR / "task_metadata.csv", index=False)
        print(f"  Saved: task_metadata.csv ({len(meta_combined)} items)")

        # Model summary
        summary_df = pd.DataFrame(all_model_summaries)
        summary_df.to_csv(PROCESSED_DIR / "model_summary.csv", index=False)
        print(f"  Saved: model_summary.csv ({len(summary_df)} entries)")

        # Print statistics
        print("\n" + "=" * 70)
        print("Summary Statistics:")
        print(f"  Total unique models: {combined.shape[0]}")
        print(f"  Total items: {combined.shape[1]}")
        non_nan = combined.notna().sum().sum()
        total_cells = combined.shape[0] * combined.shape[1]
        print(f"  Fill rate: {non_nan}/{total_cells} ({100*non_nan/total_cells:.1f}%)")

        for bench in ["LawBench", "LexEval"]:
            bench_items = [c for c in combined.columns if c.startswith(bench.lower().replace("bench", "bench"))]
            if not bench_items:
                bench_prefix = bench.lower().replace("bench", "bench")
                bench_items = [c for c in combined.columns
                               if c.startswith("lawbench" if bench == "LawBench" else "lexeval")]
            print(f"\n  {bench}:")
            print(f"    Items: {len(bench_items)}")
            sub = combined[bench_items]
            n_models_with_data = (sub.notna().sum(axis=1) > 0).sum()
            print(f"    Models with data: {n_models_with_data}")
            if len(bench_items) > 0 and n_models_with_data > 0:
                mean_scores = sub.mean(axis=1).dropna()
                if len(mean_scores) > 0:
                    print(f"    Mean score (across models): {mean_scores.mean():.4f}")
                    print(f"    Min model mean: {mean_scores.min():.4f}")
                    print(f"    Max model mean: {mean_scores.max():.4f}")


if __name__ == "__main__":
    main()

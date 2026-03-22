#!/usr/bin/env python3
"""
Process medical benchmark data for validity analysis.

This script processes per-item evaluation data from multiple medical benchmarks:

BENCHMARKS WITH PER-ITEM MODEL PREDICTIONS:
1. IgakuQA (Japanese) - 5 years of exam data with GPT-3/4, ChatGPT, student predictions
2. IgakuQA119 (Japanese) - 119th exam with per-item grading results

BENCHMARKS WITH QUESTION DATA (for running evaluations / future per-item collection):
3. MedQA Chinese (Mainland) - 3,426 test items from Chinese medical licensing exam
4. CMB (Chinese Medical Benchmark) - 11,200 test items across 28 medical subcategories
5. CMExam (Chinese Medical Exam) - 6,811 test items with rich annotations
6. MMedBench (Multilingual) - Test items in Chinese(3426), Japanese(199), Spanish(2742), French(622), Russian(256), English(1273)
7. KorMedMCQA (Korean) - Doctor(285), Nurse(587), Pharmacist(614) test items
8. MedExpQA (Spanish) - Multilingual medical QA with explanations
9. MedArabiQ (Arabic) - Multiple choice questions and fill-in-blank
10. MMLU-Pro Health subset - 826 health-related items with per-model predictions

Output format matches AfriMedQA structure:
- response_matrix.csv: rows=items, cols=models, values=0/1
- task_metadata.csv: item metadata (question, answer, specialty, etc.)
- model_summary.csv: per-model aggregate accuracy
"""

import json
import csv
import os
import glob
import sys
from collections import defaultdict
from pathlib import Path

BASE_OUT = "/lfs/skampere1/0/sttruong/torch_measure/data/medeval_data/processed"
RAW_DIR = "/lfs/skampere1/0/sttruong/torch_measure/data/medeval_data/raw"


def process_igakuqa():
    """Process IgakuQA (2018-2022) - Japanese Medical Licensing Exam.

    Has per-item predictions for: gpt3, gpt4, chatgpt, translate_chatgpt-en, student-majority
    """
    base = "/tmp/IgakuQA"
    if not os.path.exists(base):
        print("IgakuQA not found at /tmp/IgakuQA, skipping")
        return

    out_dir = os.path.join(BASE_OUT, "igakuqa")
    os.makedirs(out_dir, exist_ok=True)

    # Collect all questions and their metadata
    questions = {}  # problem_id -> metadata
    predictions = defaultdict(dict)  # problem_id -> {model: prediction}

    models = ["gpt3", "gpt4", "chatgpt", "translate_chatgpt-en", "student-majority"]

    for year in range(2018, 2023):
        year_str = str(year)
        data_dir = os.path.join(base, "data", year_str)
        results_dir = os.path.join(base, "baseline_results", year_str)

        if not os.path.exists(data_dir):
            continue

        # Load questions
        for qfile in sorted(glob.glob(os.path.join(data_dir, "*.jsonl"))):
            fname = os.path.basename(qfile)
            if "_metadata" in fname or "_translate" in fname:
                continue

            with open(qfile) as f:
                for line in f:
                    item = json.loads(line)
                    pid = item["problem_id"]
                    questions[pid] = {
                        "problem_id": pid,
                        "year": year,
                        "question": item.get("problem_text", ""),
                        "choices": json.dumps(item.get("choices", []), ensure_ascii=False),
                        "answer": ",".join(item.get("answer", [])),
                        "points": item.get("points", "1"),
                        "text_only": item.get("text_only", True),
                    }

        # Load predictions
        for model in models:
            for rfile in sorted(glob.glob(os.path.join(results_dir, f"*_{model}.jsonl"))):
                with open(rfile) as f:
                    for line in f:
                        item = json.loads(line)
                        pid = item["problem_id"]
                        pred = item.get("prediction", "")
                        if pid in questions:
                            correct_answer = questions[pid]["answer"]
                            is_correct = 1 if pred == correct_answer else 0
                            predictions[pid][model] = is_correct

    if not questions:
        print("No IgakuQA questions found")
        return

    # Write task_metadata.csv
    pids = sorted(questions.keys())
    with open(os.path.join(out_dir, "task_metadata.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "question", "answer", "year", "points", "text_only", "language", "benchmark"])
        for pid in pids:
            q = questions[pid]
            writer.writerow([pid, q["question"], q["answer"], q["year"], q["points"], q["text_only"], "Japanese", "IgakuQA"])

    # Write response_matrix.csv
    with open(os.path.join(out_dir, "response_matrix.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id"] + models)
        for pid in pids:
            row = [pid]
            for model in models:
                row.append(predictions[pid].get(model, ""))
            writer.writerow(row)

    # Write model_summary.csv
    with open(os.path.join(out_dir, "model_summary.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "num_items", "num_correct", "accuracy"])
        for model in models:
            total = sum(1 for pid in pids if model in predictions[pid])
            correct = sum(predictions[pid].get(model, 0) for pid in pids if model in predictions[pid])
            acc = correct / total if total > 0 else 0
            writer.writerow([model, total, correct, f"{acc:.4f}"])

    print(f"IgakuQA: {len(pids)} questions, {len(models)} models -> {out_dir}")


def process_igakuqa119():
    """Process IgakuQA119 - 119th Japanese Medical Licensing Exam.

    Has per-item grading results for demo models.
    """
    base = "/tmp/IgakuQA119"
    if not os.path.exists(base):
        print("IgakuQA119 not found, skipping")
        return

    out_dir = os.path.join(BASE_OUT, "igakuqa119")
    os.makedirs(out_dir, exist_ok=True)

    # Load correct answers
    correct = {}
    with open(os.path.join(base, "results", "correct_answers.csv"), encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qnum = row.get("\u554f\u984c\u756a\u53f7", row.get("問題番号", ""))
            ans = row.get("\u89e3\u7b54", row.get("解答", ""))
            if qnum and ans:
                correct[qnum] = ans

    # Load questions
    questions = {}
    for qfile in glob.glob(os.path.join(base, "questions", "*.json")):
        with open(qfile) as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    qnum = item.get("question_number", "")
                    questions[qnum] = {
                        "question": item.get("question_text", ""),
                        "choices": json.dumps(item.get("choices", []), ensure_ascii=False),
                        "has_image": item.get("has_image", False),
                    }

    # Load grading results
    predictions = defaultdict(dict)
    models = set()

    for gfile in glob.glob(os.path.join(base, "results", "demo", "*_grading_results.csv")):
        with open(gfile, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                qnum = row.get("question_number", "")
                model = row.get("model", "").split("/")[-1].replace(":latest", "")
                is_correct = 1 if row.get("is_correct", "").lower() == "true" else 0
                if qnum and model:
                    predictions[qnum][model] = is_correct
                    models.add(model)

    if not correct:
        print("No IgakuQA119 correct answers found")
        return

    models = sorted(models)
    qnums = sorted(correct.keys())

    # Write files
    with open(os.path.join(out_dir, "task_metadata.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "question", "answer", "has_image", "language", "benchmark"])
        for qnum in qnums:
            q = questions.get(qnum, {})
            writer.writerow([qnum, q.get("question", ""), correct[qnum], q.get("has_image", ""), "Japanese", "IgakuQA119"])

    with open(os.path.join(out_dir, "response_matrix.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id"] + models)
        for qnum in qnums:
            row = [qnum]
            for model in models:
                row.append(predictions[qnum].get(model, ""))
            writer.writerow(row)

    with open(os.path.join(out_dir, "model_summary.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "num_items", "num_correct", "accuracy"])
        for model in models:
            total = sum(1 for q in qnums if model in predictions[q])
            corr = sum(predictions[q].get(model, 0) for q in qnums if model in predictions[q])
            acc = corr / total if total > 0 else 0
            writer.writerow([model, total, corr, f"{acc:.4f}"])

    print(f"IgakuQA119: {len(qnums)} questions, {len(models)} models -> {out_dir}")


def process_mmlupro_health():
    """Extract MMLU-Pro health subset with per-item model predictions."""
    mmlu_dir = "/lfs/skampere1/0/sttruong/torch_measure/data/mmlupro_data/processed"
    if not os.path.exists(mmlu_dir):
        print("MMLU-Pro data not found, skipping")
        return

    out_dir = os.path.join(BASE_OUT, "mmlupro_health")
    os.makedirs(out_dir, exist_ok=True)

    # Load question metadata to find health items
    health_ids = set()
    metadata = {}
    with open(os.path.join(mmlu_dir, "question_metadata.csv")) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["category"] == "health":
                qid = row["question_id"]
                health_ids.add(qid)
                metadata[qid] = row

    # Load response matrix and filter health items
    with open(os.path.join(mmlu_dir, "response_matrix.csv")) as f:
        reader = csv.DictReader(f)
        models = [col for col in reader.fieldnames if col != "question_id"]

        rows = []
        for row in reader:
            if row["question_id"] in health_ids:
                rows.append(row)

    # Write filtered data
    with open(os.path.join(out_dir, "task_metadata.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "category", "answer", "answer_index", "source", "language", "benchmark"])
        for qid in sorted(health_ids):
            m = metadata[qid]
            writer.writerow([qid, m["category"], m["answer"], m["answer_index"], m.get("src", ""), "English", "MMLU-Pro-Health"])

    with open(os.path.join(out_dir, "response_matrix.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id"] + models)
        for row in rows:
            writer.writerow([row["question_id"]] + [row.get(m, "") for m in models])

    # Model summary
    with open(os.path.join(out_dir, "model_summary.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "num_items", "num_correct", "accuracy"])
        for model in models:
            vals = [row.get(model, "") for row in rows]
            valid = [v for v in vals if v != ""]
            correct = sum(1 for v in valid if float(v) == 1.0)
            total = len(valid)
            acc = correct / total if total > 0 else 0
            writer.writerow([model, total, correct, f"{acc:.4f}"])

    print(f"MMLU-Pro Health: {len(health_ids)} items, {len(models)} models -> {out_dir}")


def process_medqa_chinese():
    """Process MedQA Chinese (Mainland) test set questions."""
    data_dir = "/tmp/medqa_extracted/data_clean/questions/Mainland/4_options"
    if not os.path.exists(data_dir):
        data_dir = "/tmp/medqa_extracted/data_clean/questions/Mainland"
    if not os.path.exists(data_dir):
        print("MedQA Chinese not found, skipping")
        return

    out_dir = os.path.join(BASE_OUT, "medqa_chinese")
    os.makedirs(out_dir, exist_ok=True)

    test_file = os.path.join(data_dir, "test.jsonl")
    items = []
    with open(test_file) as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            items.append({
                "sample_id": f"medqa_zh_{i:05d}",
                "question": item.get("question", ""),
                "options": json.dumps(item.get("options", {}), ensure_ascii=False),
                "answer": item.get("answer", ""),
                "answer_idx": item.get("answer_idx", ""),
                "meta_info": item.get("meta_info", ""),
            })

    with open(os.path.join(out_dir, "task_metadata.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "question", "answer", "answer_idx", "meta_info", "language", "benchmark"])
        for item in items:
            writer.writerow([item["sample_id"], item["question"], item["answer"],
                           item["answer_idx"], item["meta_info"], "Chinese", "MedQA-Chinese"])

    print(f"MedQA Chinese: {len(items)} test items (questions only, no per-item model predictions) -> {out_dir}")


def process_cmb():
    """Process CMB (Chinese Medical Benchmark) test questions."""
    data_file = "/tmp/CMB_extracted/CMB/CMB-Exam/CMB-test/CMB-test-choice-question-merge.json"
    answer_file = "/tmp/CMB/data/CMB-test-choice-answer.json"

    if not os.path.exists(data_file):
        print("CMB data not found, skipping")
        return

    out_dir = os.path.join(BASE_OUT, "cmb")
    os.makedirs(out_dir, exist_ok=True)

    with open(data_file) as f:
        questions = json.load(f)

    answers = {}
    if os.path.exists(answer_file):
        with open(answer_file) as f:
            ans_list = json.load(f)
            for a in ans_list:
                answers[a["id"]] = a["answer"]

    with open(os.path.join(out_dir, "task_metadata.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "question", "answer", "exam_type", "exam_class", "exam_subject", "question_type", "language", "benchmark"])
        for q in questions:
            writer.writerow([
                f"cmb_{q['id']:05d}",
                q.get("question", ""),
                answers.get(q["id"], ""),
                q.get("exam_type", ""),
                q.get("exam_class", ""),
                q.get("exam_subject", ""),
                q.get("question_type", ""),
                "Chinese",
                "CMB"
            ])

    print(f"CMB: {len(questions)} test items (questions+answers, no per-item model predictions) -> {out_dir}")


def process_cmexam():
    """Process CMExam - Chinese National Medical Licensing Examination."""
    test_file = "/tmp/CMExam_hf/test.json"
    if not os.path.exists(test_file):
        print("CMExam data not found, skipping")
        return

    out_dir = os.path.join(BASE_OUT, "cmexam")
    os.makedirs(out_dir, exist_ok=True)

    items = []
    with open(test_file) as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            items.append({
                "sample_id": f"cmexam_{i:05d}",
                "question": item.get("Question", ""),
                "answer": item.get("Answer", ""),
                "explanation": item.get("Explanation", "")[:200] if item.get("Explanation") else "",
            })

    with open(os.path.join(out_dir, "task_metadata.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "question", "answer", "explanation_preview", "language", "benchmark"])
        for item in items:
            writer.writerow([item["sample_id"], item["question"], item["answer"],
                           item["explanation"], "Chinese", "CMExam"])

    print(f"CMExam: {len(items)} test items (questions+answers, no per-item model predictions) -> {out_dir}")


def process_mmedbench():
    """Process MMedBench - Multilingual Medical Benchmark (6 languages)."""
    test_dir = "/tmp/MMedBench_extracted/MMedBench/Test"
    if not os.path.exists(test_dir):
        print("MMedBench not found, skipping")
        return

    out_dir = os.path.join(BASE_OUT, "mmedbench")
    os.makedirs(out_dir, exist_ok=True)

    all_items = []
    for lang_file in sorted(glob.glob(os.path.join(test_dir, "*.jsonl"))):
        lang = os.path.basename(lang_file).replace(".jsonl", "")
        with open(lang_file) as f:
            for i, line in enumerate(f):
                item = json.loads(line)
                all_items.append({
                    "sample_id": f"mmedbench_{lang.lower()}_{i:05d}",
                    "question": item.get("question", ""),
                    "options": json.dumps(item.get("options", {}), ensure_ascii=False),
                    "answer": item.get("answer", ""),
                    "answer_idx": item.get("answer_idx", ""),
                    "meta_info": item.get("meta_info", ""),
                    "language": lang,
                })

    with open(os.path.join(out_dir, "task_metadata.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "question", "answer", "answer_idx", "meta_info", "language", "benchmark"])
        for item in all_items:
            writer.writerow([item["sample_id"], item["question"], item["answer"],
                           item["answer_idx"], item["meta_info"], item["language"], "MMedBench"])

    # Language breakdown
    lang_counts = defaultdict(int)
    for item in all_items:
        lang_counts[item["language"]] += 1

    print(f"MMedBench: {len(all_items)} test items across {len(lang_counts)} languages:")
    for lang, count in sorted(lang_counts.items()):
        print(f"  {lang}: {count}")
    print(f"  (questions+answers, no per-item model predictions) -> {out_dir}")


def process_kormedmcqa():
    """Process KorMedMCQA - Korean Healthcare Professional Licensing Exams."""
    data_dir = "/tmp/KorMedMCQA/data"
    if not os.path.exists(data_dir):
        print("KorMedMCQA not found, skipping")
        return

    out_dir = os.path.join(BASE_OUT, "kormedmcqa")
    os.makedirs(out_dir, exist_ok=True)

    all_items = []
    for test_file in sorted(glob.glob(os.path.join(data_dir, "*-test.csv"))):
        subject = os.path.basename(test_file).replace("-test.csv", "")
        with open(test_file) as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                all_items.append({
                    "sample_id": f"kormed_{subject}_{i:05d}",
                    "subject": row.get("subject", subject),
                    "year": row.get("year", ""),
                    "question": row.get("question", ""),
                    "A": row.get("A", ""),
                    "B": row.get("B", ""),
                    "C": row.get("C", ""),
                    "D": row.get("D", ""),
                    "E": row.get("E", ""),
                    "answer": row.get("answer", ""),
                })

    with open(os.path.join(out_dir, "task_metadata.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "question", "answer", "subject", "year", "language", "benchmark"])
        for item in all_items:
            writer.writerow([item["sample_id"], item["question"], item["answer"],
                           item["subject"], item["year"], "Korean", "KorMedMCQA"])

    print(f"KorMedMCQA: {len(all_items)} test items (questions+answers, no per-item model predictions) -> {out_dir}")


def process_medexpqa():
    """Process MedExpQA - Spanish Medical QA with explanations."""
    data_dir = "/tmp/MedExpQA/data/es"
    if not os.path.exists(data_dir):
        print("MedExpQA not found, skipping")
        return

    out_dir = os.path.join(BASE_OUT, "medexpqa")
    os.makedirs(out_dir, exist_ok=True)

    test_file = os.path.join(data_dir, "test.es.casimedicos.rag.jsonl")
    items = []
    with open(test_file) as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            items.append({
                "sample_id": f"medexpqa_es_{i:05d}",
                "correct_option": item.get("correct_option", ""),
                "full_question": item.get("full_question", ""),
                "question": item.get("question", ""),
            })

    with open(os.path.join(out_dir, "task_metadata.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "question", "correct_option", "language", "benchmark"])
        for item in items:
            q = item["full_question"] or item["question"]
            writer.writerow([item["sample_id"], q, item["correct_option"], "Spanish", "MedExpQA"])

    # Also process other languages
    for lang_code, lang_name in [("en", "English"), ("fr", "French"), ("it", "Italian")]:
        lang_dir = f"/tmp/MedExpQA/data/{lang_code}"
        if os.path.exists(lang_dir):
            test_files = glob.glob(os.path.join(lang_dir, f"test.{lang_code}.*"))
            if test_files:
                count = sum(1 for _ in open(test_files[0]))
                print(f"  MedExpQA {lang_name}: {count} test items also available")

    print(f"MedExpQA Spanish: {len(items)} test items (questions+answers, no per-item model predictions) -> {out_dir}")


def process_medarabiq():
    """Process MedArabiQ - Arabic Medical Tasks."""
    data_dir = "/tmp/MedArabiQ/datasets"
    if not os.path.exists(data_dir):
        print("MedArabiQ not found, skipping")
        return

    out_dir = os.path.join(BASE_OUT, "medarabiq")
    os.makedirs(out_dir, exist_ok=True)

    all_items = []
    for csvfile in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
        task = os.path.basename(csvfile).replace(".csv", "")
        try:
            with open(csvfile, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    all_items.append({
                        "sample_id": f"medarabiq_{task}_{i:05d}",
                        "task": task,
                        "question": row.get("Question", ""),
                        "answer": row.get("Answer", ""),
                        "category": row.get("Category", ""),
                    })
        except Exception as e:
            print(f"  Warning: Error reading {csvfile}: {e}")

    with open(os.path.join(out_dir, "task_metadata.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "question", "answer", "task", "category", "language", "benchmark"])
        for item in all_items:
            writer.writerow([item["sample_id"], item["question"], item["answer"],
                           item["task"], item["category"], "Arabic", "MedArabiQ"])

    # Task breakdown
    task_counts = defaultdict(int)
    for item in all_items:
        task_counts[item["task"]] += 1

    print(f"MedArabiQ: {len(all_items)} items across {len(task_counts)} tasks:")
    for task, count in sorted(task_counts.items()):
        print(f"  {task}: {count}")
    print(f"  (questions+answers, no per-item model predictions) -> {out_dir}")


def process_frenchmedmcqa():
    """Process FrenchMedMCQA - French Medical Specialization Exam MCQ."""
    test_file = "/tmp/FrenchMedMCQA_extracted/test.json"
    if not os.path.exists(test_file):
        print("FrenchMedMCQA not found, skipping")
        return

    out_dir = os.path.join(BASE_OUT, "frenchmedmcqa")
    os.makedirs(out_dir, exist_ok=True)

    with open(test_file) as f:
        data = json.load(f)

    with open(os.path.join(out_dir, "task_metadata.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "question", "correct_answers", "nbr_correct_answers", "subject", "language", "benchmark"])
        for item in data:
            writer.writerow([
                item.get("id", ""),
                item.get("question", ""),
                ",".join(item.get("correct_answers", [])),
                item.get("nbr_correct_answers", ""),
                item.get("subject_name", ""),
                "French",
                "FrenchMedMCQA"
            ])

    print(f"FrenchMedMCQA: {len(data)} test items (questions+answers, no per-item model predictions) -> {out_dir}")


def process_permedcqa():
    """Process PerMedCQA - Persian Medical Consumer QA."""
    test_file = "/tmp/PerMedCQA/Data/test.json"
    if not os.path.exists(test_file):
        print("PerMedCQA not found, skipping")
        return

    out_dir = os.path.join(BASE_OUT, "permedcqa")
    os.makedirs(out_dir, exist_ok=True)

    with open(test_file) as f:
        data = json.load(f)

    with open(os.path.join(out_dir, "task_metadata.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "title", "category", "specialty", "sex", "age", "source", "language", "benchmark"])
        for item in data:
            writer.writerow([
                item.get("instance_id", ""),
                item.get("Title", ""),
                item.get("Category", ""),
                item.get("Specialty", ""),
                item.get("Sex", ""),
                item.get("Age", ""),
                item.get("dataset_source", ""),
                "Persian",
                "PerMedCQA"
            ])

    print(f"PerMedCQA: {len(data)} test items (free-text QA, no per-item model predictions) -> {out_dir}")


def write_summary():
    """Write a comprehensive summary of all processed benchmarks."""
    summary_file = os.path.join(BASE_OUT, "SUMMARY.txt")

    lines = [
        "=" * 80,
        "MEDICAL EVALUATION DATA SUMMARY",
        "=" * 80,
        "",
        "This directory contains medical benchmark data organized for validity analysis.",
        "Data was collected from multiple sources, covering non-English and Global South",
        "medical benchmarks.",
        "",
        "STATUS KEY:",
        "  [PER-ITEM] = Has per-item model predictions (response matrix available)",
        "  [QUESTIONS] = Has question data + answer keys (no per-item predictions yet)",
        "",
        "-" * 80,
        "",
    ]

    benchmarks = [
        ("igakuqa", "[PER-ITEM]", "Japanese", "IgakuQA (2018-2022)",
         "Japanese Medical Licensing Exam with GPT-3/4, ChatGPT, student-majority predictions"),
        ("igakuqa119", "[PER-ITEM]", "Japanese", "IgakuQA 119th Exam",
         "119th Japanese Medical Licensing Exam with pfn-medllm-qwen-72b, qwen2.5-72b, DeepSeek-R1 predictions"),
        ("mmlupro_health", "[PER-ITEM]", "English", "MMLU-Pro Health Subset",
         "826 health items from MMLU-Pro with 50+ model predictions"),
        ("medqa_chinese", "[QUESTIONS]", "Chinese", "MedQA Chinese (Mainland)",
         "3,426 test items from Chinese medical licensing exam (4-option MCQ)"),
        ("cmb", "[QUESTIONS]", "Chinese", "CMB (Chinese Medical Benchmark)",
         "11,200 test items across 28 medical subcategories"),
        ("cmexam", "[QUESTIONS]", "Chinese", "CMExam",
         "6,811 test items from Chinese National Medical Licensing Examination"),
        ("mmedbench", "[QUESTIONS]", "6 languages", "MMedBench (Multilingual)",
         "8,518 test items in Chinese(3426), Spanish(2742), English(1273), French(622), Russian(256), Japanese(199)"),
        ("kormedmcqa", "[QUESTIONS]", "Korean", "KorMedMCQA",
         "~1,488 test items: Doctor(285), Nurse(587), Pharmacist(614)"),
        ("medexpqa", "[QUESTIONS]", "Spanish+3", "MedExpQA",
         "Spanish medical QA from CasiMedicos with explanations (also EN, FR, IT)"),
        ("medarabiq", "[QUESTIONS]", "Arabic", "MedArabiQ",
         "Arabic medical tasks: MCQ, fill-in-blank, patient-doctor QA"),
        ("frenchmedmcqa", "[QUESTIONS]", "French", "FrenchMedMCQA",
         "622 test items from French medical specialization diploma in pharmacy"),
        ("permedcqa", "[QUESTIONS]", "Persian", "PerMedCQA",
         "3,512 Persian consumer medical QA (free-text, not MCQ)"),
    ]

    for dirname, status, lang, name, desc in benchmarks:
        dirpath = os.path.join(BASE_OUT, dirname)
        if os.path.exists(dirpath):
            files = os.listdir(dirpath)
            lines.append(f"{status} {name}")
            lines.append(f"  Language: {lang}")
            lines.append(f"  Description: {desc}")
            lines.append(f"  Files: {', '.join(sorted(files))}")
            lines.append("")

    lines.extend([
        "-" * 80,
        "",
        "ADDITIONAL DATA SOURCES (raw data available for future processing):",
        "",
        "- HEAD-QA (Spanish): Healthcare exam QA at /tmp/head-qa/",
        "- PubMedQA: Biomedical research QA (English, yes/no/maybe format)",
        "- MedRBench: Clinical case reasoning with per-item inference results",
        "  (English only, diagnostic reasoning rather than MCQ)",
        "- BRIDGE: Multilingual clinical NLP benchmark (9 languages, 87 tasks)",
        "  Dataset: huggingface.co/datasets/YLab-Open/BRIDGE-Open",
        "- WorldMedQA-V: Multimodal medical QA (Brazil, Israel, Japan, Spain)",
        "  GitHub: github.com/WorldMedQA",
        "- Open Medical LLM Leaderboard: Aggregate scores only (no per-item)",
        "  huggingface.co/spaces/openlifescienceai/open_medical_llm_leaderboard",
        "- Med-Gemini MedQA Relabelling: Expert relabelling of MedQA (English)",
        "  github.com/Google-Health/med-gemini-medqa-relabelling",
        "",
        "NOTE: Most medical benchmarks publish only aggregate scores, not per-item",
        "model predictions. The IgakuQA family and MMLU-Pro Health are the main",
        "non-English sources with actual response matrices available.",
    ])

    with open(summary_file, "w") as f:
        f.write("\n".join(lines))

    print(f"\nSummary written to {summary_file}")


if __name__ == "__main__":
    print("Processing medical benchmark data...\n")

    process_igakuqa()
    print()
    process_igakuqa119()
    print()
    process_mmlupro_health()
    print()
    process_medqa_chinese()
    print()
    process_cmb()
    print()
    process_cmexam()
    print()
    process_mmedbench()
    print()
    process_kormedmcqa()
    print()
    process_medexpqa()
    print()
    process_medarabiq()
    print()
    process_frenchmedmcqa()
    print()
    process_permedcqa()
    print()

    write_summary()
    print("\nDone!")

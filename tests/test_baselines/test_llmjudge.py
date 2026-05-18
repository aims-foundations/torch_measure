# Copyright (c) 2026 AIMS Foundations. MIT License.

import argparse
import glob
import os
import random

import pandas as pd
import torch
from huggingface_hub import snapshot_download
from tqdm import tqdm

from torch_measure.models import LLMJudge


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an LLMJudge model.")
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2-7B-Instruct",
        help="HuggingFace model identifier",
    )
    parser.add_argument(
        "--max-icl",
        type=int,
        default=5,
        help="Maximum same-subject in-context examples to prepend",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for LLM inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device map passed to from_pretrained (e.g. 'auto', 'cuda', 'cpu')",
    )
    parser.add_argument(
        "--size",
        type=float,
        default=0.0001,
        help="Fraction of total data to sample (default: 0.0001)",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Fraction of sampled data to hold out as test (default: 0.2)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=35,
        help="Number of labeled examples for ICL / acquisition evaluation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Download data snapshot and load lookup tables
    snapshot_path = snapshot_download(
        repo_id="aims-foundations/measurement-db",
        repo_type="dataset",
    )
    subjects_df = pd.read_parquet(os.path.join(snapshot_path, "subjects.parquet"))
    items_df = pd.read_parquet(os.path.join(snapshot_path, "items.parquet"))

    # Load trial files (skip _traces, subjects, items, benchmarks)
    skip = {"subjects.parquet", "items.parquet", "benchmarks.parquet"}
    trial_files = [
        f
        for f in glob.glob(os.path.join(snapshot_path, "*.parquet"))
        if not os.path.basename(f).endswith("_traces.parquet") and os.path.basename(f) not in skip
    ]
    print(f"\nLoading {len(trial_files)} trial files")
    trials = pd.concat(
        [pd.read_parquet(f, columns=["subject_id", "item_id", "response"]) for f in trial_files],
        ignore_index=True,
    ).dropna(subset=["response"])
    # Keep only binary pass/fail labels
    trials = trials[trials["response"].isin([0.0, 1.0])]
    trials = trials.merge(subjects_df, on="subject_id", how="inner").merge(items_df, on="item_id", how="inner")
    print(f"Total samples: {len(trials)}")

    trials = trials.sample(frac=args.size).reset_index(drop=True)
    print(f"Using {len(trials)} samples ({args.size * 100:.4g}% of total)")
    n_test = max(1, int(len(trials) * args.test_split))
    test_trials = trials.iloc[:n_test]
    train_trials = trials.iloc[n_test:]
    print(f"Train: {len(train_trials)} | Test: {len(test_trials)}")

    model = LLMJudge(
        model_id=args.model_id,
        max_icl=args.max_icl,
        batch_size=args.batch_size,
        device=args.device,
    )

    candidates = [
        {
            "subject_content": row["display_name"],
            "item_content": row["content"],
            "benchmark": row.get("benchmark", ""),
            "condition": row.get("condition", ""),
            "label": float(row["response"]),
        }
        for _, row in test_trials.iterrows()
    ]

    top_k = None
    try:
        from labeling import select_top_k

        print(f"\nRunning acquisition on {len(candidates)} test candidates...")
        top_k = select_top_k(candidates, k=args.k)
        print("\nTop 5 acquisition-selected entries:")
        for i, entry in enumerate(top_k[:5], 1):
            print(f"  {i}. subject='{entry['subject_content'][:60]}...'  item='{entry['item_content'][:60]}...'")
    except Exception as e:
        print(f"WARNING: top-k acquisition unavailable ({e}). Skipping top-k evaluation.")

    rand_k = random.sample(candidates, k=min(args.k, len(candidates)))

    def _eval_nll(labeled_samples=None, held_out_df=None):
        labeled = (
            [
                {k: e[k] for k in ("subject_content", "item_content", "benchmark", "condition", "label")}
                for e in labeled_samples
            ]
            if labeled_samples
            else None
        )

        inps, ys = [], []
        for _, row in held_out_df.iterrows():
            inps.append(
                {
                    "subject_content": row["display_name"],
                    "item_content": row["content"],
                    "benchmark": row.get("benchmark", ""),
                    "condition": row.get("condition", ""),
                }
            )
            ys.append(float(row["response"]))

        prompts = [model._build_prompt(inp, labeled) for inp in inps]

        # Sort by token length so each batch has minimal padding waste
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        sorted_prompts = [prompts[i] for i in order]

        preds_sorted = []
        with tqdm(total=len(sorted_prompts), desc="  evaluating") as pbar:
            for i in range(0, len(sorted_prompts), args.batch_size):
                batch = sorted_prompts[i : i + args.batch_size]
                preds_sorted.extend(model._batch_probs(batch))
                pbar.update(len(batch))

        preds = [None] * len(prompts)
        for orig_i, pred in zip(order, preds_sorted, strict=False):
            preds[orig_i] = pred

        preds_t = torch.tensor(preds).clamp(1e-7, 1 - 1e-7)
        ys_t = torch.tensor(ys)
        return torch.nn.functional.binary_cross_entropy(preds_t, ys_t).item()

    def _exclude(df, samples):
        keys = {(e["subject_content"], e["item_content"]) for e in samples}
        mask = df.apply(lambda r: (r["display_name"], r["content"]) in keys, axis=1)
        return df[~mask]

    if top_k is not None:
        topk_eval_df = _exclude(test_trials, top_k)
        print("\nEvaluating top-k acquisition strategy...")
        topk_nll = _eval_nll(top_k, topk_eval_df)
    else:
        topk_nll = None

    rand_eval_df = _exclude(test_trials, rand_k)
    print("\nEvaluating random selection strategy...")
    rand_nll = _eval_nll(rand_k, rand_eval_df)

    print("Evaluating no selection strategy...")
    no_label_nll = _eval_nll(None, test_trials)

    print(f"\nRandom selection  test NLL (BCE): {rand_nll:.4f}  (n={len(rand_eval_df)})")
    if topk_nll is not None:
        print(f"Top-k acquisition test NLL (BCE): {topk_nll:.4f}  (n={len(topk_eval_df)})")
        print(f"Delta (topk - random):            {topk_nll - rand_nll:+.4f}")
    else:
        print("Top-k acquisition test NLL (BCE): skipped (acquisition unavailable)")
    print(f"No labeling       test NLL (BCE): {no_label_nll:.4f}  (n={len(test_trials)})")

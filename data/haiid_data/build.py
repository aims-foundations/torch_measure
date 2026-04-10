"""
01_build_response_matrix.py — Download and build intervention matrices from HAIID
(Human-AI Interactions Dataset).

Downloads from: https://github.com/kailas-v/human-ai-interactions
30,000+ interactions across 5 classification domains: art, census, cities, sarcasm, dermatology.
Paper: Vodrahalli et al., "Do Humans Trust Advice More if it Comes from AI?"

Structure: participants x items x {pre-advice, post-advice} x {ai_source, human_source} -> response

Columns:
  - response_1: pre-advice response (continuous [-1, 1], positive = correct label)
  - response_2: post-advice response (continuous [-1, 1], positive = correct label)
  - advice_source: "ai" or "human" (the treatment label shown to participant)
  - advice: the confidence score given as advice

Output:
  - intervention_table.csv: full long-format data
  - Per-domain paired matrices:
    - response_matrix_{domain}_pre.csv: participant x item -> binary correct (before advice)
    - response_matrix_{domain}_post_ai.csv: participant x item -> binary correct (after AI-labeled advice)
    - response_matrix_{domain}_post_human.csv: participant x item -> binary correct (after human-labeled advice)
"""

import json
import subprocess
from pathlib import Path

import pandas as pd

_BENCHMARK_DIR = Path(__file__).resolve().parent
RAW_DIR = _BENCHMARK_DIR / "raw"
OUTPUT_DIR = _BENCHMARK_DIR / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    repo_dir = RAW_DIR / "repo"

    if repo_dir.exists():
        print("repo already cloned, skipping")
        return

    print("Cloning HAIID repo...")
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/kailas-v/human-ai-interactions.git", str(repo_dir)],
        check=True,
    )
    print(f"Done. Raw files in {repo_dir}")


def main():
    download()

    data_path = RAW_DIR / "repo" / "haiid_dataset.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing: {data_path}")

    print("Loading HAIID data...")
    df = pd.read_csv(data_path, low_memory=False)
    print(f"Raw data: {len(df)} rows, {len(df.columns)} columns")
    print(f"Domains: {df['task_name'].unique()}")
    print(f"Participants: {df['participant_id'].nunique()}")
    print(f"Items: {df['task_instance_id'].nunique()}")
    print(f"Advice sources: {df['advice_source'].unique()}")

    # Binarize responses: positive = correct, negative/zero = incorrect
    df["correct_pre"] = (df["response_1"] > 0).astype(int)
    df["correct_post"] = (df["response_2"] > 0).astype(int)

    # Save full intervention table
    df.to_csv(OUTPUT_DIR / "intervention_table.csv", index=False)
    print(f"\nSaved intervention_table.csv ({len(df)} rows)")

    # Build item_content.csv: extract content for each task instance
    tasks_dir = RAW_DIR / "repo" / "tasks"
    items = df.drop_duplicates(subset=["task_instance_id"]).sort_values("task_instance_id")

    # Load dermatology metadata for richer descriptions
    derm_meta_path = tasks_dir / "dermatology" / "metadata.csv"
    derm_meta = {}
    if derm_meta_path.exists():
        derm_df = pd.read_csv(derm_meta_path)
        for _, dm in derm_df.iterrows():
            derm_meta[dm["human-ai-dataset-map"]] = (
                f"sex: {dm['sex']}, age: {dm['age_approx']}, "
                f"site: {dm['anatom_site_general_challenge']}, diagnosis: {dm['diagnosis']}"
            )

    item_content_rows = []
    for _, row in items.iterrows():
        item_id = row["task_instance_id"]
        task_path = row["path_to_task"]
        correct_label = row["correct_label"]
        incorrect_label = row["incorrect_label"]
        task_file = tasks_dir / task_path
        content = None
        if task_file.exists():
            suffix = task_file.suffix.lower()
            if suffix == ".json":
                try:
                    with open(task_file) as f:
                        data = json.load(f)
                    content = (
                        f"Classify income (correct: {correct_label}). "
                        + ", ".join(f"{k}: {v}" for k, v in data.items())
                    )
                except Exception:
                    pass
            elif suffix == ".txt":
                try:
                    content = f"Classify sarcasm (correct: {correct_label}). Text: {task_file.read_text().strip()}"
                except Exception:
                    pass
        if content is None:
            domain = row["task_name"]
            if item_id in derm_meta:
                content = (
                    f"dermatology classification (correct: {correct_label}, distractor: {incorrect_label}). "
                    f"{derm_meta[item_id]}, image: {task_path}"
                )
            else:
                content = (
                    f"{domain} classification instance (correct: {correct_label}, "
                    f"distractor: {incorrect_label}), image: {task_path}"
                )
        item_content_rows.append({"item_id": item_id, "content": content})
    item_content_df = pd.DataFrame(item_content_rows)
    item_content_df.to_csv(OUTPUT_DIR / "item_content.csv", index=False)
    print(f"\nSaved item_content.csv ({len(item_content_df)} items)")

    # Build per-domain matrices
    for domain in sorted(df["task_name"].unique()):
        domain_df = df[df["task_name"] == domain]
        n_participants = domain_df["participant_id"].nunique()
        n_items = domain_df["task_instance_id"].nunique()
        print(f"\n--- {domain}: {n_participants} participants x {n_items} items ---")

        # Pre-advice matrix (same regardless of advice source — response before seeing advice)
        pre_matrix = domain_df.drop_duplicates(subset=["participant_id", "task_instance_id"]).pivot_table(
            index="participant_id", columns="task_instance_id", values="correct_pre", aggfunc="first"
        )
        pre_matrix.to_csv(OUTPUT_DIR / f"response_matrix_{domain}_pre.csv")
        print(f"  Pre-advice: {pre_matrix.shape[0]} x {pre_matrix.shape[1]}, "
              f"accuracy={pre_matrix.mean().mean():.3f}")

        # Post-advice matrices split by advice source
        for source in ["ai", "human"]:
            source_df = domain_df[domain_df["advice_source"] == source]
            if len(source_df) == 0:
                continue
            post_matrix = source_df.pivot_table(
                index="participant_id", columns="task_instance_id", values="correct_post", aggfunc="first"
            )
            post_matrix.to_csv(OUTPUT_DIR / f"response_matrix_{domain}_post_{source}.csv")
            print(f"  Post-advice ({source}): {post_matrix.shape[0]} x {post_matrix.shape[1]}, "
                  f"accuracy={post_matrix.mean().mean():.3f}")

    # Summary stats
    print("\n=== Overall Summary ===")
    for domain in sorted(df["task_name"].unique()):
        domain_df = df[df["task_name"] == domain]
        pre_acc = domain_df["correct_pre"].mean()
        for source in ["ai", "human"]:
            source_df = domain_df[domain_df["advice_source"] == source]
            post_acc = source_df["correct_post"].mean()
            delta = post_acc - pre_acc
            print(f"  {domain} ({source} advice): pre={pre_acc:.3f} -> post={post_acc:.3f} (delta={delta:+.3f})")


if __name__ == "__main__":
    main()

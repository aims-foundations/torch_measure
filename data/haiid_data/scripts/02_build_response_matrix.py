"""
Build intervention matrices from HAIID (Human-AI Interactions Dataset).

Source: https://github.com/kailas-v/human-ai-interactions
Paper: Vodrahalli et al., "Do Humans Trust Advice More if it Comes from AI?"

Structure: participants x items x {pre-advice, post-advice} x {ai_source, human_source} -> response
5 domains: art, cities, sarcasm, census, dermatology

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

from pathlib import Path

import pandas as pd

_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = _BENCHMARK_DIR / "raw"
OUTPUT_DIR = _BENCHMARK_DIR / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    data_path = RAW_DIR / "repo" / "haiid_dataset.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Run 01_download_raw.sh first. Missing: {data_path}")

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

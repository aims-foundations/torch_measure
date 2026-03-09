"""
Extract Table 2 results from Terminal-Bench paper (arXiv:2601.11868).
These are the 55 model-agent combinations evaluated in the paper.
"""

import json
import pandas as pd
from pathlib import Path

_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = str(_BENCHMARK_DIR / "raw")

# Table 2 from the paper (manually transcribed from HTML)
TABLE_2_DATA = [
    ("GPT-5.2", "Codex CLI", 62.9, 3.0, 137.5, 2.3),
    ("Claude Opus 4.5", "Terminus 2", 57.8, 2.5, 3.9, 1.3),
    ("Gemini 3 Pro", "Terminus 2", 56.9, 2.5, 5.1, 2.2),
    ("GPT-5.2", "Terminus 2", 54.0, 2.9, 12.4, 2.6),
    ("Claude Opus 4.5", "Claude Code", 52.1, 2.5, 256.9, 0.8),
    ("Claude Opus 4.5", "OpenHands", 51.9, 2.9, 151.4, 1.4),
    ("Gemini 3 Flash", "Terminus 2", 51.7, 3.1, 52.1, 2.6),
    ("GPT-5", "Codex CLI", 49.6, 2.9, 2.6, 0.8),
    ("Claude Sonnet 4.5", "Terminus 2", 42.8, 2.8, 3.1, 1.1),
    ("Claude Sonnet 4.5", "Mini-SWE-Agent", 42.5, 2.8, 3.4, 1.4),
    ("GPT-5", "OpenHands", 41.5, 2.8, 2.8, 3.6),
    ("Claude Sonnet 4.5", "OpenHands", 40.3, 2.6, 3.4, 1.4),
    ("Claude Sonnet 4.5", "Claude Code", 40.1, 2.9, 2.0, 0.1),
    ("Claude Opus 4.1", "Terminus 2", 38.0, 2.6, 2.3, 0.9),
    ("Kimi K2 Thinking", "Terminus 2", 35.7, 2.8, 84.5, 1.6),
    ("GPT-5", "Terminus 2", 35.2, 3.1, 3.1, 2.1),
    ("Claude Opus 4.1", "Mini-SWE-Agent", 35.1, 2.5, 2.0, 0.9),
    ("Claude Opus 4.1", "OpenHands", 34.9, 2.6, 3.2, 1.3),
    ("Claude Opus 4.1", "Claude Code", 34.8, 2.9, 0.2, 0.3),
    ("GPT-5", "Mini-SWE-Agent", 33.9, 2.9, 1.8, 2.7),
    ("Gemini 2.5 Pro", "Terminus 2", 32.6, 3.0, 6.1, 2.9),
    ("GPT-5-Mini", "Codex CLI", 31.9, 3.0, 3.4, 0.7),
    ("MiniMax M2", "Terminus 2", 30.0, 2.7, 89.9, 1.5),
    ("Claude Haiku 4.5", "Mini-SWE-Agent", 29.8, 2.5, 3.6, 1.4),
    ("Grok 4", "Mini-SWE-Agent", 29.0, 4.6, 0.3, 0.1),
    ("Claude Haiku 4.5", "Terminus 2", 28.3, 2.9, 3.9, 1.3),
    ("Kimi K2 Instruct", "Terminus 2", 27.8, 2.5, 76.3, 0.9),
    ("GPT-5-Mini", "OpenHands", 27.7, 2.6, 6.0, 3.1),
    ("Claude Haiku 4.5", "Claude Code", 27.5, 2.8, 0.2, 0.3),
    ("Gemini 2.5 Pro", "Mini-SWE-Agent", 26.1, 2.5, 12.4, 3.7),
    ("Kimi K2 Instruct", "OpenHands", 25.6, 2.6, 129.2, 0.9),
    ("Grok Code Fast 1", "Mini-SWE-Agent", 24.5, 2.6, 1.6, 0.2),
    ("GLM 4.6", "Terminus 2", 24.5, 2.4, 5.4, 1.0),
    ("Qwen 3 Coder 480B", "OpenHands", 24.3, 2.5, 146.9, 1.1),
    ("GPT-5-Mini", "Terminus 2", 24.0, 2.5, 5.9, 1.9),
    ("Qwen 3 Coder 480B", "Terminus 2", 23.9, 2.8, 81.0, 0.8),
    ("Grok 4", "Terminus 2", 23.4, 2.9, 1.2, 0.3),
    ("GPT-5-Mini", "Mini-SWE-Agent", 22.2, 2.6, 2.9, 1.9),
    ("Grok 4", "OpenHands", 19.6, 3.5, 0.9, 0.1),
    ("Gemini 2.5 Pro", "Gemini CLI", 19.6, 2.9, 8.7, 2.5),
    ("GPT-OSS-120B", "Terminus 2", 18.7, 2.7, 13.4, 0.8),
    ("Gemini 2.5 Flash", "Mini-SWE-Agent", 17.1, 2.5, 18.4, 6.0),
    ("Gemini 2.5 Flash", "Terminus 2", 16.9, 2.4, 10.5, 3.1),
    ("Gemini 2.5 Pro", "OpenHands", 15.7, 2.6, 13.2, 0.9),
    ("Gemini 2.5 Flash", "OpenHands", 15.5, 2.3, 14.8, 2.8),
    ("Gemini 2.5 Flash", "Gemini CLI", 15.4, 2.3, 6.8, 1.5),
    ("Grok Code Fast 1", "Terminus 2", 14.5, 2.6, 1.6, 0.2),
    ("GPT-OSS-120B", "Mini-SWE-Agent", 14.2, 2.3, 8.1, 0.6),
    ("Claude Haiku 4.5", "OpenHands", 13.3, 2.6, 663.1, 3.3),
    ("GPT-5-Nano", "Codex CLI", 11.5, 2.3, 2.1, 0.8),
    ("GPT-5-Nano", "OpenHands", 9.5, 2.0, 16.7, 10.7),
    ("GPT-5-Nano", "Terminus 2", 7.9, 1.9, 13.7, 5.3),
    ("GPT-5-Nano", "Mini-SWE-Agent", 7.0, 1.9, 1.3, 2.2),
    ("GPT-OSS-20B", "Mini-SWE-Agent", 3.4, 1.4, 11.5, 0.8),
    ("GPT-OSS-20B", "Terminus 2", 3.1, 1.5, 75.5, 1.2),
]

COLUMNS = [
    "model", "agent", "resolution_rate_pct", "std_error",
    "input_tokens_M", "output_tokens_M",
]


def main():
    df = pd.DataFrame(TABLE_2_DATA, columns=COLUMNS)
    df.to_csv(f"{OUTPUT_DIR}/paper_table2_results.csv", index=False)

    records = df.to_dict(orient="records")
    with open(f"{OUTPUT_DIR}/paper_table2_results.json", "w") as f:
        json.dump(records, f, indent=2)

    print(f"Saved {len(df)} model-agent combinations from Table 2")
    print(f"\n=== Summary ===")
    print(f"Unique models: {df['model'].nunique()} -> {sorted(df['model'].unique())}")
    print(f"Unique agents: {df['agent'].nunique()} -> {sorted(df['agent'].unique())}")
    print(f"Resolution rate range: {df['resolution_rate_pct'].min():.1f}% - "
          f"{df['resolution_rate_pct'].max():.1f}%")

    # Best per model
    print(f"\n--- Best resolution rate per model ---")
    best = df.loc[df.groupby("model")["resolution_rate_pct"].idxmax()]
    for _, row in best.sort_values("resolution_rate_pct", ascending=False).iterrows():
        print(f"  {row['model']:25s} {row['resolution_rate_pct']:5.1f}% "
              f"(with {row['agent']})")


if __name__ == "__main__":
    main()

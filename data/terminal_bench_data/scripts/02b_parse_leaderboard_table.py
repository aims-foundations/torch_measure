"""
Parse the leaderboard HTML table into a clean CSV.
"""

import re
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path

_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = str(_BENCHMARK_DIR / "raw")


def main():
    with open(f"{RAW_DIR}/leaderboard_raw.html") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    table = soup.find("table")
    rows = table.find_all("tr")

    # Parse header
    header_cells = rows[0].find_all(["th", "td"])
    headers = [c.get_text(strip=True) for c in header_cells]
    print(f"Headers: {headers}")

    # Parse data rows
    records = []
    for row in rows[1:]:
        cells = row.find_all(["th", "td"])
        values = [c.get_text(strip=True) for c in cells]
        if len(values) < 8:
            continue

        # Parse accuracy like "77.3%±2.2"
        acc_str = values[7]
        acc_match = re.match(r"([\d.]+)%\s*[±+/-]+([\d.]+)", acc_str)
        if acc_match:
            accuracy = float(acc_match.group(1))
            std_error = float(acc_match.group(2))
        else:
            accuracy = None
            std_error = None

        records.append({
            "rank": int(values[1]) if values[1].isdigit() else None,
            "agent": values[2],
            "model": values[3],
            "date": values[4],
            "agent_org": values[5],
            "model_org": values[6],
            "accuracy_pct": accuracy,
            "std_error": std_error,
            "accuracy_raw": acc_str,
        })

    df = pd.DataFrame(records)
    df.to_csv(f"{RAW_DIR}/leaderboard_results.csv", index=False)
    print(f"\nSaved {len(df)} leaderboard entries to leaderboard_results.csv")

    # Summary
    print(f"\n=== Leaderboard Summary ===")
    print(f"Total entries: {len(df)}")
    print(f"Unique agents: {df['agent'].nunique()}")
    print(f"Unique models: {df['model'].nunique()}")
    print(f"Unique agent orgs: {df['agent_org'].nunique()}")
    print(f"Unique model orgs: {df['model_org'].nunique()}")
    print(f"Accuracy range: {df['accuracy_pct'].min():.1f}% - "
          f"{df['accuracy_pct'].max():.1f}%")
    print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")

    print(f"\n--- Top 10 ---")
    print(df[["rank", "agent", "model", "accuracy_pct", "std_error"]]
          .head(10).to_string(index=False))

    print(f"\n--- Unique Models ---")
    for m in sorted(df["model"].unique()):
        best = df[df["model"] == m]["accuracy_pct"].max()
        count = (df["model"] == m).sum()
        print(f"  {m}: best={best:.1f}%, {count} entries")

    print(f"\n--- Unique Agents ---")
    for a in sorted(df["agent"].unique()):
        best = df[df["agent"] == a]["accuracy_pct"].max()
        count = (df["agent"] == a).sum()
        print(f"  {a}: best={best:.1f}%, {count} entries")


if __name__ == "__main__":
    main()

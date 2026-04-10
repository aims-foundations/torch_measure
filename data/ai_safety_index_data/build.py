#!/usr/bin/env python3
"""01_build_response_matrix.py -- Download and build AI Safety Index response matrix.

Source: https://futureoflife.org/ai-safety-index-winter-2025/
Downloads:
  - winter2025_full.pdf: Winter 2025 full report (8 companies x 35 indicators x 6 domains)
  - summer2025_full.pdf: Summer 2025 full report (7 companies x 33 indicators)

Data extracted from Winter 2025 Full Report PDF (page 3).

Structure: 8 companies x 6 domains -> GPA score [0, 4.3]
This IS a response matrix: rows = AI companies (subjects), columns = safety domains (items),
values = continuous GPA scores.

Output:
  - response_matrix.csv: company x domain -> GPA
  - response_matrix_normalized.csv: company x domain -> GPA / 4.3 (normalized to [0, 1])
"""

import urllib.request
from pathlib import Path

import pandas as pd

_BENCHMARK_DIR = Path(__file__).resolve().parent
RAW_DIR = _BENCHMARK_DIR / "raw"
OUTPUT_DIR = _BENCHMARK_DIR / "processed"

REPORTS = {
    "winter2025_full.pdf":
        "https://futureoflife.org/wp-content/uploads/2025/12/AI-Safety-Index-Report_131225_Full_Report_Digital.pdf",
    "summer2025_full.pdf":
        "https://futureoflife.org/wp-content/uploads/2025/07/FLI-AI-Safety-Index-Report-Summer-2025.pdf",
}

# Grade to GPA conversion (US scale)
GRADE_TO_GPA = {
    "A+": 4.3, "A": 4.0, "A-": 3.7,
    "B+": 3.3, "B": 3.0, "B-": 2.7,
    "C+": 2.3, "C": 2.0, "C-": 1.7,
    "D+": 1.3, "D": 1.0, "D-": 0.7,
    "F": 0.0,
}

# Winter 2025 data extracted from report page 3
# Columns: domains; Rows: companies
WINTER_2025_GRADES = {
    "Anthropic": {"Risk Assessment": "B", "Current Harms": "C+", "Safety Frameworks": "C+",
                  "Existential Safety": "D", "Governance & Accountability": "B-",
                  "Information Sharing": "A-"},
    "OpenAI": {"Risk Assessment": "B", "Current Harms": "C-", "Safety Frameworks": "C+",
               "Existential Safety": "D", "Governance & Accountability": "C+",
               "Information Sharing": "B"},
    "Google DeepMind": {"Risk Assessment": "C+", "Current Harms": "C", "Safety Frameworks": "C+",
                        "Existential Safety": "D", "Governance & Accountability": "C-",
                        "Information Sharing": "C"},
    "xAI": {"Risk Assessment": "D", "Current Harms": "F", "Safety Frameworks": "D+",
            "Existential Safety": "F", "Governance & Accountability": "D",
            "Information Sharing": "C"},
    "Zhipu AI": {"Risk Assessment": "D+", "Current Harms": "D", "Safety Frameworks": "D-",
                 "Existential Safety": "F", "Governance & Accountability": "D",
                 "Information Sharing": "C-"},
    "Meta": {"Risk Assessment": "D", "Current Harms": "D+", "Safety Frameworks": "D+",
             "Existential Safety": "F", "Governance & Accountability": "D",
             "Information Sharing": "D-"},
    "DeepSeek": {"Risk Assessment": "D", "Current Harms": "D+", "Safety Frameworks": "F",
                 "Existential Safety": "F", "Governance & Accountability": "D",
                 "Information Sharing": "C-"},
    "Alibaba Cloud": {"Risk Assessment": "D", "Current Harms": "D+", "Safety Frameworks": "F",
                      "Existential Safety": "F", "Governance & Accountability": "D+",
                      "Information Sharing": "D+"},
}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for filename, url in REPORTS.items():
        dest = RAW_DIR / filename
        if dest.exists():
            print(f"{filename} already exists, skipping")
            continue
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, dest)
        print(f"Downloaded {filename}")

    print(f"Done. Raw files in {RAW_DIR}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    download()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building AI Safety Index response matrix (Winter 2025)...")

    # Convert grades to GPA
    gpa_data = {}
    for company, domains in WINTER_2025_GRADES.items():
        gpa_data[company] = {d: GRADE_TO_GPA[g] for d, g in domains.items()}

    df = pd.DataFrame(gpa_data).T
    df.index.name = "company"
    df.to_csv(OUTPUT_DIR / "response_matrix.csv")
    print(f"\nResponse matrix: {df.shape[0]} companies x {df.shape[1]} domains")
    print(df.to_string())

    # Save item content
    items = pd.DataFrame({
        "item_id": df.columns,
        "content": df.columns,
    })
    items.to_csv(OUTPUT_DIR / "item_content.csv", index=False)
    print(f"Saved item_content.csv ({len(items)} items)")

    # Normalized to [0, 1]
    df_norm = df / 4.3
    df_norm.to_csv(OUTPUT_DIR / "response_matrix_normalized.csv")

    # Also save the letter grades
    grade_df = pd.DataFrame({c: d for c, d in WINTER_2025_GRADES.items()}).T
    grade_df.index.name = "company"
    grade_df.to_csv(OUTPUT_DIR / "grades.csv")

    # Summary
    print(f"\nOverall GPA by company (mean across domains):")
    for company in df.index:
        mean_gpa = df.loc[company].mean()
        print(f"  {company}: {mean_gpa:.2f}")

    print(f"\nMean GPA by domain (across companies):")
    for domain in df.columns:
        mean_gpa = df[domain].mean()
        print(f"  {domain}: {mean_gpa:.2f}")


if __name__ == "__main__":
    main()

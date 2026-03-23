"""
Build response matrix from AI Safety Index (Future of Life Institute).

Source: https://futureoflife.org/ai-safety-index-winter-2025/
Data extracted from Winter 2025 Full Report PDF (page 3).

Structure: 8 companies x 6 domains -> GPA score [0, 4.3]
This IS a response matrix: rows = AI companies (subjects), columns = safety domains (items),
values = continuous GPA scores.

Output:
  - response_matrix.csv: company x domain -> GPA
  - response_matrix_normalized.csv: company x domain -> GPA / 4.3 (normalized to [0, 1])
"""

from pathlib import Path

import pandas as pd

_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = _BENCHMARK_DIR / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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


def main():
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

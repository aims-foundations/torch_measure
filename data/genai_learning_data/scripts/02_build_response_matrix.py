"""
Build intervention matrices from GenAICanHarmLearning.

Source: https://github.com/obastani/GenAICanHarmLearning
Paper: Bastani et al., "Generative AI without guardrails can harm learning", PNAS 2025

Structure: High school students x math problems x 3 conditions -> binary score
Conditions: control (no AI), GPTBase (vanilla ChatGPT), GPTTutor (guardrailed tutor)
Design: Between-subjects RCT in Turkish high schools

Data files:
  - final_data.csv: student-level aggregates (Part2Tot, Part3Tot = practice/exam scores)
  - problem_part2.csv: student x problem -> score during PRACTICE phase (with/without AI)
  - problem_part3.csv: student x problem -> score during EXAM phase (no AI for anyone)

The key measurement: Part 2 = practice with treatment, Part 3 = exam without treatment.
The "uplift" question: does practicing with AI help or hurt exam performance?

Output:
  - intervention_table.csv: student-level data
  - response_matrix_practice_{condition}.csv: student x problem -> score (Part 2, during treatment)
  - response_matrix_exam_{condition}.csv: student x problem -> score (Part 3, post-treatment exam)
"""

from pathlib import Path

import pandas as pd

_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = _BENCHMARK_DIR / "raw"
OUTPUT_DIR = _BENCHMARK_DIR / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    repo_dir = RAW_DIR / "repo" / "main_regressions"
    if not repo_dir.exists():
        raise FileNotFoundError(f"Run 01_download_raw.sh first. Missing: {repo_dir}")

    # Load student-level data
    student_path = repo_dir / "final_data.csv"
    print("Loading student-level data...")
    students = pd.read_csv(student_path)
    print(f"Students: {len(students)} rows")
    print(f"Columns: {list(students.columns)}")
    print(f"Treatment arms: {students['Treatment arm'].value_counts().to_dict()}")

    students.to_csv(OUTPUT_DIR / "intervention_table.csv", index=False)
    print(f"Saved intervention_table.csv")

    # Load problem-level data for Part 2 (practice, WITH treatment)
    p2_path = repo_dir / "problem_part2.csv"
    if p2_path.exists():
        print("\nLoading Part 2 (practice phase) problem-level data...")
        p2 = pd.read_csv(p2_path)
        print(f"Part 2: {len(p2)} rows, {p2['Student ID'].nunique()} students, "
              f"{p2['Problem'].nunique()} problems")
        print(f"Treatment arms: {p2['Treatment arm'].value_counts().to_dict()}")

        for arm in sorted(p2["Treatment arm"].unique()):
            arm_df = p2[p2["Treatment arm"] == arm]
            matrix = arm_df.pivot_table(index="Student ID", columns="Problem", values="Score", aggfunc="first")
            safe_arm = arm.replace(" ", "_").lower()
            matrix.to_csv(OUTPUT_DIR / f"response_matrix_practice_{safe_arm}.csv")
            print(f"  Practice {arm}: {matrix.shape[0]} students x {matrix.shape[1]} problems, "
                  f"mean score={matrix.mean().mean():.3f}")

    # Load problem-level data for Part 3 (exam, NO treatment for anyone)
    p3_path = repo_dir / "problem_part3.csv"
    if p3_path.exists():
        print("\nLoading Part 3 (exam phase) problem-level data...")
        p3 = pd.read_csv(p3_path)
        print(f"Part 3: {len(p3)} rows, {p3['Student ID'].nunique()} students, "
              f"{p3['Problem'].nunique()} problems")

        for arm in sorted(p3["Treatment arm"].unique()):
            arm_df = p3[p3["Treatment arm"] == arm]
            matrix = arm_df.pivot_table(index="Student ID", columns="Problem", values="Score", aggfunc="first")
            safe_arm = arm.replace(" ", "_").lower()
            matrix.to_csv(OUTPUT_DIR / f"response_matrix_exam_{safe_arm}.csv")
            print(f"  Exam {arm}: {matrix.shape[0]} students x {matrix.shape[1]} problems, "
                  f"mean score={matrix.mean().mean():.3f}")

    # Summary
    print("\n=== Summary ===")
    for arm in sorted(students["Treatment arm"].unique()):
        arm_students = students[students["Treatment arm"] == arm]
        print(f"  {arm}: N={len(arm_students)}, "
              f"practice mean={arm_students['Part2Tot'].mean():.3f}, "
              f"exam mean={arm_students['Part3Tot'].mean():.3f}")


if __name__ == "__main__":
    main()

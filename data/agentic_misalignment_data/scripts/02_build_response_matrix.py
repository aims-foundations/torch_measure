"""
02_build_response_matrix.py — Agentic Misalignment dataset exploration and processing.

Loads from raw/agentic-misalignment (git clone of anthropic-experimental/agentic-misalignment).
Expected structure:
  - configs/ — experiment configs (YAML) defining models and scenarios
  - templates/ — system prompt templates
  - classifiers/ — misalignment classifiers (murder, leak, blackmail, etc.)
  - scripts/ — experiment running and classification scripts
Finds model evaluation results, builds model x scenario matrix if data available.
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd

BENCHMARK_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BENCHMARK_DIR / "raw"
PROCESSED_DIR = BENCHMARK_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def list_raw_contents():
    """Recursively list files in raw/ (excluding .git), print summary."""
    print("=" * 60)
    print("FILES IN raw/")
    print("=" * 60)
    all_files = []
    for root, dirs, files in os.walk(RAW_DIR):
        dirs[:] = [d for d in dirs if d != ".git"]
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), RAW_DIR)
            all_files.append(rel)
    for f in sorted(all_files)[:80]:
        print(f"  {f}")
    if len(all_files) > 80:
        print(f"  ... and {len(all_files) - 80} more files")
    print(f"\nTotal files: {len(all_files)}")
    return all_files


def load_yaml_safe(fpath):
    """Load a YAML file safely."""
    try:
        import yaml

        with open(fpath) as f:
            return yaml.safe_load(f)
    except ImportError:
        print("  WARNING: PyYAML not installed. Trying manual parsing.")
        try:
            with open(fpath) as f:
                content = f.read()
            print(f"  Raw content (first 500 chars):\n{content[:500]}")
        except Exception as e:
            print(f"  Error reading {fpath}: {e}")
        return None
    except Exception as e:
        print(f"  Error loading YAML {fpath}: {e}")
        return None


def find_data_files(base_dir, extensions=(".json", ".csv", ".jsonl", ".yaml", ".yml", ".parquet")):
    """Find data files recursively, excluding .git."""
    data_files = []
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if d != ".git"]
        for f in files:
            if any(f.endswith(ext) for ext in extensions):
                data_files.append(os.path.join(root, f))
    return sorted(data_files)


def main():
    print("Agentic Misalignment Dataset Exploration")
    print("=" * 60)

    all_files = list_raw_contents()

    repo_dir = RAW_DIR / "agentic-misalignment"
    if not repo_dir.exists():
        for candidate in RAW_DIR.iterdir():
            if candidate.is_dir() and "misalignment" in candidate.name.lower():
                repo_dir = candidate
                break

    if not repo_dir.exists():
        print(f"\nERROR: Repo not found. Searched in {RAW_DIR}")
        return

    print(f"\nRepo directory: {repo_dir}")
    print(f"  Contents: {sorted(os.listdir(repo_dir))}")

    # Explore configs to understand experiment structure
    print("\n" + "=" * 60)
    print("EXPERIMENT CONFIGS")
    print("=" * 60)

    configs_dir = repo_dir / "configs"
    scenario_list = []
    model_list = []

    if configs_dir.exists():
        config_files = sorted(configs_dir.glob("*.yaml")) + sorted(configs_dir.glob("*.yml"))
        print(f"Config files: {[f.name for f in config_files]}")

        for cf in config_files:
            print(f"\n--- {cf.name} ---")
            config = load_yaml_safe(cf)
            if config is None:
                continue

            if isinstance(config, dict):
                print(f"  Top-level keys: {list(config.keys())}")
                for k, v in config.items():
                    if isinstance(v, list):
                        print(f"  {k}: list of {len(v)} items")
                        for item in v[:5]:
                            print(f"    - {str(item)[:120]}")
                        if len(v) > 5:
                            print(f"    ... and {len(v) - 5} more")
                        # Track models and scenarios
                        if any(kw in k.lower() for kw in ["model", "llm"]):
                            if isinstance(v[0], str):
                                model_list.extend(v)
                            elif isinstance(v[0], dict):
                                for item in v:
                                    if "name" in item:
                                        model_list.append(item["name"])
                                    elif "model" in item:
                                        model_list.append(item["model"])
                        if any(kw in k.lower() for kw in ["scenario", "task", "prompt"]):
                            if isinstance(v[0], str):
                                scenario_list.extend(v)
                            elif isinstance(v[0], dict):
                                for item in v:
                                    if "name" in item:
                                        scenario_list.append(item["name"])
                    elif isinstance(v, dict):
                        print(f"  {k}: dict with keys {list(v.keys())[:10]}")
                    else:
                        print(f"  {k}: {str(v)[:120]}")
    else:
        print("  configs/ directory not found")

    # Explore classifiers to understand what misalignment behaviors are tracked
    print("\n" + "=" * 60)
    print("MISALIGNMENT CLASSIFIERS")
    print("=" * 60)

    classifiers_dir = repo_dir / "classifiers"
    classifier_types = []

    if classifiers_dir.exists():
        classifier_files = sorted(classifiers_dir.glob("*.py"))
        print(f"Classifier files: {[f.name for f in classifier_files]}")

        for cf in classifier_files:
            if cf.name.startswith("_") or cf.name == "__init__.py":
                continue
            classifier_name = cf.stem.replace("_classifier", "")
            classifier_types.append(classifier_name)
            print(f"\n  {classifier_name}:")
            # Read first few lines to understand the classifier
            try:
                with open(cf) as f:
                    lines = f.readlines()
                # Look for class definition and docstring
                for i, line in enumerate(lines[:30]):
                    if "class " in line or '"""' in line or "def " in line:
                        print(f"    {line.rstrip()}")
            except Exception as e:
                print(f"    Error reading: {e}")
    else:
        print("  classifiers/ directory not found")

    # Explore templates for scenario structure
    print("\n" + "=" * 60)
    print("SYSTEM PROMPT TEMPLATES")
    print("=" * 60)

    templates_dir = repo_dir / "templates"
    if templates_dir.exists():
        template_files = sorted(templates_dir.glob("*"))
        print(f"Template files: {[f.name for f in template_files]}")

        for tf in template_files:
            if tf.suffix == ".py":
                print(f"\n--- {tf.name} ---")
                try:
                    with open(tf) as f:
                        content = f.read()
                    # Look for template definitions (strings, dicts, etc.)
                    lines = content.split("\n")
                    for i, line in enumerate(lines[:50]):
                        if any(kw in line.lower() for kw in ["template", "prompt", "scenario", "def ", "class "]):
                            print(f"  L{i+1}: {line.rstrip()[:120]}")
                except Exception as e:
                    print(f"  Error: {e}")

    # Look for any results/data files
    print("\n" + "=" * 60)
    print("SEARCHING FOR RESULTS DATA")
    print("=" * 60)

    data_files = find_data_files(repo_dir)
    print(f"Data files found: {len(data_files)}")
    for f in data_files:
        rel = os.path.relpath(f, repo_dir)
        size_kb = os.path.getsize(f) / 1024
        print(f"  {rel} ({size_kb:.1f} KB)")

    # Try loading any JSON/CSV results
    loaded_results = {}
    for fpath in data_files:
        if fpath.endswith(".csv"):
            try:
                df = pd.read_csv(fpath)
                rel = os.path.relpath(fpath, repo_dir)
                print(f"\n  Loaded CSV {rel}: {df.shape}")
                print(f"    Columns: {list(df.columns)}")
                print(f"    Sample:\n{df.head(3).to_string()}")
                loaded_results[rel] = df
            except Exception as e:
                print(f"  Error loading CSV: {e}")
        elif fpath.endswith(".json") and os.path.getsize(fpath) < 10_000_000:
            try:
                with open(fpath) as f:
                    data = json.load(f)
                rel = os.path.relpath(fpath, repo_dir)
                if isinstance(data, list):
                    print(f"\n  JSON {rel}: list of {len(data)} items")
                    if data and isinstance(data[0], dict):
                        try:
                            df = pd.DataFrame(data)
                            print(f"    As DataFrame: {df.shape}, columns: {list(df.columns)}")
                            loaded_results[rel] = df
                        except Exception:
                            pass
                elif isinstance(data, dict):
                    print(f"\n  JSON {rel}: dict with {len(data)} keys: {list(data.keys())[:10]}")
            except Exception as e:
                print(f"  Error loading JSON: {e}")

    # Build model x scenario matrix if we have results
    print("\n" + "=" * 60)
    print("MODEL x SCENARIO MATRIX")
    print("=" * 60)

    if loaded_results:
        for rel, df in loaded_results.items():
            model_cols = [c for c in df.columns if any(kw in c.lower() for kw in ["model", "llm", "agent"])]
            scenario_cols = [c for c in df.columns if any(kw in c.lower() for kw in ["scenario", "task", "prompt"])]
            score_cols = [c for c in df.columns if df[c].dtype in ("float64", "int64", "float32", "int32")]

            if model_cols and scenario_cols and score_cols:
                print(f"\n  Building pivot from {rel}")
                try:
                    pivot = df.pivot_table(
                        index=scenario_cols[0],
                        columns=model_cols[0],
                        values=score_cols[0],
                        aggfunc="mean",
                    )
                    print(pivot.to_string())
                    pivot.to_csv(PROCESSED_DIR / "model_x_scenario_matrix.csv")
                    print(f"  -> Saved to processed/model_x_scenario_matrix.csv")
                except Exception as e:
                    print(f"  Error building pivot: {e}")

            # Save processed version
            out_name = f"agentic_{Path(rel).stem}.csv"
            df.to_csv(PROCESSED_DIR / out_name, index=False)
            print(f"  -> Saved to processed/{out_name}")
    else:
        print("  No tabular results found. This repo appears to be code-only.")
        print("  The experiment framework generates results when run against LLM APIs.")

    # Save what we know about the experiment structure
    print("\n" + "=" * 60)
    print("EXPERIMENT STRUCTURE SUMMARY")
    print("=" * 60)

    unique_models = sorted(set(model_list))
    unique_scenarios = sorted(set(scenario_list))
    unique_classifiers = sorted(set(classifier_types))

    print(f"Models found in configs: {unique_models}")
    print(f"Scenarios found in configs: {unique_scenarios}")
    print(f"Classifier types: {unique_classifiers}")

    structure_rows = []
    for m in unique_models:
        structure_rows.append({"type": "model", "name": m})
    for s in unique_scenarios:
        structure_rows.append({"type": "scenario", "name": s})
    for c in unique_classifiers:
        structure_rows.append({"type": "classifier", "name": c})

    if structure_rows:
        pd.DataFrame(structure_rows).to_csv(PROCESSED_DIR / "experiment_structure.csv", index=False)
        print(f"\n  -> Saved to processed/experiment_structure.csv")

    # Overall summary
    summary_rows = [
        {"metric": "n_models_in_configs", "value": len(unique_models)},
        {"metric": "n_scenarios_in_configs", "value": len(unique_scenarios)},
        {"metric": "n_classifier_types", "value": len(unique_classifiers)},
        {"metric": "n_config_files", "value": len(list(configs_dir.glob("*.yaml"))) if configs_dir.exists() else 0},
        {"metric": "n_data_files", "value": len(data_files)},
        {"metric": "n_loaded_results", "value": len(loaded_results)},
    ]
    pd.DataFrame(summary_rows).to_csv(PROCESSED_DIR / "summary_statistics.csv", index=False)
    print(f"  -> Saved to processed/summary_statistics.csv")

    print("\nDone!")


if __name__ == "__main__":
    main()

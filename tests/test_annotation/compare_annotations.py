"""Annotation comparison: all cached annotators vs Paper (GPT-4o).

Auto-discovers every *_comparison_cache.jsonl file in this directory and plots
each annotator's agreement with the paper's reference scores. Add a new model's
cache file and it appears in the figure automatically.

Run from the torch_measure directory:
    python tests/test_annotation/compare_annotations.py
"""
import csv
import json
import math
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np

HERE      = Path(__file__).parent
REPO_ROOT = HERE.parent.parent.parent  # AIMS_local/
PAPER_CSV = REPO_ROOT / "ADeLe-AIEvaluation" / "ADeLe_battery_data" / "ADeLe_batterry_v1dot0.csv"

DIMENSION_ORDER = (
    "AS","CEc","CEe","CL","MCr","MCt","MCu","MS",
    "QLl","QLq","SNs","KNa","KNc","KNf","KNn","KNs","AT","VO","UG",
)
DEMAND_DIMS = DIMENSION_ORDER[:18]

ITEM_IDS = [
    "ChemLLMBench-molecule_captioning-522",
    "ChemLLMBench-name_prediction-278",
    "ChemLLMBench-reaction_prediction-40",
    "ChemLLMBench-retrosynthesis-926",
    "Civil Service Examination-LogiQA-en-458",
    "Date Arithmetic-Date Arithmetic-52",
    "LSAT-LSAT-AR-197",
    "MCTACO-MCTACO-313",
    "MMLU-Pro-economics-474",
    "MMLU-Pro-history-360",
    "MMLU-Pro-physics-330",
    "MedCalcBench-physical-50",
    "OmniMath-Algebra-537",
    "SciBench-Chemistry-126",
    "TimeQA-TimeQA-implicit-81",
]

ITEM_SHORT = [
    "Chem\nmol","Chem\nname","Chem\nrxn","Chem\nretro",
    "LogiQA","DateArith","LSAT","MCTACO",
    "MMLU\neco","MMLU\nhist","MMLU\nphys",
    "MedCalc","OmniMath","SciBench","TimeQA",
]

# Colour palette — extended to support many annotators
_PALETTE = [
    "#4285F4",  # blue
    "#D97706",  # amber
    "#16A34A",  # green
    "#DC2626",  # red
    "#7C3AED",  # purple
    "#0891B2",  # cyan
    "#EA580C",  # orange
    "#BE185D",  # pink
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_cache(path: Path) -> tuple[dict, str]:
    """JSONL cache → ({item_id: {dim: level}}, model_id from first entry)."""
    data: dict = {}
    model_id = path.stem  # fallback label if cache is empty
    with open(path, encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            if model_id == path.stem:
                model_id = e.get("model_id", path.stem)
            data.setdefault(e["item_id"], {})[e["demand"]] = e["level"]
    return data, model_id


def _load_paper(csv_path: Path) -> dict:
    """CSV → {instance_id: {dim: score}} for the 15 test items."""
    target = set(ITEM_IDS)
    scores: dict = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["instance_id"] in target:
                scores[row["instance_id"]] = {d: float(row[d]) for d in DIMENSION_ORDER}
    return scores


def _discover_caches() -> list[tuple[Path, str]]:
    """Return (path, model_id) for every cache file in this directory, oldest first."""
    candidates = sorted(
        list(HERE.glob("*_comparison_cache.jsonl")) + list(HERE.glob("paper_comparison_cache.jsonl")),
        key=lambda p: p.stat().st_mtime,
    )
    seen = set()
    result = []
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        _, mid = _load_cache(p)
        result.append((p, mid))
    return result


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _ok(v) -> bool:
    return v is not None and not math.isnan(float(v))


def _valid_pairs(model: dict, paper: dict, dims) -> list[tuple[float, float]]:
    out = []
    for iid in ITEM_IDS:
        for dim in dims:
            m = model.get(iid, {}).get(dim, float("nan"))
            p = paper.get(iid, {}).get(dim, float("nan"))
            if _ok(m) and _ok(p):
                out.append((float(p), float(m)))
    return out


def _per_dim_stat(model: dict, paper: dict, dims, fn) -> dict:
    result = {}
    for dim in dims:
        vals = []
        for iid in ITEM_IDS:
            m = model.get(iid, {}).get(dim, float("nan"))
            p = paper.get(iid, {}).get(dim, float("nan"))
            if _ok(m) and _ok(p):
                vals.append(fn(float(p), float(m)))
        result[dim] = mean(vals) if vals else float("nan")
    return result


def _per_item_mad(model: dict, paper: dict, dims) -> list[float]:
    mads = []
    for iid in ITEM_IDS:
        diffs = []
        for dim in dims:
            m = model.get(iid, {}).get(dim, float("nan"))
            p = paper.get(iid, {}).get(dim, float("nan"))
            if _ok(m) and _ok(p):
                diffs.append(abs(float(m) - float(p)))
        mads.append(mean(diffs) if diffs else float("nan"))
    return mads


def _overall(pairs: list[tuple[float, float]]) -> dict:
    diffs  = [m - p for p, m in pairs]
    adiffs = [abs(d) for d in diffs]
    n = len(diffs)
    return {
        "MAE":    mean(adiffs),
        "Bias":   mean(diffs),
        "Exact%": 100 * sum(d == 0 for d in diffs) / n,
        "±1%":    100 * sum(abs(d) <= 1 for d in diffs) / n,
        "n":      n,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    caches = _discover_caches()
    if not caches:
        print("No *_comparison_cache.jsonl files found.")
        return

    paper = _load_paper(PAPER_CSV)

    annotators = []
    for path, model_id in caches:
        data, _ = _load_cache(path)
        pairs    = _valid_pairs(data, paper, DEMAND_DIMS)
        annotators.append({
            "label":    model_id,
            "data":     data,
            "pairs":    pairs,
            "stats":    _overall(pairs),
            "dim_mae":  _per_dim_stat(data, paper, DEMAND_DIMS, lambda p, m: abs(m - p)),
            "dim_bias": _per_dim_stat(data, paper, DEMAND_DIMS, lambda p, m: m - p),
            "item_mad": _per_item_mad(data, paper, DEMAND_DIMS),
        })

    n_ann = len(annotators)
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(n_ann)]
    bar_w  = min(0.7 / n_ann, 0.25)
    offsets = np.linspace(-(n_ann - 1) / 2 * bar_w, (n_ann - 1) / 2 * bar_w, n_ann)
    x18    = np.arange(len(DEMAND_DIMS))
    x15    = np.arange(len(ITEM_IDS))
    rng    = np.random.default_rng(42)

    fig = plt.figure(figsize=(22, 15))
    fig.suptitle(
        f"Annotator Comparison vs Paper (GPT-4o)  —  {len(ITEM_IDS)} items",
        fontsize=15, fontweight="bold", y=0.99,
    )

    # --- Panel 1: Per-dimension MAE ---
    ax1 = fig.add_subplot(2, 2, 1)
    for i, (ann, col, off) in enumerate(zip(annotators, colors, offsets)):
        vals = [ann["dim_mae"].get(d, float("nan")) for d in DEMAND_DIMS]
        ax1.bar(x18 + off, vals, bar_w, label=ann["label"], color=col, alpha=0.85)
        ax1.axhline(ann["stats"]["MAE"], color=col, linestyle="--",
                    linewidth=0.9, alpha=0.55)
    ax1.set_xticks(x18)
    ax1.set_xticklabels(DEMAND_DIMS, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("MAE vs Paper (0–5 scale)")
    ax1.set_title("Per-Dimension MAE vs Paper")
    ax1.legend(fontsize=7.5)
    ax1.set_ylim(0, 2.8)
    ax1.grid(axis="y", alpha=0.3)

    # --- Panel 2: Score scatter ---
    ax2 = fig.add_subplot(2, 2, 2)
    jit = 0.06
    for ann, col in zip(annotators, colors):
        pts = np.array(ann["pairs"])
        ax2.scatter(
            pts[:, 0] + rng.uniform(-jit, jit, len(pts)),
            pts[:, 1] + rng.uniform(-jit, jit, len(pts)),
            alpha=0.18, s=12, color=col, label=ann["label"],
        )
    ax2.plot([0, 5], [0, 5], "k--", alpha=0.4, linewidth=1.2, label="Perfect agreement")
    ax2.set_xlabel("Paper Score (GPT-4o)")
    ax2.set_ylabel("Annotator Score")
    ax2.set_title(f"Score Correlation vs Paper  ({len(ITEM_IDS)}×{len(DEMAND_DIMS)} pairs)")
    ax2.legend(fontsize=7.5)
    ax2.set_xlim(-0.4, 5.4); ax2.set_ylim(-0.4, 5.4)
    ax2.set_xticks(range(6)); ax2.set_yticks(range(6))
    ax2.grid(alpha=0.25)

    # --- Panel 3: Per-item MAD ---
    ax3 = fig.add_subplot(2, 2, 3)
    for ann, col, off in zip(annotators, colors, offsets):
        ax3.bar(x15 + off, ann["item_mad"], bar_w,
                label=ann["label"], color=col, alpha=0.85)
    ax3.set_xticks(x15)
    ax3.set_xticklabels(ITEM_SHORT, fontsize=7.5)
    ax3.set_ylabel("MAD vs Paper (demand dims)")
    ax3.set_title("Per-Item Divergence from Paper")
    ax3.legend(fontsize=7.5)
    ax3.grid(axis="y", alpha=0.3)

    # --- Panel 4: Bias per dimension ---
    ax4 = fig.add_subplot(2, 2, 4)
    for ann, col, off in zip(annotators, colors, offsets):
        vals = [ann["dim_bias"].get(d, float("nan")) for d in DEMAND_DIMS]
        ax4.bar(x18 + off, vals, bar_w, label=ann["label"], color=col, alpha=0.85)
        ax4.axhline(ann["stats"]["Bias"], color=col, linestyle="--",
                    linewidth=0.9, alpha=0.55)
    ax4.axhline(0, color="black", linewidth=0.9)
    ax4.set_xticks(x18)
    ax4.set_xticklabels(DEMAND_DIMS, rotation=45, ha="right", fontsize=8)
    ax4.set_ylabel("Mean (annotator − paper)")
    ax4.set_title("Systematic Bias per Dimension\n(negative = annotator scores lower than paper)")
    ax4.legend(fontsize=7.5)
    ax4.grid(axis="y", alpha=0.3)

    # --- Summary table ---
    header = f"{'Metric':<14}" + "".join(f"{a['label']:>20}" for a in annotators)
    rows = []
    for key, fmt in [("MAE", ".3f"), ("Bias", "+.3f"), ("Exact%", ".1f"), ("±1%", ".1f"), ("n", "d")]:
        row = f"{key:<14}" + "".join(f"{format(a['stats'][key], fmt):>20}" for a in annotators)
        rows.append(row)
    summary = "\n".join([header, "─" * (14 + 20 * n_ann)] + rows)
    fig.text(0.5, 0.005, summary, ha="center", fontsize=8.5, fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF9C4", alpha=0.9))

    plt.tight_layout(rect=[0, 0.10 + 0.015 * n_ann, 1, 0.97])

    out = HERE / "annotation_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nAnnotators plotted: {[a['label'] for a in annotators]}")
    print(f"Saved -> {out}")
    plt.show()


if __name__ == "__main__":
    main()

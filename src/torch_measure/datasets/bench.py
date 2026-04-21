"""Info-gathering benchmark dataset definitions.

This module registers response matrices from 21 LLM/agent benchmarks curated
in the info_gathering research project.  Each response matrix follows the
standard torch_measure convention:

- **Rows (subjects)**: AI models or agents.
- **Columns (items)**: Individual tasks / items from each benchmark.
- **Values**: Binary {0, 1} or continuous [0, 1] scores (NaN for missing).

These datasets complement the existing ``agentic`` family (sourced from the HAL
leaderboard) and cover a wider set of coding, agentic, and knowledge benchmarks
with generally larger model counts.

Data files live on HuggingFace Hub at ``aims-foundation/torch-measure-data``
(e.g. ``swebench.pt``).
"""

from __future__ import annotations

from torch_measure.datasets._info import DatasetInfo

_BENCH_REPO = "aims-foundation/torch-measure-data"


def _register_bench_datasets() -> dict[str, DatasetInfo]:
    """Return registry entries for info-gathering benchmark response matrices."""
    d: dict[str, DatasetInfo] = {}

    # ── AgentDojo ──────────────────────────────────────────────────────
    d["agentdojo"] = DatasetInfo(
        name="agentdojo",
        family="bench",
        description="AgentDojo — tool-use agent utility tasks",
        response_type="binary",
        n_subjects=29,
        n_items=132,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="agentdojo.pt",
        url="https://agentdojo.spylab.ai/",
        tags=["agentic", "tool-use", "security"],
    )
    d["agentdojo_security"] = DatasetInfo(
        name="agentdojo_security",
        family="bench",
        description="AgentDojo — security evaluation",
        response_type="binary",
        n_subjects=28,
        n_items=949,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="agentdojo_security.pt",
        url="https://agentdojo.spylab.ai/",
        tags=["agentic", "tool-use", "security"],
    )
    d["agentdojo_utility_attack"] = DatasetInfo(
        name="agentdojo_utility_attack",
        family="bench",
        description="AgentDojo — utility under attack",
        response_type="binary",
        n_subjects=28,
        n_items=97,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="agentdojo_utility_attack.pt",
        url="https://agentdojo.spylab.ai/",
        tags=["agentic", "tool-use", "security"],
    )

    # ── AppWorld ───────────────────────────────────────────────────────
    d["appworld"] = DatasetInfo(
        name="appworld",
        family="bench",
        description="AppWorld — multi-app interaction tasks",
        response_type="continuous",
        n_subjects=18,
        n_items=24,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="appworld.pt",
        url="https://appworld.dev/",
        tags=["agentic", "app-interaction"],
    )

    # ── BFCL ──────────────────────────────────────────────────────────
    d["bfcl"] = DatasetInfo(
        name="bfcl",
        family="bench",
        description="BFCL v3 — function calling (pass/fail)",
        response_type="binary",
        n_subjects=93,
        n_items=4751,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bfcl.pt",
        url="https://gorilla.cs.berkeley.edu/leaderboard.html",
        tags=["coding", "function-calling"],
    )

    # ── BigCodeBench ──────────────────────────────────────────────────
    d["bigcodebench"] = DatasetInfo(
        name="bigcodebench",
        family="bench",
        description="BigCodeBench — complete split",
        response_type="binary",
        n_subjects=153,
        n_items=1140,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bigcodebench.pt",
        url="https://bigcode-bench.github.io/",
        tags=["coding"],
    )
    d["bigcodebench_instruct"] = DatasetInfo(
        name="bigcodebench_instruct",
        family="bench",
        description="BigCodeBench — instruct split",
        response_type="binary",
        n_subjects=126,
        n_items=1140,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bigcodebench_instruct.pt",
        url="https://bigcode-bench.github.io/",
        tags=["coding"],
    )
    d["bigcodebench_hard_complete"] = DatasetInfo(
        name="bigcodebench_hard_complete",
        family="bench",
        description="BigCodeBench — hard tasks, complete",
        response_type="binary",
        n_subjects=199,
        n_items=148,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bigcodebench_hard_complete.pt",
        url="https://bigcode-bench.github.io/",
        tags=["coding"],
    )
    d["bigcodebench_hard_instruct"] = DatasetInfo(
        name="bigcodebench_hard_instruct",
        family="bench",
        description="BigCodeBench — hard tasks, instruct",
        response_type="binary",
        n_subjects=173,
        n_items=148,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bigcodebench_hard_instruct.pt",
        url="https://bigcode-bench.github.io/",
        tags=["coding"],
    )

    # ── ClineBench ────────────────────────────────────────────────────
    d["clinebench"] = DatasetInfo(
        name="clinebench",
        family="bench",
        description="ClineBench — AI coding agent evaluation",
        response_type="continuous",
        n_subjects=3,
        n_items=12,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="clinebench.pt",
        url="https://github.com/cline/cline",
        tags=["coding", "agentic"],
    )

    # ── CORE-Bench ────────────────────────────────────────────────────
    d["corebench"] = DatasetInfo(
        name="corebench",
        family="bench",
        description="CORE-Bench — computational reproducibility (binary)",
        response_type="binary",
        n_subjects=15,
        n_items=270,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="corebench.pt",
        url="https://github.com/siegelz/core-bench",
        tags=["agentic", "reproducibility"],
    )
    d["corebench_scores"] = DatasetInfo(
        name="corebench_scores",
        family="bench",
        description="CORE-Bench — computational reproducibility (scores)",
        response_type="continuous",
        n_subjects=15,
        n_items=270,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="corebench_scores.pt",
        url="https://github.com/siegelz/core-bench",
        tags=["agentic", "reproducibility"],
    )

    # ── CRUXEval ──────────────────────────────────────────────────────
    d["cruxeval"] = DatasetInfo(
        name="cruxeval",
        family="bench",
        description="CRUXEval — code reasoning (continuous)",
        response_type="continuous",
        n_subjects=38,
        n_items=800,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="cruxeval.pt",
        url="https://github.com/facebookresearch/cruxeval",
        tags=["coding", "reasoning"],
    )
    d["cruxeval_binary"] = DatasetInfo(
        name="cruxeval_binary",
        family="bench",
        description="CRUXEval — code reasoning (binary)",
        response_type="binary",
        n_subjects=38,
        n_items=800,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="cruxeval_binary.pt",
        url="https://github.com/facebookresearch/cruxeval",
        tags=["coding", "reasoning"],
    )
    d["cruxeval_input"] = DatasetInfo(
        name="cruxeval_input",
        family="bench",
        description="CRUXEval — input prediction (continuous)",
        response_type="continuous",
        n_subjects=20,
        n_items=800,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="cruxeval_input.pt",
        url="https://github.com/facebookresearch/cruxeval",
        tags=["coding", "reasoning"],
    )
    d["cruxeval_input_binary"] = DatasetInfo(
        name="cruxeval_input_binary",
        family="bench",
        description="CRUXEval — input prediction (binary)",
        response_type="binary",
        n_subjects=20,
        n_items=800,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="cruxeval_input_binary.pt",
        url="https://github.com/facebookresearch/cruxeval",
        tags=["coding", "reasoning"],
    )
    d["cruxeval_output"] = DatasetInfo(
        name="cruxeval_output",
        family="bench",
        description="CRUXEval — output prediction (continuous)",
        response_type="continuous",
        n_subjects=18,
        n_items=800,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="cruxeval_output.pt",
        url="https://github.com/facebookresearch/cruxeval",
        tags=["coding", "reasoning"],
    )
    d["cruxeval_output_binary"] = DatasetInfo(
        name="cruxeval_output_binary",
        family="bench",
        description="CRUXEval — output prediction (binary)",
        response_type="binary",
        n_subjects=18,
        n_items=800,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="cruxeval_output_binary.pt",
        url="https://github.com/facebookresearch/cruxeval",
        tags=["coding", "reasoning"],
    )

    # ── Cybench ───────────────────────────────────────────────────────
    d["cybench"] = DatasetInfo(
        name="cybench",
        family="bench",
        description="Cybench — CTF cybersecurity tasks (unguided)",
        response_type="binary",
        n_subjects=8,
        n_items=40,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="cybench.pt",
        url="https://cybench.github.io/",
        tags=["agentic", "cybersecurity", "ctf"],
    )
    d["cybench_guided"] = DatasetInfo(
        name="cybench_guided",
        family="bench",
        description="Cybench — CTF tasks (subtask-guided)",
        response_type="binary",
        n_subjects=8,
        n_items=40,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="cybench_guided.pt",
        url="https://cybench.github.io/",
        tags=["agentic", "cybersecurity", "ctf"],
    )
    d["cybench_scores"] = DatasetInfo(
        name="cybench_scores",
        family="bench",
        description="Cybench — CTF tasks (subtask completion scores)",
        response_type="continuous",
        n_subjects=8,
        n_items=40,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="cybench_scores.pt",
        url="https://cybench.github.io/",
        tags=["agentic", "cybersecurity", "ctf"],
    )

    # ── DPAI ──────────────────────────────────────────────────────────
    d["dpai"] = DatasetInfo(
        name="dpai",
        family="bench",
        description="DPAI Arena — total score",
        response_type="continuous",
        n_subjects=9,
        n_items=141,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="dpai.pt",
        url="https://dpaia.dev/",
        tags=["agentic", "coding"],
    )
    d["dpai_blind"] = DatasetInfo(
        name="dpai_blind",
        family="bench",
        description="DPAI Arena — blind evaluation score",
        response_type="continuous",
        n_subjects=9,
        n_items=141,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="dpai_blind.pt",
        url="https://dpaia.dev/",
        tags=["agentic", "coding"],
    )
    d["dpai_informed"] = DatasetInfo(
        name="dpai_informed",
        family="bench",
        description="DPAI Arena — informed evaluation score",
        response_type="continuous",
        n_subjects=9,
        n_items=141,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="dpai_informed.pt",
        url="https://dpaia.dev/",
        tags=["agentic", "coding"],
    )
    d["dpai_binary"] = DatasetInfo(
        name="dpai_binary",
        family="bench",
        description="DPAI Arena — binary pass/fail (50% threshold)",
        response_type="binary",
        n_subjects=9,
        n_items=141,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="dpai_binary.pt",
        url="https://dpaia.dev/",
        tags=["agentic", "coding"],
    )

    # ── EDIT-Bench ────────────────────────────────────────────────────
    d["editbench"] = DatasetInfo(
        name="editbench",
        family="bench",
        description="EDIT-Bench — code editing (continuous scores)",
        response_type="continuous",
        n_subjects=44,
        n_items=540,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="editbench.pt",
        url="https://github.com/waynchi/editbench",
        tags=["coding", "editing"],
    )
    d["editbench_binary"] = DatasetInfo(
        name="editbench_binary",
        family="bench",
        description="EDIT-Bench — code editing (binary)",
        response_type="binary",
        n_subjects=44,
        n_items=540,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="editbench_binary.pt",
        url="https://github.com/waynchi/editbench",
        tags=["coding", "editing"],
    )

    # ── EvalPlus ──────────────────────────────────────────────────────
    d["evalplus"] = DatasetInfo(
        name="evalplus",
        family="bench",
        description="EvalPlus — HumanEval+ and MBPP+ combined",
        response_type="binary",
        n_subjects=31,
        n_items=164,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="evalplus.pt",
        url="https://evalplus.github.io/",
        tags=["coding"],
    )
    d["evalplus_humaneval_base"] = DatasetInfo(
        name="evalplus_humaneval_base",
        family="bench",
        description="EvalPlus — HumanEval base",
        response_type="binary",
        n_subjects=31,
        n_items=164,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="evalplus_humaneval_base.pt",
        url="https://evalplus.github.io/",
        tags=["coding"],
    )
    d["evalplus_humaneval_plus"] = DatasetInfo(
        name="evalplus_humaneval_plus",
        family="bench",
        description="EvalPlus — HumanEval+",
        response_type="binary",
        n_subjects=31,
        n_items=164,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="evalplus_humaneval_plus.pt",
        url="https://evalplus.github.io/",
        tags=["coding"],
    )
    d["evalplus_mbpp_base"] = DatasetInfo(
        name="evalplus_mbpp_base",
        family="bench",
        description="EvalPlus — MBPP base",
        response_type="binary",
        n_subjects=22,
        n_items=378,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="evalplus_mbpp_base.pt",
        url="https://evalplus.github.io/",
        tags=["coding"],
    )
    d["evalplus_mbpp_plus"] = DatasetInfo(
        name="evalplus_mbpp_plus",
        family="bench",
        description="EvalPlus — MBPP+",
        response_type="binary",
        n_subjects=22,
        n_items=378,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="evalplus_mbpp_plus.pt",
        url="https://evalplus.github.io/",
        tags=["coding"],
    )

    # ── LiveCodeBench ─────────────────────────────────────────────────
    d["livecodebench"] = DatasetInfo(
        name="livecodebench",
        family="bench",
        description="LiveCodeBench — competitive programming",
        response_type="continuous",
        n_subjects=72,
        n_items=1055,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="livecodebench.pt",
        url="https://livecodebench.github.io/",
        tags=["coding", "competitive-programming"],
    )

    # ── MLE-bench ─────────────────────────────────────────────────────
    d["mlebench"] = DatasetInfo(
        name="mlebench",
        family="bench",
        description="MLE-bench — ML engineering (continuous scores)",
        response_type="continuous",
        n_subjects=30,
        n_items=75,
        subject_entity="LLM",
        item_entity="competition",
        repo_id=_BENCH_REPO,
        filename="mlebench.pt",
        url="https://github.com/openai/mle-bench",
        tags=["agentic", "machine-learning"],
    )
    d["mlebench_binary"] = DatasetInfo(
        name="mlebench_binary",
        family="bench",
        description="MLE-bench — ML engineering (binary)",
        response_type="binary",
        n_subjects=30,
        n_items=75,
        subject_entity="LLM",
        item_entity="competition",
        repo_id=_BENCH_REPO,
        filename="mlebench_binary.pt",
        url="https://github.com/openai/mle-bench",
        tags=["agentic", "machine-learning"],
    )
    d["mlebench_above_median"] = DatasetInfo(
        name="mlebench_above_median",
        family="bench",
        description="MLE-bench — ML engineering (above-median)",
        response_type="binary",
        n_subjects=30,
        n_items=75,
        subject_entity="LLM",
        item_entity="competition",
        repo_id=_BENCH_REPO,
        filename="mlebench_above_median.pt",
        url="https://github.com/openai/mle-bench",
        tags=["agentic", "machine-learning"],
    )
    d["mlebench_scores"] = DatasetInfo(
        name="mlebench_scores",
        family="bench",
        description="MLE-bench — ML engineering (raw Kaggle scores)",
        response_type="continuous",
        n_subjects=30,
        n_items=75,
        subject_entity="LLM",
        item_entity="competition",
        repo_id=_BENCH_REPO,
        filename="mlebench_scores.pt",
        url="https://github.com/openai/mle-bench",
        tags=["agentic", "machine-learning"],
    )

    # ── MMLU-Pro ──────────────────────────────────────────────────────
    d["mmlupro"] = DatasetInfo(
        name="mmlupro",
        family="bench",
        description="MMLU-Pro — per-question accuracy (12K+ models)",
        response_type="continuous",
        n_subjects=48,
        n_items=12257,
        subject_entity="LLM",
        item_entity="question",
        repo_id=_BENCH_REPO,
        filename="mmlupro.pt",
        url="https://github.com/TIGER-AI-Lab/MMLU-Pro",
        tags=["knowledge", "reasoning"],
    )
    d["mmlupro_category"] = DatasetInfo(
        name="mmlupro_category",
        family="bench",
        description="MMLU-Pro — per-category accuracy",
        response_type="continuous",
        n_subjects=247,
        n_items=14,
        subject_entity="LLM",
        item_entity="category",
        repo_id=_BENCH_REPO,
        filename="mmlupro_category.pt",
        url="https://github.com/TIGER-AI-Lab/MMLU-Pro",
        tags=["knowledge", "reasoning"],
    )

    # ── PaperBench ────────────────────────────────────────────────────
    d["paperbench"] = DatasetInfo(
        name="paperbench",
        family="bench",
        description="PaperBench — paper reproduction evaluation",
        response_type="continuous",
        n_subjects=9,
        n_items=20,
        subject_entity="LLM",
        item_entity="paper",
        repo_id=_BENCH_REPO,
        filename="paperbench.pt",
        url="https://arxiv.org/abs/2504.01848",
        tags=["agentic", "research"],
    )
    d["paperbench_runs"] = DatasetInfo(
        name="paperbench_runs",
        family="bench",
        description="PaperBench — per-run results",
        response_type="continuous",
        n_subjects=7,
        n_items=180,
        subject_entity="LLM",
        item_entity="paper-run",
        repo_id=_BENCH_REPO,
        filename="paperbench_runs.pt",
        url="https://arxiv.org/abs/2504.01848",
        tags=["agentic", "research"],
    )

    # ── SWE-bench ─────────────────────────────────────────────────────
    d["swebench"] = DatasetInfo(
        name="swebench",
        family="bench",
        description="SWE-bench Verified — GitHub issue resolution",
        response_type="binary",
        n_subjects=134,
        n_items=500,
        subject_entity="LLM",
        item_entity="issue",
        repo_id=_BENCH_REPO,
        filename="swebench.pt",
        url="https://www.swebench.com/",
        tags=["coding", "agentic", "github"],
    )

    # ── SWE-PolyBench ─────────────────────────────────────────────────
    d["swepolybench_full"] = DatasetInfo(
        name="swepolybench_full",
        family="bench",
        description="SWE-PolyBench — multilingual SWE (full, 1 model)",
        response_type="binary",
        n_subjects=1,
        n_items=2110,
        subject_entity="LLM",
        item_entity="issue",
        repo_id=_BENCH_REPO,
        filename="swepolybench_full.pt",
        url="https://github.com/amazon-science/SWE-PolyBench",
        tags=["coding", "multilingual"],
    )
    d["swepolybench_verified"] = DatasetInfo(
        name="swepolybench_verified",
        family="bench",
        description="SWE-PolyBench — verified subset",
        response_type="binary",
        n_subjects=3,
        n_items=382,
        subject_entity="LLM",
        item_entity="issue",
        repo_id=_BENCH_REPO,
        filename="swepolybench_verified.pt",
        url="https://github.com/amazon-science/SWE-PolyBench",
        tags=["coding", "multilingual"],
    )

    # ── TAU-bench ─────────────────────────────────────────────────────
    d["taubench"] = DatasetInfo(
        name="taubench",
        family="bench",
        description="TAU-bench — all domains combined",
        response_type="continuous",
        n_subjects=32,
        n_items=329,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="taubench.pt",
        url="https://github.com/sierra-research/tau-bench",
        tags=["agentic", "customer-service"],
    )
    d["taubench_v1_airline"] = DatasetInfo(
        name="taubench_v1_airline",
        family="bench",
        description="TAU-bench — airline domain v1",
        response_type="continuous",
        n_subjects=28,
        n_items=50,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="taubench_v1_airline.pt",
        url="https://github.com/sierra-research/tau-bench",
        tags=["agentic", "customer-service"],
    )
    d["taubench_v2_airline"] = DatasetInfo(
        name="taubench_v2_airline",
        family="bench",
        description="TAU-bench — airline domain v2",
        response_type="continuous",
        n_subjects=4,
        n_items=50,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="taubench_v2_airline.pt",
        url="https://github.com/sierra-research/tau-bench",
        tags=["agentic", "customer-service"],
    )
    d["taubench_hal_airline"] = DatasetInfo(
        name="taubench_hal_airline",
        family="bench",
        description="TAU-bench — HAL airline evaluation",
        response_type="continuous",
        n_subjects=26,
        n_items=50,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="taubench_hal_airline.pt",
        url="https://github.com/sierra-research/tau-bench",
        tags=["agentic", "customer-service"],
    )
    d["taubench_retail"] = DatasetInfo(
        name="taubench_retail",
        family="bench",
        description="TAU-bench — retail domain",
        response_type="continuous",
        n_subjects=6,
        n_items=115,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="taubench_retail.pt",
        url="https://github.com/sierra-research/tau-bench",
        tags=["agentic", "customer-service"],
    )
    d["taubench_telecom"] = DatasetInfo(
        name="taubench_telecom",
        family="bench",
        description="TAU-bench — telecom domain",
        response_type="continuous",
        n_subjects=4,
        n_items=114,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="taubench_telecom.pt",
        url="https://github.com/sierra-research/tau-bench",
        tags=["agentic", "customer-service"],
    )

    # ── Terminal-Bench ────────────────────────────────────────────────
    d["terminalbench"] = DatasetInfo(
        name="terminalbench",
        family="bench",
        description="Terminal-Bench — CLI task solving (majority vote)",
        response_type="binary",
        n_subjects=128,
        n_items=89,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="terminalbench.pt",
        url="https://github.com/terminal-bench/terminal-bench",
        tags=["agentic", "cli"],
    )
    d["terminalbench_resolution"] = DatasetInfo(
        name="terminalbench_resolution",
        family="bench",
        description="Terminal-Bench — CLI task resolution rate",
        response_type="continuous",
        n_subjects=128,
        n_items=89,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="terminalbench_resolution.pt",
        url="https://github.com/terminal-bench/terminal-bench",
        tags=["agentic", "cli"],
    )

    # ── VisualWebArena ────────────────────────────────────────────────
    d["visualwebarena"] = DatasetInfo(
        name="visualwebarena",
        family="bench",
        description="VisualWebArena — multimodal web navigation",
        response_type="continuous",
        n_subjects=6,
        n_items=910,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="visualwebarena.pt",
        url="https://visualwebarena.github.io/",
        tags=["agentic", "web", "multimodal"],
    )

    # ── WebArena ──────────────────────────────────────────────────────
    d["webarena"] = DatasetInfo(
        name="webarena",
        family="bench",
        description="WebArena — autonomous web agent tasks",
        response_type="continuous",
        n_subjects=14,
        n_items=812,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="webarena.pt",
        url="https://webarena.dev/",
        tags=["agentic", "web"],
    )

    # ── AgentBench ─────────────────────────────────────────────────────
    d["agentbench"] = DatasetInfo(
        name="agentbench",
        family="bench",
        description="AgentBench — multi-environment agent evaluation",
        response_type="continuous",
        n_subjects=29,
        n_items=8,
        subject_entity="LLM",
        item_entity="environment",
        repo_id=_BENCH_REPO,
        filename="agentbench.pt",
        url="https://github.com/THUDM/AgentBench",
        tags=["agentic", "multi-domain"],
    )

    # ── AlpacaEval ─────────────────────────────────────────────────────
    d["alpacaeval"] = DatasetInfo(
        name="alpacaeval",
        family="bench",
        description="AlpacaEval — instruction following (win/loss vs GPT-4)",
        response_type="binary",
        n_subjects=221,
        n_items=805,
        subject_entity="LLM",
        item_entity="instruction",
        repo_id=_BENCH_REPO,
        filename="alpacaeval.pt",
        url="https://tatsu-lab.github.io/alpaca_eval/",
        tags=["instruction-following", "nlp"],
    )

    # ── AndroidWorld ───────────────────────────────────────────────────
    d["androidworld"] = DatasetInfo(
        name="androidworld",
        family="bench",
        description="AndroidWorld — mobile device automation tasks",
        response_type="binary",
        n_subjects=3,
        n_items=116,
        subject_entity="agent",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="androidworld.pt",
        url="https://github.com/google-research/android_world",
        tags=["agentic", "mobile"],
    )

    # ── BrowserGym ─────────────────────────────────────────────────────
    d["browsergym"] = DatasetInfo(
        name="browsergym",
        family="bench",
        description="BrowserGym — web agent benchmark aggregates",
        response_type="continuous",
        n_subjects=18,
        n_items=8,
        subject_entity="agent",
        item_entity="benchmark",
        repo_id=_BENCH_REPO,
        filename="browsergym.pt",
        url="https://github.com/ServiceNow/BrowserGym",
        tags=["agentic", "web"],
    )

    # ── GAIA ───────────────────────────────────────────────────────────
    d["gaia"] = DatasetInfo(
        name="gaia",
        family="bench",
        description="GAIA — general AI assistant tasks",
        response_type="binary",
        n_subjects=32,
        n_items=165,
        subject_entity="agent",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="gaia.pt",
        url="https://huggingface.co/gaia-benchmark",
        tags=["agentic", "general-purpose"],
    )

    # ── LiveBench ──────────────────────────────────────────────────────
    d["livebench"] = DatasetInfo(
        name="livebench",
        family="bench",
        description="LiveBench — contamination-free LLM benchmark (scores)",
        response_type="continuous",
        n_subjects=195,
        n_items=494,
        subject_entity="LLM",
        item_entity="question",
        repo_id=_BENCH_REPO,
        filename="livebench.pt",
        url="https://livebench.ai/",
        tags=["nlp", "reasoning", "coding", "math"],
    )
    d["livebench_binary"] = DatasetInfo(
        name="livebench_binary",
        family="bench",
        description="LiveBench — contamination-free LLM benchmark (binary)",
        response_type="binary",
        n_subjects=195,
        n_items=494,
        subject_entity="LLM",
        item_entity="question",
        repo_id=_BENCH_REPO,
        filename="livebench_binary.pt",
        url="https://livebench.ai/",
        tags=["nlp", "reasoning", "coding", "math"],
    )

    # ── MathArena ──────────────────────────────────────────────────────
    d["matharena"] = DatasetInfo(
        name="matharena",
        family="bench",
        description="MathArena — competition math (all competitions combined)",
        response_type="continuous",
        n_subjects=68,
        n_items=336,
        subject_entity="LLM",
        item_entity="problem",
        repo_id=_BENCH_REPO,
        filename="matharena.pt",
        url="https://matharena.ai/",
        tags=["math", "reasoning"],
    )
    d["matharena_aime2025"] = DatasetInfo(
        name="matharena_aime2025",
        family="bench",
        description="MathArena — AIME 2025 (binary pass/fail)",
        response_type="binary",
        n_subjects=62,
        n_items=30,
        subject_entity="LLM",
        item_entity="problem",
        repo_id=_BENCH_REPO,
        filename="matharena_aime2025.pt",
        url="https://matharena.ai/",
        tags=["math", "reasoning"],
    )

    # ── OSWorld ─────────────────────────────────────────────────────────
    d["osworld"] = DatasetInfo(
        name="osworld",
        family="bench",
        description="OSWorld — desktop automation tasks (scores)",
        response_type="continuous",
        n_subjects=77,
        n_items=369,
        subject_entity="agent",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="osworld.pt",
        url="https://os-world.github.io/",
        tags=["agentic", "desktop"],
    )
    d["osworld_binary"] = DatasetInfo(
        name="osworld_binary",
        family="bench",
        description="OSWorld — desktop automation tasks (binary)",
        response_type="binary",
        n_subjects=77,
        n_items=369,
        subject_entity="agent",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="osworld_binary.pt",
        url="https://os-world.github.io/",
        tags=["agentic", "desktop"],
    )

    # ── SWE-bench Full ─────────────────────────────────────────────────
    d["swebench_full"] = DatasetInfo(
        name="swebench_full",
        family="bench",
        description="SWE-bench Full — software engineering (2,294 instances)",
        response_type="binary",
        n_subjects=24,
        n_items=2294,
        subject_entity="agent",
        item_entity="instance",
        repo_id=_BENCH_REPO,
        filename="swebench_full.pt",
        url="https://www.swebench.com/",
        tags=["coding", "agentic", "software-engineering"],
    )

    # ── SWE-bench Multilingual ─────────────────────────────────────────
    d["swebench_multilingual"] = DatasetInfo(
        name="swebench_multilingual",
        family="bench",
        description="SWE-bench Multilingual — multi-language SWE (301 instances)",
        response_type="binary",
        n_subjects=13,
        n_items=301,
        subject_entity="agent",
        item_entity="instance",
        repo_id=_BENCH_REPO,
        filename="swebench_multilingual.pt",
        url="https://github.com/multi-swe-bench/multi-swe-bench-env",
        tags=["coding", "agentic", "software-engineering", "multilingual"],
    )
    d["swebench_multi_all"] = DatasetInfo(
        name="swebench_multi_all",
        family="bench",
        description="SWE-bench Multi — all languages combined (2,132 instances)",
        response_type="binary",
        n_subjects=79,
        n_items=2132,
        subject_entity="agent",
        item_entity="instance",
        repo_id=_BENCH_REPO,
        filename="swebench_multi_all.pt",
        url="https://github.com/multi-swe-bench/multi-swe-bench-env",
        tags=["coding", "agentic", "software-engineering", "multilingual"],
    )

    # ── TheAgentCompany ────────────────────────────────────────────────
    d["theagentcompany"] = DatasetInfo(
        name="theagentcompany",
        family="bench",
        description="TheAgentCompany — enterprise task automation (scores)",
        response_type="continuous",
        n_subjects=19,
        n_items=175,
        subject_entity="agent",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="theagentcompany.pt",
        url="https://github.com/TheAgentCompany/experiments",
        tags=["agentic", "enterprise"],
    )
    d["theagentcompany_binary"] = DatasetInfo(
        name="theagentcompany_binary",
        family="bench",
        description="TheAgentCompany — enterprise task automation (binary)",
        response_type="binary",
        n_subjects=19,
        n_items=175,
        subject_entity="agent",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="theagentcompany_binary.pt",
        url="https://github.com/TheAgentCompany/experiments",
        tags=["agentic", "enterprise"],
    )

    # ── ToolBench ──────────────────────────────────────────────────────
    d["toolbench"] = DatasetInfo(
        name="toolbench",
        family="bench",
        description="StableToolBench — tool-use evaluation",
        response_type="binary",
        n_subjects=10,
        n_items=765,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="toolbench.pt",
        url="https://github.com/THUNLP-MT/StableToolBench",
        tags=["agentic", "tool-use"],
    )

    # ── WildBench ──────────────────────────────────────────────────────
    d["wildbench"] = DatasetInfo(
        name="wildbench",
        family="bench",
        description="WildBench — open-ended LLM evaluation (1-10 scores)",
        response_type="continuous",
        n_subjects=63,
        n_items=1024,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="wildbench.pt",
        url="https://huggingface.co/spaces/allenai/WildBench",
        tags=["nlp", "instruction-following"],
    )

    # ── WorkArena ──────────────────────────────────────────────────────
    d["workarena"] = DatasetInfo(
        name="workarena",
        family="bench",
        description="WorkArena — ServiceNow enterprise web tasks",
        response_type="continuous",
        n_subjects=4,
        n_items=118,
        subject_entity="agent",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="workarena.pt",
        url="https://github.com/ServiceNow/WorkArena",
        tags=["agentic", "enterprise", "web"],
    )

    # ── ARC-AGI ────────────────────────────────────────────────────────
    d["arcagi"] = DatasetInfo(
        name="arcagi",
        family="bench",
        description="ARC-AGI v1 — abstract reasoning (400 public eval tasks)",
        response_type="binary",
        n_subjects=52,
        n_items=400,
        subject_entity="system",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="arcagi.pt",
        url="https://arcprize.org/",
        tags=["reasoning", "abstract"],
    )
    d["arcagi_v2"] = DatasetInfo(
        name="arcagi_v2",
        family="bench",
        description="ARC-AGI v2 — abstract reasoning (120 tasks)",
        response_type="binary",
        n_subjects=28,
        n_items=120,
        subject_entity="system",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="arcagi_v2.pt",
        url="https://arcprize.org/",
        tags=["reasoning", "abstract"],
    )

    # ── Humanity's Last Exam ───────────────────────────────────────────
    d["hle"] = DatasetInfo(
        name="hle",
        family="bench",
        description="Humanity's Last Exam — expert-level questions (1,792 items)",
        response_type="binary",
        n_subjects=19,
        n_items=1792,
        subject_entity="LLM",
        item_entity="question",
        repo_id=_BENCH_REPO,
        filename="hle.pt",
        url="https://lastexam.ai/",
        tags=["reasoning", "expert", "multi-domain"],
    )

    # ── SWE-bench Java ─────────────────────────────────────────────────
    d["swebench_java"] = DatasetInfo(
        name="swebench_java",
        family="bench",
        description="SWE-bench Java — Java issue resolution (170 instances)",
        response_type="binary",
        n_subjects=52,
        n_items=170,
        subject_entity="agent",
        item_entity="instance",
        repo_id=_BENCH_REPO,
        filename="swebench_java.pt",
        url="https://github.com/multi-swe-bench",
        tags=["coding", "agentic", "software-engineering", "java"],
    )

    # ── Global South: SIB-200 ────────────────────────────────────────
    d["sib200"] = DatasetInfo(
        name="sib200",
        family="bench",
        description="SIB-200 — Topic classification in 205 languages (GPT-4, GPT-3.5)",
        response_type="binary",
        n_subjects=2,
        n_items=41820,
        subject_entity="model",
        item_entity="item",
        repo_id=_BENCH_REPO,
        filename="sib200.pt",
        url="https://github.com/dadelani/sib-200",
        tags=["multilingual", "global-south", "topic-classification", "205-languages"],
    )

    # ── Global South: AfriMed-QA ─────────────────────────────────────
    d["afrimedqa"] = DatasetInfo(
        name="afrimedqa",
        family="bench",
        description="AfriMed-QA — Pan-African medical QA across 30 models, 20 specialties",
        response_type="binary",
        n_subjects=30,
        n_items=6910,
        subject_entity="model",
        item_entity="question",
        repo_id=_BENCH_REPO,
        filename="afrimedqa.pt",
        url="https://github.com/intron-innovation/AfriMed-QA",
        tags=["global-south", "africa", "medical", "qa"],
    )

    # ── Global South: Bridging-the-Gap ───────────────────────────────
    d["bridging_gap"] = DatasetInfo(
        name="bridging_gap",
        family="bench",
        description="Bridging-the-Gap — Winogrande in 12 languages (English + 11 African)",
        response_type="binary",
        n_subjects=36,
        n_items=1767,
        subject_entity="model_language",
        item_entity="item",
        repo_id=_BENCH_REPO,
        filename="bridging_gap.pt",
        url="https://github.com/InstituteforDiseaseModeling/Bridging-the-Gap-Low-Resource-African-Languages",
        tags=["global-south", "africa", "multilingual", "commonsense-reasoning"],
    )

    d["bridging_gap_continuous"] = DatasetInfo(
        name="bridging_gap_continuous",
        family="bench",
        description="Bridging-the-Gap — Winogrande continuous scores (mean of 3 runs)",
        response_type="continuous",
        n_subjects=36,
        n_items=1767,
        subject_entity="model_language",
        item_entity="item",
        repo_id=_BENCH_REPO,
        filename="bridging_gap_continuous.pt",
        url="https://github.com/InstituteforDiseaseModeling/Bridging-the-Gap-Low-Resource-African-Languages",
        tags=["global-south", "africa", "multilingual", "commonsense-reasoning"],
    )

    # ── Global South: La Leaderboard ─────────────────────────────────
    d["la_leaderboard"] = DatasetInfo(
        name="la_leaderboard",
        family="bench",
        description="La Leaderboard — 69 models × 108 tasks in Spanish, Catalan, Basque, Galician",
        response_type="continuous",
        n_subjects=69,
        n_items=108,
        subject_entity="model",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="la_leaderboard.pt",
        url="https://huggingface.co/datasets/la-leaderboard/results",
        tags=["global-south", "iberian", "multilingual", "leaderboard"],
    )

    # ── Global South: Portuguese LLM Leaderboard ─────────────────────
    d["pt_leaderboard"] = DatasetInfo(
        name="pt_leaderboard",
        family="bench",
        description="Portuguese LLM Leaderboard — 1,148 models × 10 Portuguese NLP tasks",
        response_type="continuous",
        n_subjects=1148,
        n_items=10,
        subject_entity="model",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="pt_leaderboard.pt",
        url="https://huggingface.co/datasets/eduagarcia-temp/llm_pt_leaderboard_raw_results",
        tags=["global-south", "portuguese", "leaderboard"],
    )

    # ── Global South: Korean LLM Leaderboard ─────────────────────────
    d["ko_leaderboard"] = DatasetInfo(
        name="ko_leaderboard",
        family="bench",
        description="Open Ko-LLM Leaderboard — 1,159 models × 9 Korean benchmark tasks",
        response_type="continuous",
        n_subjects=1159,
        n_items=9,
        subject_entity="model",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="ko_leaderboard.pt",
        url="https://huggingface.co/datasets/open-ko-llm-leaderboard/results",
        tags=["east-asia", "korean", "leaderboard"],
    )

    # ── Global South: Thai LLM Leaderboard ───────────────────────────
    d["thai_leaderboard"] = DatasetInfo(
        name="thai_leaderboard",
        family="bench",
        description="ThaiLLM Leaderboard — 72 models × 19 Thai benchmark tasks",
        response_type="continuous",
        n_subjects=72,
        n_items=19,
        subject_entity="model",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="thai_leaderboard.pt",
        url="https://huggingface.co/datasets/ThaiLLM-Leaderboard/results",
        tags=["southeast-asia", "thai", "leaderboard"],
    )

    # ── Global South: KMMLU (Korean, human baseline) ─────────────────
    d["kmmlu"] = DatasetInfo(
        name="kmmlu",
        family="bench",
        description="KMMLU — Korean MMLU, 35,030 items with per-item human accuracy baseline",
        response_type="continuous",
        n_subjects=1,
        n_items=35030,
        subject_entity="annotator_group",
        item_entity="question",
        repo_id=_BENCH_REPO,
        filename="kmmlu.pt",
        url="https://huggingface.co/datasets/HAERAE-HUB/KMMLU",
        tags=["east-asia", "korean", "knowledge", "human-baseline"],
    )

    # ── Global South: HELM African MMLU+Winogrande ───────────────────
    d["helm_african"] = DatasetInfo(
        name="helm_african",
        family="bench",
        description="HELM African MMLU+Winogrande — 23 models × 33,880 items in 11 African languages",
        response_type="binary",
        n_subjects=23,
        n_items=33880,
        subject_entity="model",
        item_entity="question",
        repo_id=_BENCH_REPO,
        filename="helm_african.pt",
        url="https://crfm.stanford.edu/helm/mmlu-winogrande-afr/latest/",
        tags=["global-south", "africa", "multilingual", "knowledge", "commonsense-reasoning"],
    )

    # ── Global South: HELM ThaiExam ──────────────────────────────────
    d["helm_thaiexam"] = DatasetInfo(
        name="helm_thaiexam",
        family="bench",
        description="HELM ThaiExam — 42 models × 565 Thai exam items",
        response_type="binary",
        n_subjects=42,
        n_items=565,
        subject_entity="model",
        item_entity="question",
        repo_id=_BENCH_REPO,
        filename="helm_thaiexam.pt",
        url="https://crfm.stanford.edu/helm/thaiexam/latest/",
        tags=["southeast-asia", "thai", "knowledge", "exams"],
    )

    # ── Global South: HELM CLEVA (Chinese) ───────────────────────────
    d["helm_cleva"] = DatasetInfo(
        name="helm_cleva",
        family="bench",
        description="HELM CLEVA — 4 models × 5,828 items across 21 Chinese NLP tasks",
        response_type="binary",
        n_subjects=4,
        n_items=5828,
        subject_entity="model",
        item_entity="question",
        repo_id=_BENCH_REPO,
        filename="helm_cleva.pt",
        url="https://crfm.stanford.edu/helm/cleva/latest/",
        tags=["east-asia", "chinese", "multilingual", "knowledge"],
    )

    # ── Global South: OALL Arabic EXAMS ──────────────────────────────
    d["oall_arabic_exams"] = DatasetInfo(
        name="oall_arabic_exams",
        family="bench",
        description="OALL Arabic EXAMS — 144 models × 537 Arabic exam items",
        response_type="binary",
        n_subjects=144,
        n_items=537,
        subject_entity="model",
        item_entity="question",
        repo_id=_BENCH_REPO,
        filename="oall_arabic_exams.pt",
        url="https://huggingface.co/spaces/OALL/Open-Arabic-LLM-Leaderboard",
        tags=["global-south", "mena", "arabic", "exams"],
    )

    # ── Global South: OALL Arabic MMLU ───────────────────────────────
    d["oall_arabic_mmlu"] = DatasetInfo(
        name="oall_arabic_mmlu",
        family="bench",
        description="OALL Arabic MMLU — 142 models × 14,042 Arabic-translated MMLU items",
        response_type="binary",
        n_subjects=142,
        n_items=14042,
        subject_entity="model",
        item_entity="question",
        repo_id=_BENCH_REPO,
        filename="oall_arabic_mmlu.pt",
        url="https://huggingface.co/spaces/OALL/Open-Arabic-LLM-Leaderboard",
        tags=["global-south", "mena", "arabic", "knowledge"],
    )

    # ── Global South: MasakhaNER v2 (sentence-level) ─────────────────
    d["masakhaner_v2"] = DatasetInfo(
        name="masakhaner_v2",
        family="bench",
        description="MasakhaNER v2 — 7 models × 27,559 sentences in 19 African languages",
        response_type="binary",
        n_subjects=7,
        n_items=27559,
        subject_entity="model",
        item_entity="sentence",
        repo_id=_BENCH_REPO,
        filename="masakhaner_v2.pt",
        url="https://github.com/masakhane-io/masakhane-ner",
        tags=["global-south", "africa", "multilingual", "ner"],
    )

    # ── Domain: LawBench (Chinese Legal) ────────────────────────────
    d["lawbench"] = DatasetInfo(
        name="lawbench",
        family="bench",
        description="LawBench — 51 models × 9,000 items across 18 Chinese legal tasks (EMNLP 2024)",
        response_type="continuous",
        n_subjects=51,
        n_items=9000,
        subject_entity="model",
        item_entity="question",
        repo_id=_BENCH_REPO,
        filename="lawbench.pt",
        url="https://github.com/open-compass/LawBench",
        tags=["domain-specific", "legal", "chinese"],
    )

    # ── Domain: LexEval (Chinese Legal) ──────────────────────────────
    d["lexeval"] = DatasetInfo(
        name="lexeval",
        family="bench",
        description="LexEval — 38 models × 14,147 items across 23 Chinese legal tasks (NeurIPS 2024)",
        response_type="continuous",
        n_subjects=38,
        n_items=14147,
        subject_entity="model",
        item_entity="question",
        repo_id=_BENCH_REPO,
        filename="lexeval.pt",
        url="https://github.com/CSHaitao/LexEval",
        tags=["domain-specific", "legal", "chinese"],
    )

    # ── Domain: IgakuQA (Japanese Medical) ───────────────────────────
    d["igakuqa"] = DatasetInfo(
        name="igakuqa",
        family="bench",
        description="IgakuQA — 5 models × 2,000 Japanese medical licensing exam questions",
        response_type="binary",
        n_subjects=5,
        n_items=2000,
        subject_entity="model",
        item_entity="question",
        repo_id=_BENCH_REPO,
        filename="igakuqa.pt",
        url="https://github.com/jungokasai/IgakuQA",
        tags=["domain-specific", "medical", "japanese"],
    )

    return d

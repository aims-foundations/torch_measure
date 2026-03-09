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
under the ``bench/`` prefix (e.g. ``bench/swebench.pt``).
"""

from __future__ import annotations

from torch_measure.datasets._info import DatasetInfo

_BENCH_REPO = "aims-foundation/torch-measure-data"


def _register_bench_datasets() -> dict[str, DatasetInfo]:
    """Return registry entries for info-gathering benchmark response matrices."""
    d: dict[str, DatasetInfo] = {}

    # ── AgentDojo ──────────────────────────────────────────────────────
    d["bench/agentdojo"] = DatasetInfo(
        name="bench/agentdojo",
        family="bench",
        description="AgentDojo — tool-use agent utility tasks",
        response_type="binary",
        n_subjects=29,
        n_items=132,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/agentdojo.pt",
        url="https://agentdojo.spylab.ai/",
        tags=["agentic", "tool-use", "security"],
    )
    d["bench/agentdojo_security"] = DatasetInfo(
        name="bench/agentdojo_security",
        family="bench",
        description="AgentDojo — security evaluation",
        response_type="binary",
        n_subjects=28,
        n_items=949,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/agentdojo_security.pt",
        url="https://agentdojo.spylab.ai/",
        tags=["agentic", "tool-use", "security"],
    )
    d["bench/agentdojo_utility_attack"] = DatasetInfo(
        name="bench/agentdojo_utility_attack",
        family="bench",
        description="AgentDojo — utility under attack",
        response_type="binary",
        n_subjects=28,
        n_items=97,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/agentdojo_utility_attack.pt",
        url="https://agentdojo.spylab.ai/",
        tags=["agentic", "tool-use", "security"],
    )

    # ── AppWorld ───────────────────────────────────────────────────────
    d["bench/appworld"] = DatasetInfo(
        name="bench/appworld",
        family="bench",
        description="AppWorld — multi-app interaction tasks",
        response_type="continuous",
        n_subjects=18,
        n_items=24,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/appworld.pt",
        url="https://appworld.dev/",
        tags=["agentic", "app-interaction"],
    )

    # ── BFCL ──────────────────────────────────────────────────────────
    d["bench/bfcl"] = DatasetInfo(
        name="bench/bfcl",
        family="bench",
        description="BFCL v3 — function calling (pass/fail)",
        response_type="binary",
        n_subjects=93,
        n_items=4751,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/bfcl.pt",
        url="https://gorilla.cs.berkeley.edu/leaderboard.html",
        tags=["coding", "function-calling"],
    )

    # ── BigCodeBench ──────────────────────────────────────────────────
    d["bench/bigcodebench"] = DatasetInfo(
        name="bench/bigcodebench",
        family="bench",
        description="BigCodeBench — complete split",
        response_type="binary",
        n_subjects=153,
        n_items=1140,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/bigcodebench.pt",
        url="https://bigcode-bench.github.io/",
        tags=["coding"],
    )
    d["bench/bigcodebench_instruct"] = DatasetInfo(
        name="bench/bigcodebench_instruct",
        family="bench",
        description="BigCodeBench — instruct split",
        response_type="binary",
        n_subjects=126,
        n_items=1140,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/bigcodebench_instruct.pt",
        url="https://bigcode-bench.github.io/",
        tags=["coding"],
    )
    d["bench/bigcodebench_hard_complete"] = DatasetInfo(
        name="bench/bigcodebench_hard_complete",
        family="bench",
        description="BigCodeBench — hard tasks, complete",
        response_type="binary",
        n_subjects=199,
        n_items=148,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/bigcodebench_hard_complete.pt",
        url="https://bigcode-bench.github.io/",
        tags=["coding"],
    )
    d["bench/bigcodebench_hard_instruct"] = DatasetInfo(
        name="bench/bigcodebench_hard_instruct",
        family="bench",
        description="BigCodeBench — hard tasks, instruct",
        response_type="binary",
        n_subjects=173,
        n_items=148,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/bigcodebench_hard_instruct.pt",
        url="https://bigcode-bench.github.io/",
        tags=["coding"],
    )

    # ── ClineBench ────────────────────────────────────────────────────
    d["bench/clinebench"] = DatasetInfo(
        name="bench/clinebench",
        family="bench",
        description="ClineBench — AI coding agent evaluation",
        response_type="continuous",
        n_subjects=3,
        n_items=12,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/clinebench.pt",
        url="https://github.com/cline/cline",
        tags=["coding", "agentic"],
    )

    # ── CORE-Bench ────────────────────────────────────────────────────
    d["bench/corebench"] = DatasetInfo(
        name="bench/corebench",
        family="bench",
        description="CORE-Bench — computational reproducibility (binary)",
        response_type="binary",
        n_subjects=15,
        n_items=270,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/corebench.pt",
        url="https://github.com/siegelz/core-bench",
        tags=["agentic", "reproducibility"],
    )
    d["bench/corebench_scores"] = DatasetInfo(
        name="bench/corebench_scores",
        family="bench",
        description="CORE-Bench — computational reproducibility (scores)",
        response_type="continuous",
        n_subjects=15,
        n_items=270,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/corebench_scores.pt",
        url="https://github.com/siegelz/core-bench",
        tags=["agentic", "reproducibility"],
    )

    # ── CRUXEval ──────────────────────────────────────────────────────
    d["bench/cruxeval"] = DatasetInfo(
        name="bench/cruxeval",
        family="bench",
        description="CRUXEval — code reasoning (continuous)",
        response_type="continuous",
        n_subjects=38,
        n_items=800,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/cruxeval.pt",
        url="https://github.com/facebookresearch/cruxeval",
        tags=["coding", "reasoning"],
    )
    d["bench/cruxeval_binary"] = DatasetInfo(
        name="bench/cruxeval_binary",
        family="bench",
        description="CRUXEval — code reasoning (binary)",
        response_type="binary",
        n_subjects=38,
        n_items=800,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/cruxeval_binary.pt",
        url="https://github.com/facebookresearch/cruxeval",
        tags=["coding", "reasoning"],
    )
    d["bench/cruxeval_input"] = DatasetInfo(
        name="bench/cruxeval_input",
        family="bench",
        description="CRUXEval — input prediction (continuous)",
        response_type="continuous",
        n_subjects=20,
        n_items=800,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/cruxeval_input.pt",
        url="https://github.com/facebookresearch/cruxeval",
        tags=["coding", "reasoning"],
    )
    d["bench/cruxeval_input_binary"] = DatasetInfo(
        name="bench/cruxeval_input_binary",
        family="bench",
        description="CRUXEval — input prediction (binary)",
        response_type="binary",
        n_subjects=20,
        n_items=800,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/cruxeval_input_binary.pt",
        url="https://github.com/facebookresearch/cruxeval",
        tags=["coding", "reasoning"],
    )
    d["bench/cruxeval_output"] = DatasetInfo(
        name="bench/cruxeval_output",
        family="bench",
        description="CRUXEval — output prediction (continuous)",
        response_type="continuous",
        n_subjects=18,
        n_items=800,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/cruxeval_output.pt",
        url="https://github.com/facebookresearch/cruxeval",
        tags=["coding", "reasoning"],
    )
    d["bench/cruxeval_output_binary"] = DatasetInfo(
        name="bench/cruxeval_output_binary",
        family="bench",
        description="CRUXEval — output prediction (binary)",
        response_type="binary",
        n_subjects=18,
        n_items=800,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/cruxeval_output_binary.pt",
        url="https://github.com/facebookresearch/cruxeval",
        tags=["coding", "reasoning"],
    )

    # ── Cybench ───────────────────────────────────────────────────────
    d["bench/cybench"] = DatasetInfo(
        name="bench/cybench",
        family="bench",
        description="Cybench — CTF cybersecurity tasks (unguided)",
        response_type="binary",
        n_subjects=8,
        n_items=40,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/cybench.pt",
        url="https://cybench.github.io/",
        tags=["agentic", "cybersecurity", "ctf"],
    )
    d["bench/cybench_guided"] = DatasetInfo(
        name="bench/cybench_guided",
        family="bench",
        description="Cybench — CTF tasks (subtask-guided)",
        response_type="binary",
        n_subjects=8,
        n_items=40,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/cybench_guided.pt",
        url="https://cybench.github.io/",
        tags=["agentic", "cybersecurity", "ctf"],
    )
    d["bench/cybench_scores"] = DatasetInfo(
        name="bench/cybench_scores",
        family="bench",
        description="Cybench — CTF tasks (subtask completion scores)",
        response_type="continuous",
        n_subjects=8,
        n_items=40,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/cybench_scores.pt",
        url="https://cybench.github.io/",
        tags=["agentic", "cybersecurity", "ctf"],
    )

    # ── DPAI ──────────────────────────────────────────────────────────
    d["bench/dpai"] = DatasetInfo(
        name="bench/dpai",
        family="bench",
        description="DPAI Arena — total score",
        response_type="continuous",
        n_subjects=9,
        n_items=141,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/dpai.pt",
        url="https://dpaia.dev/",
        tags=["agentic", "coding"],
    )
    d["bench/dpai_blind"] = DatasetInfo(
        name="bench/dpai_blind",
        family="bench",
        description="DPAI Arena — blind evaluation score",
        response_type="continuous",
        n_subjects=9,
        n_items=141,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/dpai_blind.pt",
        url="https://dpaia.dev/",
        tags=["agentic", "coding"],
    )
    d["bench/dpai_informed"] = DatasetInfo(
        name="bench/dpai_informed",
        family="bench",
        description="DPAI Arena — informed evaluation score",
        response_type="continuous",
        n_subjects=9,
        n_items=141,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/dpai_informed.pt",
        url="https://dpaia.dev/",
        tags=["agentic", "coding"],
    )
    d["bench/dpai_binary"] = DatasetInfo(
        name="bench/dpai_binary",
        family="bench",
        description="DPAI Arena — binary pass/fail (50% threshold)",
        response_type="binary",
        n_subjects=9,
        n_items=141,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/dpai_binary.pt",
        url="https://dpaia.dev/",
        tags=["agentic", "coding"],
    )

    # ── EDIT-Bench ────────────────────────────────────────────────────
    d["bench/editbench"] = DatasetInfo(
        name="bench/editbench",
        family="bench",
        description="EDIT-Bench — code editing (continuous scores)",
        response_type="continuous",
        n_subjects=44,
        n_items=540,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/editbench.pt",
        url="https://github.com/waynchi/editbench",
        tags=["coding", "editing"],
    )
    d["bench/editbench_binary"] = DatasetInfo(
        name="bench/editbench_binary",
        family="bench",
        description="EDIT-Bench — code editing (binary)",
        response_type="binary",
        n_subjects=44,
        n_items=540,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/editbench_binary.pt",
        url="https://github.com/waynchi/editbench",
        tags=["coding", "editing"],
    )

    # ── EvalPlus ──────────────────────────────────────────────────────
    d["bench/evalplus"] = DatasetInfo(
        name="bench/evalplus",
        family="bench",
        description="EvalPlus — HumanEval+ and MBPP+ combined",
        response_type="binary",
        n_subjects=31,
        n_items=164,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/evalplus.pt",
        url="https://evalplus.github.io/",
        tags=["coding"],
    )
    d["bench/evalplus_humaneval_base"] = DatasetInfo(
        name="bench/evalplus_humaneval_base",
        family="bench",
        description="EvalPlus — HumanEval base",
        response_type="binary",
        n_subjects=31,
        n_items=164,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/evalplus_humaneval_base.pt",
        url="https://evalplus.github.io/",
        tags=["coding"],
    )
    d["bench/evalplus_humaneval_plus"] = DatasetInfo(
        name="bench/evalplus_humaneval_plus",
        family="bench",
        description="EvalPlus — HumanEval+",
        response_type="binary",
        n_subjects=31,
        n_items=164,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/evalplus_humaneval_plus.pt",
        url="https://evalplus.github.io/",
        tags=["coding"],
    )
    d["bench/evalplus_mbpp_base"] = DatasetInfo(
        name="bench/evalplus_mbpp_base",
        family="bench",
        description="EvalPlus — MBPP base",
        response_type="binary",
        n_subjects=22,
        n_items=378,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/evalplus_mbpp_base.pt",
        url="https://evalplus.github.io/",
        tags=["coding"],
    )
    d["bench/evalplus_mbpp_plus"] = DatasetInfo(
        name="bench/evalplus_mbpp_plus",
        family="bench",
        description="EvalPlus — MBPP+",
        response_type="binary",
        n_subjects=22,
        n_items=378,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/evalplus_mbpp_plus.pt",
        url="https://evalplus.github.io/",
        tags=["coding"],
    )

    # ── LiveCodeBench ─────────────────────────────────────────────────
    d["bench/livecodebench"] = DatasetInfo(
        name="bench/livecodebench",
        family="bench",
        description="LiveCodeBench — competitive programming",
        response_type="continuous",
        n_subjects=72,
        n_items=1055,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/livecodebench.pt",
        url="https://livecodebench.github.io/",
        tags=["coding", "competitive-programming"],
    )

    # ── MLE-bench ─────────────────────────────────────────────────────
    d["bench/mlebench"] = DatasetInfo(
        name="bench/mlebench",
        family="bench",
        description="MLE-bench — ML engineering (continuous scores)",
        response_type="continuous",
        n_subjects=30,
        n_items=75,
        subject_entity="LLM",
        item_entity="competition",
        repo_id=_BENCH_REPO,
        filename="bench/mlebench.pt",
        url="https://github.com/openai/mle-bench",
        tags=["agentic", "machine-learning"],
    )
    d["bench/mlebench_binary"] = DatasetInfo(
        name="bench/mlebench_binary",
        family="bench",
        description="MLE-bench — ML engineering (binary)",
        response_type="binary",
        n_subjects=30,
        n_items=75,
        subject_entity="LLM",
        item_entity="competition",
        repo_id=_BENCH_REPO,
        filename="bench/mlebench_binary.pt",
        url="https://github.com/openai/mle-bench",
        tags=["agentic", "machine-learning"],
    )
    d["bench/mlebench_above_median"] = DatasetInfo(
        name="bench/mlebench_above_median",
        family="bench",
        description="MLE-bench — ML engineering (above-median)",
        response_type="binary",
        n_subjects=30,
        n_items=75,
        subject_entity="LLM",
        item_entity="competition",
        repo_id=_BENCH_REPO,
        filename="bench/mlebench_above_median.pt",
        url="https://github.com/openai/mle-bench",
        tags=["agentic", "machine-learning"],
    )
    d["bench/mlebench_scores"] = DatasetInfo(
        name="bench/mlebench_scores",
        family="bench",
        description="MLE-bench — ML engineering (raw Kaggle scores)",
        response_type="continuous",
        n_subjects=30,
        n_items=75,
        subject_entity="LLM",
        item_entity="competition",
        repo_id=_BENCH_REPO,
        filename="bench/mlebench_scores.pt",
        url="https://github.com/openai/mle-bench",
        tags=["agentic", "machine-learning"],
    )

    # ── MMLU-Pro ──────────────────────────────────────────────────────
    d["bench/mmlupro"] = DatasetInfo(
        name="bench/mmlupro",
        family="bench",
        description="MMLU-Pro — per-question accuracy (12K+ models)",
        response_type="continuous",
        n_subjects=48,
        n_items=12257,
        subject_entity="LLM",
        item_entity="question",
        repo_id=_BENCH_REPO,
        filename="bench/mmlupro.pt",
        url="https://github.com/TIGER-AI-Lab/MMLU-Pro",
        tags=["knowledge", "reasoning"],
    )
    d["bench/mmlupro_category"] = DatasetInfo(
        name="bench/mmlupro_category",
        family="bench",
        description="MMLU-Pro — per-category accuracy",
        response_type="continuous",
        n_subjects=247,
        n_items=14,
        subject_entity="LLM",
        item_entity="category",
        repo_id=_BENCH_REPO,
        filename="bench/mmlupro_category.pt",
        url="https://github.com/TIGER-AI-Lab/MMLU-Pro",
        tags=["knowledge", "reasoning"],
    )

    # ── PaperBench ────────────────────────────────────────────────────
    d["bench/paperbench"] = DatasetInfo(
        name="bench/paperbench",
        family="bench",
        description="PaperBench — paper reproduction evaluation",
        response_type="continuous",
        n_subjects=9,
        n_items=20,
        subject_entity="LLM",
        item_entity="paper",
        repo_id=_BENCH_REPO,
        filename="bench/paperbench.pt",
        url="https://arxiv.org/abs/2504.01848",
        tags=["agentic", "research"],
    )
    d["bench/paperbench_runs"] = DatasetInfo(
        name="bench/paperbench_runs",
        family="bench",
        description="PaperBench — per-run results",
        response_type="continuous",
        n_subjects=7,
        n_items=180,
        subject_entity="LLM",
        item_entity="paper-run",
        repo_id=_BENCH_REPO,
        filename="bench/paperbench_runs.pt",
        url="https://arxiv.org/abs/2504.01848",
        tags=["agentic", "research"],
    )

    # ── SWE-bench ─────────────────────────────────────────────────────
    d["bench/swebench"] = DatasetInfo(
        name="bench/swebench",
        family="bench",
        description="SWE-bench Verified — GitHub issue resolution",
        response_type="binary",
        n_subjects=134,
        n_items=500,
        subject_entity="LLM",
        item_entity="issue",
        repo_id=_BENCH_REPO,
        filename="bench/swebench.pt",
        url="https://www.swebench.com/",
        tags=["coding", "agentic", "github"],
    )

    # ── SWE-PolyBench ─────────────────────────────────────────────────
    d["bench/swepolybench_full"] = DatasetInfo(
        name="bench/swepolybench_full",
        family="bench",
        description="SWE-PolyBench — multilingual SWE (full, 1 model)",
        response_type="binary",
        n_subjects=1,
        n_items=2110,
        subject_entity="LLM",
        item_entity="issue",
        repo_id=_BENCH_REPO,
        filename="bench/swepolybench_full.pt",
        url="https://github.com/amazon-science/SWE-PolyBench",
        tags=["coding", "multilingual"],
    )
    d["bench/swepolybench_verified"] = DatasetInfo(
        name="bench/swepolybench_verified",
        family="bench",
        description="SWE-PolyBench — verified subset",
        response_type="binary",
        n_subjects=3,
        n_items=382,
        subject_entity="LLM",
        item_entity="issue",
        repo_id=_BENCH_REPO,
        filename="bench/swepolybench_verified.pt",
        url="https://github.com/amazon-science/SWE-PolyBench",
        tags=["coding", "multilingual"],
    )

    # ── TAU-bench ─────────────────────────────────────────────────────
    d["bench/taubench"] = DatasetInfo(
        name="bench/taubench",
        family="bench",
        description="TAU-bench — all domains combined",
        response_type="continuous",
        n_subjects=32,
        n_items=329,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/taubench.pt",
        url="https://github.com/sierra-research/tau-bench",
        tags=["agentic", "customer-service"],
    )
    d["bench/taubench_v1_airline"] = DatasetInfo(
        name="bench/taubench_v1_airline",
        family="bench",
        description="TAU-bench — airline domain v1",
        response_type="continuous",
        n_subjects=28,
        n_items=50,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/taubench_v1_airline.pt",
        url="https://github.com/sierra-research/tau-bench",
        tags=["agentic", "customer-service"],
    )
    d["bench/taubench_v2_airline"] = DatasetInfo(
        name="bench/taubench_v2_airline",
        family="bench",
        description="TAU-bench — airline domain v2",
        response_type="continuous",
        n_subjects=4,
        n_items=50,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/taubench_v2_airline.pt",
        url="https://github.com/sierra-research/tau-bench",
        tags=["agentic", "customer-service"],
    )
    d["bench/taubench_hal_airline"] = DatasetInfo(
        name="bench/taubench_hal_airline",
        family="bench",
        description="TAU-bench — HAL airline evaluation",
        response_type="continuous",
        n_subjects=26,
        n_items=50,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/taubench_hal_airline.pt",
        url="https://github.com/sierra-research/tau-bench",
        tags=["agentic", "customer-service"],
    )
    d["bench/taubench_retail"] = DatasetInfo(
        name="bench/taubench_retail",
        family="bench",
        description="TAU-bench — retail domain",
        response_type="continuous",
        n_subjects=6,
        n_items=115,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/taubench_retail.pt",
        url="https://github.com/sierra-research/tau-bench",
        tags=["agentic", "customer-service"],
    )
    d["bench/taubench_telecom"] = DatasetInfo(
        name="bench/taubench_telecom",
        family="bench",
        description="TAU-bench — telecom domain",
        response_type="continuous",
        n_subjects=4,
        n_items=114,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/taubench_telecom.pt",
        url="https://github.com/sierra-research/tau-bench",
        tags=["agentic", "customer-service"],
    )

    # ── Terminal-Bench ────────────────────────────────────────────────
    d["bench/terminalbench"] = DatasetInfo(
        name="bench/terminalbench",
        family="bench",
        description="Terminal-Bench — CLI task solving (majority vote)",
        response_type="binary",
        n_subjects=128,
        n_items=89,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/terminalbench.pt",
        url="https://github.com/terminal-bench/terminal-bench",
        tags=["agentic", "cli"],
    )
    d["bench/terminalbench_resolution"] = DatasetInfo(
        name="bench/terminalbench_resolution",
        family="bench",
        description="Terminal-Bench — CLI task resolution rate",
        response_type="continuous",
        n_subjects=128,
        n_items=89,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/terminalbench_resolution.pt",
        url="https://github.com/terminal-bench/terminal-bench",
        tags=["agentic", "cli"],
    )

    # ── VisualWebArena ────────────────────────────────────────────────
    d["bench/visualwebarena"] = DatasetInfo(
        name="bench/visualwebarena",
        family="bench",
        description="VisualWebArena — multimodal web navigation",
        response_type="continuous",
        n_subjects=6,
        n_items=910,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/visualwebarena.pt",
        url="https://visualwebarena.github.io/",
        tags=["agentic", "web", "multimodal"],
    )

    # ── WebArena ──────────────────────────────────────────────────────
    d["bench/webarena"] = DatasetInfo(
        name="bench/webarena",
        family="bench",
        description="WebArena — autonomous web agent tasks",
        response_type="continuous",
        n_subjects=14,
        n_items=812,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/webarena.pt",
        url="https://webarena.dev/",
        tags=["agentic", "web"],
    )

    # ── AgentBench ─────────────────────────────────────────────────────
    d["bench/agentbench"] = DatasetInfo(
        name="bench/agentbench",
        family="bench",
        description="AgentBench — multi-environment agent evaluation",
        response_type="continuous",
        n_subjects=29,
        n_items=8,
        subject_entity="LLM",
        item_entity="environment",
        repo_id=_BENCH_REPO,
        filename="bench/agentbench.pt",
        url="https://github.com/THUDM/AgentBench",
        tags=["agentic", "multi-domain"],
    )

    # ── AlpacaEval ─────────────────────────────────────────────────────
    d["bench/alpacaeval"] = DatasetInfo(
        name="bench/alpacaeval",
        family="bench",
        description="AlpacaEval — instruction following (win/loss vs GPT-4)",
        response_type="binary",
        n_subjects=221,
        n_items=805,
        subject_entity="LLM",
        item_entity="instruction",
        repo_id=_BENCH_REPO,
        filename="bench/alpacaeval.pt",
        url="https://tatsu-lab.github.io/alpaca_eval/",
        tags=["instruction-following", "nlp"],
    )

    # ── AndroidWorld ───────────────────────────────────────────────────
    d["bench/androidworld"] = DatasetInfo(
        name="bench/androidworld",
        family="bench",
        description="AndroidWorld — mobile device automation tasks",
        response_type="binary",
        n_subjects=3,
        n_items=116,
        subject_entity="agent",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/androidworld.pt",
        url="https://github.com/google-research/android_world",
        tags=["agentic", "mobile"],
    )

    # ── BrowserGym ─────────────────────────────────────────────────────
    d["bench/browsergym"] = DatasetInfo(
        name="bench/browsergym",
        family="bench",
        description="BrowserGym — web agent benchmark aggregates",
        response_type="continuous",
        n_subjects=18,
        n_items=8,
        subject_entity="agent",
        item_entity="benchmark",
        repo_id=_BENCH_REPO,
        filename="bench/browsergym.pt",
        url="https://github.com/ServiceNow/BrowserGym",
        tags=["agentic", "web"],
    )

    # ── GAIA ───────────────────────────────────────────────────────────
    d["bench/gaia"] = DatasetInfo(
        name="bench/gaia",
        family="bench",
        description="GAIA — general AI assistant tasks",
        response_type="binary",
        n_subjects=32,
        n_items=165,
        subject_entity="agent",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/gaia.pt",
        url="https://huggingface.co/gaia-benchmark",
        tags=["agentic", "general-purpose"],
    )

    # ── LiveBench ──────────────────────────────────────────────────────
    d["bench/livebench"] = DatasetInfo(
        name="bench/livebench",
        family="bench",
        description="LiveBench — contamination-free LLM benchmark (scores)",
        response_type="continuous",
        n_subjects=195,
        n_items=494,
        subject_entity="LLM",
        item_entity="question",
        repo_id=_BENCH_REPO,
        filename="bench/livebench.pt",
        url="https://livebench.ai/",
        tags=["nlp", "reasoning", "coding", "math"],
    )
    d["bench/livebench_binary"] = DatasetInfo(
        name="bench/livebench_binary",
        family="bench",
        description="LiveBench — contamination-free LLM benchmark (binary)",
        response_type="binary",
        n_subjects=195,
        n_items=494,
        subject_entity="LLM",
        item_entity="question",
        repo_id=_BENCH_REPO,
        filename="bench/livebench_binary.pt",
        url="https://livebench.ai/",
        tags=["nlp", "reasoning", "coding", "math"],
    )

    # ── MathArena ──────────────────────────────────────────────────────
    d["bench/matharena"] = DatasetInfo(
        name="bench/matharena",
        family="bench",
        description="MathArena — competition math (all competitions combined)",
        response_type="continuous",
        n_subjects=68,
        n_items=336,
        subject_entity="LLM",
        item_entity="problem",
        repo_id=_BENCH_REPO,
        filename="bench/matharena.pt",
        url="https://matharena.ai/",
        tags=["math", "reasoning"],
    )
    d["bench/matharena_aime2025"] = DatasetInfo(
        name="bench/matharena_aime2025",
        family="bench",
        description="MathArena — AIME 2025 (binary pass/fail)",
        response_type="binary",
        n_subjects=62,
        n_items=30,
        subject_entity="LLM",
        item_entity="problem",
        repo_id=_BENCH_REPO,
        filename="bench/matharena_aime2025.pt",
        url="https://matharena.ai/",
        tags=["math", "reasoning"],
    )

    # ── OSWorld ─────────────────────────────────────────────────────────
    d["bench/osworld"] = DatasetInfo(
        name="bench/osworld",
        family="bench",
        description="OSWorld — desktop automation tasks (scores)",
        response_type="continuous",
        n_subjects=77,
        n_items=369,
        subject_entity="agent",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/osworld.pt",
        url="https://os-world.github.io/",
        tags=["agentic", "desktop"],
    )
    d["bench/osworld_binary"] = DatasetInfo(
        name="bench/osworld_binary",
        family="bench",
        description="OSWorld — desktop automation tasks (binary)",
        response_type="binary",
        n_subjects=77,
        n_items=369,
        subject_entity="agent",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/osworld_binary.pt",
        url="https://os-world.github.io/",
        tags=["agentic", "desktop"],
    )

    # ── SWE-bench Full ─────────────────────────────────────────────────
    d["bench/swebench_full"] = DatasetInfo(
        name="bench/swebench_full",
        family="bench",
        description="SWE-bench Full — software engineering (2,294 instances)",
        response_type="binary",
        n_subjects=24,
        n_items=2294,
        subject_entity="agent",
        item_entity="instance",
        repo_id=_BENCH_REPO,
        filename="bench/swebench_full.pt",
        url="https://www.swebench.com/",
        tags=["coding", "agentic", "software-engineering"],
    )

    # ── SWE-bench Multilingual ─────────────────────────────────────────
    d["bench/swebench_multilingual"] = DatasetInfo(
        name="bench/swebench_multilingual",
        family="bench",
        description="SWE-bench Multilingual — multi-language SWE (301 instances)",
        response_type="binary",
        n_subjects=13,
        n_items=301,
        subject_entity="agent",
        item_entity="instance",
        repo_id=_BENCH_REPO,
        filename="bench/swebench_multilingual.pt",
        url="https://github.com/multi-swe-bench/multi-swe-bench-env",
        tags=["coding", "agentic", "software-engineering", "multilingual"],
    )
    d["bench/swebench_multi_all"] = DatasetInfo(
        name="bench/swebench_multi_all",
        family="bench",
        description="SWE-bench Multi — all languages combined (2,132 instances)",
        response_type="binary",
        n_subjects=79,
        n_items=2132,
        subject_entity="agent",
        item_entity="instance",
        repo_id=_BENCH_REPO,
        filename="bench/swebench_multi_all.pt",
        url="https://github.com/multi-swe-bench/multi-swe-bench-env",
        tags=["coding", "agentic", "software-engineering", "multilingual"],
    )

    # ── TheAgentCompany ────────────────────────────────────────────────
    d["bench/theagentcompany"] = DatasetInfo(
        name="bench/theagentcompany",
        family="bench",
        description="TheAgentCompany — enterprise task automation (scores)",
        response_type="continuous",
        n_subjects=19,
        n_items=175,
        subject_entity="agent",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/theagentcompany.pt",
        url="https://github.com/TheAgentCompany/experiments",
        tags=["agentic", "enterprise"],
    )
    d["bench/theagentcompany_binary"] = DatasetInfo(
        name="bench/theagentcompany_binary",
        family="bench",
        description="TheAgentCompany — enterprise task automation (binary)",
        response_type="binary",
        n_subjects=19,
        n_items=175,
        subject_entity="agent",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/theagentcompany_binary.pt",
        url="https://github.com/TheAgentCompany/experiments",
        tags=["agentic", "enterprise"],
    )

    # ── ToolBench ──────────────────────────────────────────────────────
    d["bench/toolbench"] = DatasetInfo(
        name="bench/toolbench",
        family="bench",
        description="StableToolBench — tool-use evaluation",
        response_type="binary",
        n_subjects=10,
        n_items=765,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/toolbench.pt",
        url="https://github.com/THUNLP-MT/StableToolBench",
        tags=["agentic", "tool-use"],
    )

    # ── WildBench ──────────────────────────────────────────────────────
    d["bench/wildbench"] = DatasetInfo(
        name="bench/wildbench",
        family="bench",
        description="WildBench — open-ended LLM evaluation (1-10 scores)",
        response_type="continuous",
        n_subjects=63,
        n_items=1024,
        subject_entity="LLM",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/wildbench.pt",
        url="https://huggingface.co/spaces/allenai/WildBench",
        tags=["nlp", "instruction-following"],
    )

    # ── WorkArena ──────────────────────────────────────────────────────
    d["bench/workarena"] = DatasetInfo(
        name="bench/workarena",
        family="bench",
        description="WorkArena — ServiceNow enterprise web tasks",
        response_type="continuous",
        n_subjects=4,
        n_items=118,
        subject_entity="agent",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/workarena.pt",
        url="https://github.com/ServiceNow/WorkArena",
        tags=["agentic", "enterprise", "web"],
    )

    # ── ARC-AGI ────────────────────────────────────────────────────────
    d["bench/arcagi"] = DatasetInfo(
        name="bench/arcagi",
        family="bench",
        description="ARC-AGI v1 — abstract reasoning (400 public eval tasks)",
        response_type="binary",
        n_subjects=52,
        n_items=400,
        subject_entity="system",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/arcagi.pt",
        url="https://arcprize.org/",
        tags=["reasoning", "abstract"],
    )
    d["bench/arcagi_v2"] = DatasetInfo(
        name="bench/arcagi_v2",
        family="bench",
        description="ARC-AGI v2 — abstract reasoning (120 tasks)",
        response_type="binary",
        n_subjects=28,
        n_items=120,
        subject_entity="system",
        item_entity="task",
        repo_id=_BENCH_REPO,
        filename="bench/arcagi_v2.pt",
        url="https://arcprize.org/",
        tags=["reasoning", "abstract"],
    )

    # ── Humanity's Last Exam ───────────────────────────────────────────
    d["bench/hle"] = DatasetInfo(
        name="bench/hle",
        family="bench",
        description="Humanity's Last Exam — expert-level questions (1,792 items)",
        response_type="binary",
        n_subjects=19,
        n_items=1792,
        subject_entity="LLM",
        item_entity="question",
        repo_id=_BENCH_REPO,
        filename="bench/hle.pt",
        url="https://lastexam.ai/",
        tags=["reasoning", "expert", "multi-domain"],
    )

    # ── SWE-bench Java ─────────────────────────────────────────────────
    d["bench/swebench_java"] = DatasetInfo(
        name="bench/swebench_java",
        family="bench",
        description="SWE-bench Java — Java issue resolution (170 instances)",
        response_type="binary",
        n_subjects=52,
        n_items=170,
        subject_entity="agent",
        item_entity="instance",
        repo_id=_BENCH_REPO,
        filename="bench/swebench_java.pt",
        url="https://github.com/multi-swe-bench",
        tags=["coding", "agentic", "software-engineering", "java"],
    )

    return d

# Comprehensive Catalog of Coding & Software Engineering Agent Benchmarks

Compiled: 2026-02-28

---

## Table of Contents
1. [Classic Code Generation Benchmarks](#1-classic-code-generation-benchmarks)
2. [Enhanced / Extended Code Generation Benchmarks](#2-enhanced--extended-code-generation-benchmarks)
3. [Competitive Programming Benchmarks](#3-competitive-programming-benchmarks)
4. [Multilingual Code Benchmarks](#4-multilingual-code-benchmarks)
5. [Repository-Level & Code Completion Benchmarks](#5-repository-level--code-completion-benchmarks)
6. [Software Engineering Agent Benchmarks (SWE-bench Family)](#6-software-engineering-agent-benchmarks-swe-bench-family)
7. [Code Editing & Debugging Benchmarks](#7-code-editing--debugging-benchmarks)
8. [Code Review, Commit & Documentation Benchmarks](#8-code-review-commit--documentation-benchmarks)
9. [Security & Vulnerability Benchmarks](#9-security--vulnerability-benchmarks)
10. [Data Science & Specialized Domain Benchmarks](#10-data-science--specialized-domain-benchmarks)
11. [Tool Use & Function Calling Benchmarks](#11-tool-use--function-calling-benchmarks)
12. [Web Agent Benchmarks](#12-web-agent-benchmarks)
13. [OS / Desktop / Mobile Agent Benchmarks](#13-os--desktop--mobile-agent-benchmarks)
14. [Interactive Coding Agent Benchmarks](#14-interactive-coding-agent-benchmarks)
15. [Research Reproduction & Scientific Agent Benchmarks](#15-research-reproduction--scientific-agent-benchmarks)
16. [Comprehensive Evaluation Platforms & Leaderboards](#16-comprehensive-evaluation-platforms--leaderboards)
17. [Coding Arena / Human Preference Benchmarks](#17-coding-arena--human-preference-benchmarks)

---

## 1. Classic Code Generation Benchmarks

### HumanEval
- **What it evaluates**: Function-level Python code generation from docstrings
- **Tasks**: 164 problems with unit tests
- **Models evaluated**: 100+ (widely adopted across all major LLMs)
- **Per-task data available**: Yes -- GitHub repo has test cases; many papers report pass@k per problem
- **Links**: [GitHub](https://github.com/openai/human-eval) | [Paper](https://arxiv.org/abs/2107.03374) | [HF Dataset](https://huggingface.co/datasets/openai_humaneval)

### MBPP (Mostly Basic Python Programming)
- **What it evaluates**: Basic Python programming from natural language descriptions
- **Tasks**: 974 problems (427 in sanitized subset)
- **Models evaluated**: 50+ across various papers
- **Per-task data available**: Yes -- dataset is public, many evaluations share per-problem pass rates
- **Links**: [Paper](https://arxiv.org/abs/2108.07732) | [HF Dataset](https://huggingface.co/datasets/google-research-datasets/mbpp)

### APPS (Automated Programming Progress Standard)
- **What it evaluates**: Code generation at introductory/interview/competition difficulty levels
- **Tasks**: 10,000 problems (5,000 train, 5,000 test) across 3 difficulty levels
- **Models evaluated**: 20+ in original; widely used since
- **Per-task data available**: Yes -- public dataset with test cases
- **Links**: [Paper](https://arxiv.org/abs/2105.09938) | [GitHub](https://github.com/hendrycks/apps) | [HF Dataset](https://huggingface.co/datasets/codeparrot/apps)

### CodeContests (AlphaCode)
- **What it evaluates**: Competition-level code generation (Codeforces-style)
- **Tasks**: ~13,000 problems (train+test) with extensive test cases
- **Models evaluated**: AlphaCode, AlphaCode2, and various LLMs in subsequent studies
- **Per-task data available**: Yes -- public dataset on HuggingFace/GitHub
- **Links**: [GitHub](https://github.com/google-deepmind/code_contests) | [Paper](https://arxiv.org/abs/2203.07814) | [HF Dataset](https://huggingface.co/datasets/deepmind/code_contests)

---

## 2. Enhanced / Extended Code Generation Benchmarks

### EvalPlus (HumanEval+ / MBPP+)
- **What it evaluates**: Rigorous functional correctness with 80x (HumanEval) / 35x (MBPP) more test cases
- **Tasks**: 164 (HumanEval+) + 399 (MBPP+)
- **Models evaluated**: 100+ models on leaderboard
- **Per-task data available**: Yes -- leaderboard provides per-model scores; harness is open
- **Links**: [GitHub](https://github.com/evalplus/evalplus) | [Leaderboard](https://evalplus.github.io/leaderboard.html) | [Paper (NeurIPS 2023)](https://arxiv.org/abs/2305.01210)

### BigCodeBench
- **What it evaluates**: Complex function-level code generation using 139 library APIs
- **Tasks**: 1,140 tasks with avg 5.6 test cases each, 99% branch coverage
- **Models evaluated**: 60+ LLMs
- **Per-task data available**: Yes -- leaderboard + public dataset
- **Links**: [GitHub](https://github.com/bigcode-project/bigcodebench) | [Leaderboard](https://huggingface.co/spaces/bigcode/bigcodebench-leaderboard) | [Paper (ICLR 2025)](https://openreview.net/forum?id=YrycTjllL0) | [HF Dataset](https://huggingface.co/datasets/bigcode/bigcodebench)

### HumanEval Pro / MBPP Pro / BigCodeBench-Lite Pro
- **What it evaluates**: Self-invoking code generation (solve base problem, use its solution for harder problem)
- **Tasks**: Extended versions of HumanEval, MBPP, and BigCodeBench-Lite
- **Models evaluated**: 20+ including o1-mini, GPT-4o, Qwen2.5-Coder
- **Per-task data available**: Yes -- GitHub repo with evaluation harness
- **Links**: [GitHub](https://github.com/CodeEval-Pro/CodeEval-Pro) | [Paper (ACL 2025 Findings)](https://arxiv.org/abs/2412.21199)

### HumanEvalPack (OctoPack)
- **What it evaluates**: Code generation, fixing, and explanation across 6 languages
- **Tasks**: 164 x 3 tasks x 6 languages (Python, JS, Java, Go, C++, Rust)
- **Models evaluated**: Multiple in OctoPack study
- **Per-task data available**: Yes -- HuggingFace dataset
- **Links**: [HF Dataset](https://huggingface.co/datasets/bigcode/humanevalpack) | [GitHub](https://github.com/bigcode-project/octopack) | [Paper](https://arxiv.org/abs/2308.07124)

### AutoCodeBench
- **What it evaluates**: Automated code generation benchmark across 20 programming languages
- **Tasks**: 3,920 problems
- **Models evaluated**: Multiple LLMs
- **Per-task data available**: Dataset available
- **Links**: [Paper](https://arxiv.org/abs/2508.09101)

### NaturalCodeBench (NCB)
- **What it evaluates**: Application-driven code synthesis aligned with real-world usage
- **Tasks**: 402 Python and Java problems across 6 domains
- **Models evaluated**: Multiple LLMs
- **Per-task data available**: Yes
- **Links**: [Paper](https://arxiv.org/abs/2405.04520)

### FullStackBench
- **What it evaluates**: Full-stack programming across 16 languages, covering data analysis, ML, web dev, etc.
- **Tasks**: Multi-domain tasks across 16 programming languages
- **Models evaluated**: Multiple LLMs
- **Per-task data available**: Yes -- HuggingFace dataset
- **Links**: [GitHub](https://github.com/bytedance/FullStackBench) | [HF Dataset](https://huggingface.co/datasets/ByteDance/FullStackBench) | [Paper](https://arxiv.org/abs/2412.00535)

### ComplexCodeEval
- **What it evaluates**: Code generation, completion, API recommendation, and test case generation
- **Tasks**: 3,897 Java + 7,184 Python samples from high-star GitHub repos
- **Models evaluated**: Multiple LCMs
- **Per-task data available**: Yes -- GitHub repo
- **Links**: [GitHub](https://github.com/ComplexCodeEval/ComplexCodeEval) | [Paper](https://arxiv.org/abs/2409.10280)

### DomainEval
- **What it evaluates**: Multi-domain code generation (computation, network, system, visualization, cryptography)
- **Tasks**: 2,454 subjects with 5,892 test cases across 6 domains
- **Models evaluated**: Multiple LLMs
- **Per-task data available**: Yes
- **Links**: [Paper](https://arxiv.org/abs/2408.13204)

### DevEval
- **What it evaluates**: Manually-annotated code generation aligned with real-world repos
- **Tasks**: Multiple manually annotated tasks
- **Models evaluated**: Several LLMs
- **Per-task data available**: Yes
- **Links**: [Paper](https://aclanthology.org/2024.findings-acl.214)

### ClassEval
- **What it evaluates**: Class-level code generation (multiple interdependent methods)
- **Tasks**: Class-level coding tasks
- **Models evaluated**: 11 LLMs including GPT-4, GPT-3.5, WizardCoder
- **Per-task data available**: Yes -- public benchmark
- **Links**: [Paper](https://arxiv.org/abs/2308.01861)

### CoderEval
- **What it evaluates**: Code generation with 6 levels of contextual dependency
- **Tasks**: Leveled tasks from standalone to deeply context-dependent
- **Models evaluated**: Multiple LLMs
- **Per-task data available**: Yes
- **Links**: [Paper](https://arxiv.org/abs/2302.00288)

---

## 3. Competitive Programming Benchmarks

### LiveCodeBench
- **What it evaluates**: Code generation, self-repair, execution prediction, and test output prediction (contamination-free)
- **Tasks**: 1,055 problems (release_v6) from LeetCode, AtCoder, CodeForces (continuously updated)
- **Models evaluated**: 50+ models on leaderboard
- **Per-task data available**: Yes -- leaderboard + public dataset
- **Links**: [Website](https://livecodebench.github.io/) | [Leaderboard](https://livecodebench.github.io/leaderboard.html) | [GitHub](https://github.com/LiveCodeBench/LiveCodeBench) | [Paper](https://arxiv.org/abs/2403.07974)

### CodeElo
- **What it evaluates**: Competition-style programming using Elo rating system (Codeforces-based)
- **Tasks**: Competitive programming problems from Codeforces
- **Models evaluated**: Multiple LLMs ranked via Elo
- **Per-task data available**: Yes -- based on Codeforces platform
- **Links**: [Website](https://codeelo-bench.github.io/)

### CodeClash
- **What it evaluates**: Goal-oriented software engineering via multi-round tournaments (BattleSnake, Poker, RoboCode)
- **Tasks**: 1,680 tournaments across 3 arenas
- **Models evaluated**: 8 LLMs (Claude Sonnet 4.5, GPT-5, Gemini 2.5 Pro, Qwen3-Coder, Grok Code Fast, etc.)
- **Per-task data available**: Yes -- tournament results per model
- **Links**: [Website](https://codeclash.ai/) | [GitHub](https://github.com/CodeClash-ai/CodeClash) | [Paper](https://arxiv.org/abs/2511.00839)

### LiveCodeBench Pro
- **What it evaluates**: Extended version of LiveCodeBench
- **Links**: [Website](https://livecodebenchpro.com/)

---

## 4. Multilingual Code Benchmarks

### MultiPL-E
- **What it evaluates**: Multilingual code generation (translation of HumanEval/MBPP to many languages)
- **Tasks**: HumanEval + MBPP translated to 18+ programming languages
- **Models evaluated**: Tracked on Big Code Models Leaderboard
- **Per-task data available**: Yes -- GitHub repo + HuggingFace
- **Links**: [GitHub](https://github.com/nuprl/MultiPL-E) | [Paper](https://arxiv.org/abs/2208.08227)

### McEval
- **What it evaluates**: Multilingual code generation, explanation, and completion
- **Tasks**: 16,000 samples across 40 programming languages
- **Models evaluated**: Multiple LLMs
- **Per-task data available**: Yes -- GitHub repo
- **Links**: [Website](https://mceval.github.io/) | [GitHub](https://github.com/MCEVAL/McEval) | [Paper](https://arxiv.org/abs/2406.07436)

### xCodeEval
- **What it evaluates**: Multilingual multitask (code understanding, generation, translation, retrieval)
- **Tasks**: 25M examples from 7,500 unique problems across 17 languages
- **Models evaluated**: Multiple code LMs
- **Per-task data available**: Yes -- GitHub + execution engine
- **Links**: [GitHub](https://github.com/ntunlp/xCodeEval) | [Paper](https://arxiv.org/abs/2303.03004)

### HumanEval-XL
- **What it evaluates**: Cross-lingual code generation (23 natural languages x 12 programming languages)
- **Tasks**: 22,080 prompts with avg 8.33 test cases
- **Models evaluated**: Multiple LLMs
- **Per-task data available**: Yes
- **Links**: [Paper](https://arxiv.org/abs/2402.16694)

### CanAICode
- **What it evaluates**: Real-world interview-style coding across multiple difficulty levels
- **Tasks**: Multi-level coding problems
- **Models evaluated**: Multiple on leaderboard
- **Per-task data available**: Yes -- leaderboard
- **Links**: [Leaderboard](https://huggingface.co/spaces/mike-ravkine/can-ai-code-results)

---

## 5. Repository-Level & Code Completion Benchmarks

### RepoBench
- **What it evaluates**: Repository-level code auto-completion (Retrieval + Completion + Pipeline)
- **Tasks**: 3 sub-tasks in Python and Java
- **Models evaluated**: Multiple code completion models
- **Per-task data available**: Yes -- GitHub repo
- **Links**: [GitHub](https://github.com/Leolty/repobench) | [Paper (ICLR 2024)](https://arxiv.org/abs/2306.03091)

### CrossCodeEval
- **What it evaluates**: Cross-file context code completion in Python, Java, TypeScript, C#
- **Tasks**: Instances requiring cross-file context
- **Models evaluated**: Multiple LLMs
- **Per-task data available**: Yes
- **Links**: [Paper](https://aclanthology.org/2024.findings-acl.214)

### ExecRepoBench
- **What it evaluates**: Multi-level executable code completion at repository level
- **Tasks**: 1,200 samples from 50 active Python repos with unit tests
- **Models evaluated**: Multiple LLMs
- **Per-task data available**: Yes
- **Links**: [Website](https://execrepobench.github.io/) | [Paper](https://arxiv.org/abs/2412.11990)

### RepoMasterEval
- **What it evaluates**: Code completion via real-world repositories (single line, block, function level)
- **Tasks**: Multi-level tasks from real repos with mutation testing
- **Models evaluated**: Multiple code completion models
- **Per-task data available**: Yes
- **Links**: [Paper](https://arxiv.org/abs/2408.03519)

### RepoEval
- **What it evaluates**: Repository-level code completion
- **Tasks**: Code completion tasks from repositories
- **Models evaluated**: Multiple models
- **Per-task data available**: Yes
- **Links**: Referenced in multiple code completion studies

### DevBench
- **What it evaluates**: Realistic code completion from developer telemetry
- **Tasks**: 1,800 instances across 6 languages and 6 task categories
- **Models evaluated**: Multiple LLMs
- **Per-task data available**: Yes
- **Links**: [Paper](https://arxiv.org/abs/2403.08604) | [OpenReview](https://openreview.net/forum?id=P9RZQ24j1z)

### LoCoBench
- **What it evaluates**: Long-context LLMs in complex software engineering (10K-1M token contexts)
- **Tasks**: 8,000 evaluation scenarios across 10 programming languages
- **Models evaluated**: Multiple LLMs
- **Per-task data available**: Yes -- 17 metrics across 4 dimensions
- **Links**: [Paper](https://arxiv.org/abs/2509.09614)

---

## 6. Software Engineering Agent Benchmarks (SWE-bench Family)

### SWE-bench (Original)
- **What it evaluates**: Resolving real-world GitHub issues with code patches
- **Tasks**: 2,294 task instances from 12 Python repos
- **Models evaluated**: 50+ agents/models on leaderboard
- **Per-task data available**: Yes -- full leaderboard with per-instance results
- **Links**: [Website](https://www.swebench.com/) | [GitHub](https://github.com/SWE-bench/SWE-bench) | [Paper](https://arxiv.org/abs/2310.06770)

### SWE-bench Verified
- **What it evaluates**: Human-validated subset of SWE-bench
- **Tasks**: 500 verified-solvable problems
- **Models evaluated**: 30+ on leaderboard
- **Per-task data available**: Yes
- **Links**: [Leaderboard](https://www.swebench.com/) | [OpenAI Blog](https://openai.com/index/introducing-swe-bench-verified/)

### SWE-bench Lite
- **What it evaluates**: Lightweight subset of SWE-bench
- **Tasks**: 300 problems (easier subset)
- **Models evaluated**: 30+
- **Per-task data available**: Yes
- **Links**: [Leaderboard](https://www.swebench.com/)

### SWE-bench-Live
- **What it evaluates**: Continuously updated SWE-bench with fresh GitHub issues (monthly additions)
- **Tasks**: 1,565 task instances from 164 repositories (growing monthly with 50 new tasks/month)
- **Models evaluated**: Multiple agents on leaderboard
- **Per-task data available**: Yes
- **Links**: [Leaderboard](https://swe-bench-live.github.io/) | [Paper](https://arxiv.org/abs/2505.23419)

### SWE-bench Multilingual / Multi-SWE-bench
- **What it evaluates**: SWE-bench extended to 7 non-Python languages (Java, TypeScript, JS, Go, Rust, C, C++)
- **Tasks**: 1,632 expert-curated instances
- **Models evaluated**: Multiple agents
- **Per-task data available**: Yes
- **Links**: [Website](https://www.swebench.com/multilingual.html) | [GitHub](https://github.com/multi-swe-bench/multi-swe-bench)

### SWE-bench Multimodal
- **What it evaluates**: GitHub issues containing visual elements (screenshots, diagrams)
- **Tasks**: 517 issues with visual content
- **Models evaluated**: Multiple multimodal models
- **Per-task data available**: Yes
- **Links**: [Website](https://www.swebench.com/multimodal.html)

### SWE-bench Pro
- **What it evaluates**: Long-horizon, complex software engineering tasks
- **Tasks**: Complex multi-step programming challenges
- **Models evaluated**: On Scale AI leaderboard
- **Per-task data available**: Yes
- **Links**: [Leaderboard](https://scale.com/leaderboard/swe_bench_pro_public) | [Paper](https://arxiv.org/abs/2509.16941)

### SWE-PolyBench
- **What it evaluates**: Multi-language repository-level evaluation (Java, JS, TypeScript, Python)
- **Tasks**: 2,110 instances from 21 repos (+ SWE-PolyBench500 subsample)
- **Models evaluated**: Multiple agents
- **Per-task data available**: Yes -- evaluation harness provided
- **Links**: [Website](https://amazon-science.github.io/SWE-PolyBench/) | [GitHub](https://github.com/amazon-science/SWE-PolyBench) | [Paper](https://arxiv.org/abs/2504.08703)

### SWE-EVO
- **What it evaluates**: Long-horizon software evolution (multi-step modifications across files)
- **Tasks**: 48 evolution tasks from 7 Python projects (avg 21 files modified, 874 tests per instance)
- **Models evaluated**: GPT-5, Claude models via OpenHands
- **Per-task data available**: Yes
- **Links**: [GitHub](https://github.com/SWE-EVO/SWE-EVO) | [Paper](https://arxiv.org/abs/2512.18470)

### SWT-Bench
- **What it evaluates**: Automated test generation, repair, and execution (not patch creation)
- **Tasks**: Real project test suites
- **Models evaluated**: Multiple agents
- **Per-task data available**: Yes
- **Links**: Released October 2024 by LogicStar AI

### SWE-Gym
- **What it evaluates**: Training environment for SWE agents (with evaluation component)
- **Tasks**: 2,438 Python task instances from 11 repos
- **Models evaluated**: Multiple open-weight LLMs
- **Per-task data available**: Yes -- trajectories and models released
- **Links**: [GitHub](https://github.com/SWE-Gym/SWE-Gym) | [HF](https://huggingface.co/SWE-Gym) | [Paper (ICML 2025)](https://arxiv.org/abs/2412.21139)

### R2E-Gym
- **What it evaluates**: Procedural environment for training SWE agents with hybrid verifiers
- **Tasks**: 8,100+ problems across 13 repos
- **Models evaluated**: Multiple open-weight models; achieved 51% on SWE-bench Verified
- **Per-task data available**: Yes
- **Links**: [Website](https://r2e-gym.github.io/) | [GitHub](https://github.com/R2E-Gym/R2E-Gym) | [Paper (COLM 2025)](https://arxiv.org/abs/2504.07164)

---

## 7. Code Editing & Debugging Benchmarks

### CodeEditorBench
- **What it evaluates**: Code editing (debugging, translating, polishing, requirement switching)
- **Tasks**: 7,961 tasks with avg 44 test cases each (C++, Java, Python)
- **Models evaluated**: 17 LLMs
- **Per-task data available**: Yes
- **Links**: [Website](https://codeeditorbench.github.io/) | [GitHub](https://github.com/CodeEditorBench/CodeEditorBench) | [Paper](https://arxiv.org/abs/2404.03543)

### EDIT-Bench
- **What it evaluates**: Real-world instructed code edits collected in-the-wild from VS Code users
- **Tasks**: 540 problems across multiple languages and use cases
- **Models evaluated**: 40 LLMs (best: claude-sonnet-4 at 64.8% pass@1)
- **Per-task data available**: Yes
- **Links**: [Website](https://waynechi.com/edit-bench/) | [Paper (ICML 2025)](https://arxiv.org/abs/2511.04486)

### DebugBench
- **What it evaluates**: LLM debugging capability across 4 bug categories and 18 bug types
- **Tasks**: 4,253 instances in C++, Java, Python
- **Models evaluated**: Multiple LLMs
- **Per-task data available**: Yes -- GitHub repo
- **Links**: [GitHub](https://github.com/thunlp/DebugBench) | [Paper](https://arxiv.org/abs/2401.04621)

### MdEval
- **What it evaluates**: Multilingual code debugging (bug fixing, localization, identification)
- **Tasks**: 3,900 problems across 20 languages and 47 error types
- **Models evaluated**: Multiple LLMs + custom xDebugCoder
- **Per-task data available**: Yes -- HuggingFace dataset
- **Links**: [HF Dataset](https://huggingface.co/datasets/Multilingual-Multimodal-NLP/MdEval) | [Paper](https://arxiv.org/abs/2411.02310)

### CRUXEval
- **What it evaluates**: Code reasoning, understanding, and execution (input/output prediction)
- **Tasks**: 800 Python functions (CRUXEval-I + CRUXEval-O)
- **Models evaluated**: Multiple including GPT-4, Code Llama
- **Per-task data available**: Yes -- GitHub repo
- **Links**: [Website](https://crux-eval.github.io/) | [GitHub](https://github.com/facebookresearch/cruxeval) | [Paper](https://arxiv.org/abs/2401.03065)

### GitChameleon
- **What it evaluates**: Version-specific code generation (library version awareness)
- **Tasks**: 116 Python problems conditioned on specific library versions
- **Models evaluated**: Multiple including GPT-4o (39.9% pass@10)
- **Per-task data available**: Yes
- **Links**: [Paper](https://arxiv.org/abs/2411.05830) | GitChameleon 2.0: [Paper](https://arxiv.org/abs/2507.12367)

### OSS-Bench
- **What it evaluates**: Auto-generated benchmark from real open-source software (compilability, correctness, memory safety)
- **Tasks**: Large-scale tasks from PHP and SQL projects
- **Models evaluated**: 17 LLMs
- **Per-task data available**: Yes
- **Links**: [Paper](https://arxiv.org/abs/2505.12331)

---

## 8. Code Review, Commit & Documentation Benchmarks

### CommitBench
- **What it evaluates**: Commit message generation
- **Tasks**: 1M+ real commits from thousands of repos across 6 languages
- **Models evaluated**: Multiple LLMs in survey studies
- **Per-task data available**: Yes -- public dataset
- **Links**: [Paper](https://arxiv.org/abs/2403.05188)

### CodeFuse-CR-Bench
- **What it evaluates**: End-to-end code review evaluation
- **Tasks**: 601 instances from 70 Python projects covering 9 PR problem domains
- **Models evaluated**: Multiple including Gemini-2.5-Pro (highest)
- **Per-task data available**: Yes
- **Links**: [Paper](https://arxiv.org/abs/2509.14856)

### TestGenEval
- **What it evaluates**: Test generation (compile@k, pass@k, coverage improvement)
- **Tasks**: 1,210 Python test files
- **Models evaluated**: Multiple LLMs
- **Per-task data available**: Yes
- **Links**: [Website](https://testgeneval.github.io/)

---

## 9. Security & Vulnerability Benchmarks

### SEC-bench
- **What it evaluates**: LLM agents on real-world software security tasks (vulnerability reproduction + patching)
- **Tasks**: Automatically constructed CVE instances ($0.87/instance)
- **Models evaluated**: Multiple agents
- **Per-task data available**: Yes
- **Links**: [Paper (NeurIPS 2025)](https://arxiv.org/abs/2506.11791)

### SecBench
- **What it evaluates**: LLM cybersecurity knowledge (MCQ + short answer)
- **Tasks**: 44,823 MCQs + 3,087 SAQs across 9 sub-domains
- **Models evaluated**: Multiple LLMs
- **Per-task data available**: Yes -- GitHub repo
- **Links**: [GitHub](https://github.com/secbench-git/SecBench)

### CyberSecEval (Meta/Purple Llama)
- **What it evaluates**: Cybersecurity vulnerabilities and defensive capabilities of LLMs
- **Tasks**: Comprehensive security benchmark suite
- **Models evaluated**: Multiple LLMs
- **Per-task data available**: Yes
- **Links**: [GitHub](https://github.com/meta-llama/PurpleLlama/blob/main/CybersecurityBenchmarks/README.md)

### SafeGenBench
- **What it evaluates**: Security of LLM-generated code (vulnerability detection)
- **Tasks**: 558 prompts across 12 languages, 44 vulnerability types (OWASP/CWE-rooted)
- **Models evaluated**: Multiple LLMs (avg 37.44% accuracy)
- **Per-task data available**: Yes
- **Links**: [Paper](https://arxiv.org/abs/2506.05692)

### SecureAgentBench
- **What it evaluates**: Secure code generation under realistic vulnerability scenarios
- **Tasks**: 105 tasks requiring analysis of codebases up to 36.4K files
- **Models evaluated**: Multiple coding agents
- **Per-task data available**: Yes
- **Links**: [Paper](https://arxiv.org/abs/2509.22097)

### Defects4J (for LLMs)
- **What it evaluates**: Java bug fixing (classic APR benchmark, now used for LLM evaluation)
- **Tasks**: 835 real bugs from 17 Java projects
- **Models evaluated**: Multiple LLMs in recent studies (potential contamination concerns)
- **Per-task data available**: Yes
- **Links**: [GitHub](https://github.com/rjust/defects4j)

### Defects4C
- **What it evaluates**: C/C++ bug fixing and vulnerability repair with LLMs
- **Tasks**: 248 buggy functions + 102 vulnerable functions with test cases
- **Models evaluated**: 24 state-of-the-art LLMs
- **Per-task data available**: Yes -- GitHub repo
- **Links**: [GitHub](https://github.com/defects4c/defects4c) | [Paper](https://arxiv.org/abs/2510.11059)

### GitBug-Java
- **What it evaluates**: Recent Java bugs (2023 commit history, avoids contamination)
- **Tasks**: 199 bugs from 55 repos
- **Models evaluated**: Multiple LLMs
- **Per-task data available**: Yes -- reproducible benchmark
- **Links**: [Paper](https://arxiv.org/abs/2402.02961)

### GHRB (GitHub Recent Bugs)
- **What it evaluates**: Continuously gathered real-world Java bugs for LLM debugging
- **Tasks**: Continuously growing dataset
- **Models evaluated**: Multiple LLMs
- **Per-task data available**: Yes
- **Links**: [IEEE Paper](https://ieeexplore.ieee.org/document/10638568/)

---

## 10. Data Science & Specialized Domain Benchmarks

### DS-1000
- **What it evaluates**: Data science code generation using NumPy, Pandas, etc. (from StackOverflow)
- **Tasks**: 1,000 problems spanning 7 Python libraries
- **Models evaluated**: Codex-002 (43.3%), and others
- **Per-task data available**: Yes -- GitHub + dataset
- **Links**: [Website](https://ds1000-code-gen.github.io/) | [GitHub](https://github.com/xlang-ai/DS-1000) | [Paper (ICML 2023)](https://arxiv.org/abs/2211.11501)

### Spider 2.0
- **What it evaluates**: Enterprise text-to-SQL workflows (BigQuery, Snowflake)
- **Tasks**: 632 real-world problems with 1,000+ column databases
- **Models evaluated**: Multiple LLMs
- **Per-task data available**: Yes
- **Links**: [Website](https://spider2-sql.github.io/) | [Paper](https://arxiv.org/abs/2411.07763)

### ODEX
- **What it evaluates**: Open-domain code generation from NL across 79 libraries, 4 natural languages
- **Tasks**: 945 NL-Code pairs + 1,707 test cases
- **Models evaluated**: Codex, CodeGen, and others
- **Per-task data available**: Yes -- GitHub repo
- **Links**: [GitHub](https://github.com/zorazrw/odex) | [Paper (EMNLP 2023)](https://arxiv.org/abs/2212.10481)

### InfiBench
- **What it evaluates**: Freeform QA about coding across 15 programming languages
- **Tasks**: 234 Stack Overflow questions with 4 metric types
- **Models evaluated**: 100+ code LLMs
- **Per-task data available**: Yes -- GitHub evaluation harness + HuggingFace dataset
- **Links**: [Website](https://infi-coder.github.io/infibench/) | [GitHub](https://github.com/infi-coder/infibench-evaluation-harness) | [HF Dataset](https://huggingface.co/datasets/llylly001/InfiBench) | [Paper (NeurIPS 2024)](https://arxiv.org/abs/2404.07940)

### CodeRAG-Bench
- **What it evaluates**: Retrieval-augmented code generation (9K tasks + 25M retrieval documents)
- **Tasks**: 9,000 coding tasks from basic to repo-level
- **Models evaluated**: 10 retrievers x 10 LMs
- **Per-task data available**: Yes -- GitHub + website
- **Links**: [Website](https://code-rag-bench.github.io/) | [GitHub](https://github.com/code-rag-bench/code-rag-bench) | [Paper (NAACL 2025 Findings)](https://arxiv.org/abs/2406.14497)

### FrontendBench
- **What it evaluates**: Front-end development (concept explanation, utilities, games, web interfaces, data viz)
- **Tasks**: 148 curated tasks across 5 categories
- **Models evaluated**: Multiple LLMs (90.54% agreement with human evaluation)
- **Per-task data available**: Yes
- **Links**: [Paper](https://arxiv.org/abs/2506.13832)

### WebUIBench
- **What it evaluates**: Multimodal WebUI-to-Code (perception, HTML programming, understanding)
- **Tasks**: 21,000 QA pairs from 700+ real websites
- **Models evaluated**: Multiple multimodal LLMs
- **Per-task data available**: Yes
- **Links**: [Paper](https://aclanthology.org/2025.findings-acl.815/)

---

## 11. Tool Use & Function Calling Benchmarks

### Berkeley Function Calling Leaderboard (BFCL / Gorilla)
- **What it evaluates**: Function calling / tool use across languages with AST-based evaluation
- **Tasks**: Serial and parallel function calls, multiple languages
- **Models evaluated**: 60+ models on leaderboard
- **Per-task data available**: Yes -- leaderboard
- **Links**: [Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) | [GitHub](https://github.com/ShishirPatil/gorilla) | [Paper](https://arxiv.org/abs/2305.15334)

### ToolBench
- **What it evaluates**: Tool manipulation with 16,000+ APIs
- **Tasks**: 126,000+ instruction-solution pairs across 3,451 tools
- **Models evaluated**: Multiple LLMs
- **Per-task data available**: Yes -- GitHub
- **Links**: [GitHub](https://github.com/sambanova/toolbench)

### API-Bank
- **What it evaluates**: Tool-augmented LLM planning, retrieval, and API calling
- **Tasks**: 73 API tools, 314 dialogues, 753 API calls
- **Models evaluated**: Multiple LLMs
- **Per-task data available**: Yes
- **Links**: [Paper](https://arxiv.org/abs/2304.08244)

---

## 12. Web Agent Benchmarks

### WebArena
- **What it evaluates**: Realistic web tasks on self-hosted domains
- **Tasks**: 812 tasks across multiple web environments
- **Models evaluated**: Multiple LLM-based agents
- **Per-task data available**: Yes -- GitHub repo
- **Links**: [GitHub](https://github.com/web-arena-x/webarena) | [Paper](https://arxiv.org/abs/2307.13854)

### VisualWebArena
- **What it evaluates**: Multimodal web agent tasks (image-text comprehension, spatial reasoning)
- **Tasks**: 910 tasks across Classifieds, Shopping, Reddit environments
- **Models evaluated**: Multiple multimodal agents
- **Per-task data available**: Yes -- GitHub repo
- **Links**: [GitHub](https://github.com/web-arena-x/visualwebarena) | [Paper](https://arxiv.org/abs/2401.13649)

### Mind2Web
- **What it evaluates**: Web agent performance on static human-collected data
- **Tasks**: Large-scale dataset of web tasks
- **Models evaluated**: Multiple web agents
- **Per-task data available**: Yes
- **Links**: [GitHub](https://github.com/OSU-NLP-Group/Online-Mind2Web) | [Paper](https://arxiv.org/abs/2306.06070)

### MiniWoB++
- **What it evaluates**: Low-level web tasks (button clicking, form filling, navigation)
- **Tasks**: 100+ diverse single-page HTML tasks
- **Models evaluated**: Multiple RL and LLM agents
- **Per-task data available**: Yes
- **Links**: Part of BrowserGym ecosystem

### WorkArena
- **What it evaluates**: Enterprise software workflows on ServiceNow platform
- **Tasks**: Realistic daily workflows (filtering, forms, search, catalogs)
- **Models evaluated**: Multiple agents
- **Per-task data available**: Yes
- **Links**: Part of [BrowserGym](https://github.com/ServiceNow/BrowserGym)

### WebVoyager
- **What it evaluates**: Real-world website navigation and task completion
- **Tasks**: 643 tasks across 15 real websites
- **Models evaluated**: Multiple web agents
- **Per-task data available**: Yes
- **Links**: [Paper](https://arxiv.org/abs/2401.13919)

### ST-WebAgentBench
- **What it evaluates**: Safety and trustworthiness of web agents (enterprise scenarios)
- **Tasks**: 222 tasks across 3 applications, 6 safety dimensions
- **Models evaluated**: 3+ state-of-the-art agents
- **Per-task data available**: Yes -- GitHub + HuggingFace
- **Links**: [GitHub](https://github.com/segev-shlomov/ST-WebAgentBench) | [HF Dataset](https://huggingface.co/datasets/dolev31/st-webagentbench) | [Paper](https://arxiv.org/abs/2410.06703)

### BrowserGym
- **What it evaluates**: Unified gym-like environment for web agent evaluation
- **Tasks**: Aggregates multiple web benchmarks
- **Models evaluated**: Multiple agents via AgentLab
- **Per-task data available**: Yes
- **Links**: [GitHub](https://github.com/ServiceNow/BrowserGym)

---

## 13. OS / Desktop / Mobile Agent Benchmarks

### OSWorld
- **What it evaluates**: Open-ended tasks in real computer environments (Ubuntu, Windows, macOS)
- **Tasks**: 369 tasks (OSWorld-Verified) spanning file management, system config, document editing, etc.
- **Models evaluated**: Multiple multimodal agents
- **Per-task data available**: Yes
- **Links**: [Website](https://os-world.github.io/)

### WindowsAgentArena
- **What it evaluates**: Multi-modal OS agent tasks on real Windows OS
- **Tasks**: 150+ diverse Windows tasks
- **Models evaluated**: Multiple agents (parallelizable on Azure)
- **Per-task data available**: Yes
- **Links**: [Website](https://microsoft.github.io/WindowsAgentArena/) | [Paper](https://arxiv.org/abs/2409.08264)

### AndroidWorld
- **What it evaluates**: Dynamic, parameterized mobile tasks on Android
- **Tasks**: Effectively infinite unique task instances via parameterized generation
- **Models evaluated**: Multiple mobile agents
- **Per-task data available**: Yes
- **Links**: [Paper (ICLR 2025)](https://arxiv.org/abs/2405.14573)

---

## 14. Interactive Coding Agent Benchmarks

### AppWorld
- **What it evaluates**: Interactive coding agents using 457 APIs across 9 apps
- **Tasks**: 750 autonomous agent tasks (normal + challenge difficulty)
- **Models evaluated**: GPT-4o (~49% normal, ~30% challenge), and others
- **Per-task data available**: Yes -- leaderboard
- **Links**: [Website](https://appworld.dev/) | [Leaderboard](https://github.com/StonyBrookNLP/appworld-leaderboard) | [GitHub](https://github.com/StonyBrookNLP/appworld) | [Paper (ACL 2024 Best Resource)](https://arxiv.org/abs/2407.18901)

### AgentBench
- **What it evaluates**: LLMs as agents across 8 distinct environments
- **Tasks**: 8 environments (OS, DB, web, coding, gaming, etc.)
- **Models evaluated**: 29 API-based and open-source LLMs
- **Per-task data available**: Yes -- GitHub repo
- **Links**: [GitHub](https://github.com/THUDM/AgentBench) | [Paper (ICLR 2024)](https://arxiv.org/abs/2308.03688)

### GitTaskBench
- **What it evaluates**: Code agents solving real-world tasks through repo leveraging
- **Tasks**: 54 tasks across 7 modalities and 7 domains
- **Models evaluated**: OpenHands+Claude 3.7 (48.15%), RepoMaster+Claude 3.5 (62.96%)
- **Per-task data available**: Yes -- GitHub repo with evaluation harness
- **Links**: [GitHub](https://github.com/QuantaAlpha/GitTaskBench) | [Paper](https://arxiv.org/abs/2508.18993)

---

## 15. Research Reproduction & Scientific Agent Benchmarks

### PaperBench (OpenAI)
- **What it evaluates**: AI's ability to replicate ICML 2024 research papers from scratch
- **Tasks**: 8,316 individually gradable tasks across 20 ICML papers
- **Models evaluated**: Claude 3.5 Sonnet (21.0%), human PhDs as baseline
- **Per-task data available**: Yes -- rubrics + open-source code
- **Links**: [OpenAI Blog](https://openai.com/index/paperbench/) | [GitHub](https://github.com/openai/preparedness) | [Paper (ICML 2025)](https://arxiv.org/abs/2504.01848)

### CORE-Bench
- **What it evaluates**: Computational reproducibility of published research
- **Tasks**: 270 tasks from 90 papers across CS, social science, medicine (3 difficulty levels)
- **Models evaluated**: Multiple agents
- **Per-task data available**: Yes
- **Links**: [Paper](https://arxiv.org/abs/2409.11363)

### ScienceAgentBench
- **What it evaluates**: Scientific reasoning + code generation in data-driven discovery workflows
- **Tasks**: 102 tasks from 44 publications across 4 scientific disciplines
- **Models evaluated**: Multiple agents (best: 32.4%)
- **Per-task data available**: Yes -- leaderboard
- **Links**: [Leaderboard](https://hal.cs.princeton.edu/scienceagentbench) | [GitHub](https://github.com/OSU-NLP-Group/ScienceAgentBench) | [Paper (ICLR 2025)](https://arxiv.org/abs/2410.05080)

### MLAgentBench
- **What it evaluates**: LLM agents on ML experimentation tasks
- **Tasks**: 13 tasks from improving CIFAR-10 performance to BabyLM challenges
- **Models evaluated**: Multiple LLM agents
- **Per-task data available**: Yes
- **Links**: [Paper](https://arxiv.org/abs/2310.03302)

### MLGym
- **What it evaluates**: AI research agents (first Gym environment for research tasks)
- **Tasks**: Open-ended ML research tasks
- **Models evaluated**: Multiple agents
- **Per-task data available**: Yes
- **Links**: [Paper](https://arxiv.org/abs/2502.14499)

### ResearchCodeBench
- **What it evaluates**: Implementing code from recent ML research papers
- **Tasks**: 212 coding challenges from 20 top 2024-2025 papers
- **Models evaluated**: 30+ LLMs (best: Gemini-2.5-Pro at 37.3%)
- **Per-task data available**: Yes -- execution-based tests
- **Links**: [Website](https://researchcodebench.github.io/) | [Paper](https://arxiv.org/abs/2506.02314)

---

## 16. Comprehensive Evaluation Platforms & Leaderboards

### DPAI Arena (JetBrains)
- **What it evaluates**: Multi-language, multi-framework, multi-workflow AI coding agent evaluation
- **Tasks**: Track-based architecture (patching, bug fixing, PR review, test generation, static analysis)
- **Models evaluated**: Multiple coding agents
- **Per-task data available**: Yes -- open platform
- **Links**: [Website](https://dpaia.dev/) | [Blog](https://blog.jetbrains.com/blog/2025/10/28/introducing-developer-productivity-ai-arena-an-open-platform-for-ai-coding-agents-benchmarks/)

### Big Code Models Leaderboard
- **What it evaluates**: Open-source code generation models on HumanEval, MultiPL-E, etc.
- **Tasks**: Multiple benchmarks aggregated
- **Models evaluated**: Many open-source models
- **Per-task data available**: Yes -- interactive leaderboard
- **Links**: [HuggingFace Space](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard)

### bigcode-evaluation-harness
- **What it evaluates**: Framework supporting HumanEval, HumanEval+, MBPP, MBPP+, APPS, DS-1000, and more
- **Tasks**: 7+ benchmarks
- **Models evaluated**: Any autoregressive code LM
- **Per-task data available**: Yes -- evaluation harness generates per-task results
- **Links**: [GitHub](https://github.com/bigcode-project/bigcode-evaluation-harness)

### Aider Leaderboard
- **What it evaluates**: LLM code editing skill (Exercism-based)
- **Tasks**: 133 Python exercises (original) + 225 polyglot exercises (C++, Go, Java, JS, Python, Rust)
- **Models evaluated**: 30+ models
- **Per-task data available**: Yes -- pass rates per task available
- **Links**: [Leaderboard](https://aider.chat/docs/leaderboards/) | [GitHub](https://github.com/Aider-AI/aider/blob/main/benchmark/README.md)

### LiveBench (Coding Category)
- **What it evaluates**: Contamination-limited coding tasks with verifiable ground truth
- **Tasks**: Subset of 18 diverse tasks including coding (refreshed monthly)
- **Models evaluated**: Many LLMs
- **Per-task data available**: Yes -- all results verifiable
- **Links**: [Website](https://livebench.ai/) | [GitHub](https://github.com/LiveBench/LiveBench) | [Paper](https://arxiv.org/abs/2406.19314)

---

## 17. Coding Arena / Human Preference Benchmarks

### Copilot Arena
- **What it evaluates**: LLM code completion via real developer preferences (VS Code extension)
- **Tasks**: 4.5M+ suggestions served; 32,000+ pairwise judgments
- **Models evaluated**: 10+ models (Claude, DeepSeek top performers)
- **Per-task data available**: Yes -- Elo ratings + pairwise data
- **Links**: [Website](https://arena.ai/blog/copilot-arena/) | [Blog](https://blog.ml.cmu.edu/2025/04/09/copilot-arena-a-platform-for-code/) | [Paper (ICML 2025)](https://arxiv.org/abs/2502.09328)

### Code Arena (LMArena)
- **What it evaluates**: AI models building complete applications (agentic behavior, file management, live rendering)
- **Tasks**: Full application building with persistent sessions
- **Models evaluated**: Multiple LLMs with confidence intervals
- **Per-task data available**: Yes -- structured evaluations
- **Links**: [Website](https://codearenaeval.github.io/) | [InfoQ](https://www.infoq.com/news/2025/11/code-arena/)

### AutoCodeArena
- **What it evaluates**: Automated LLM code generation ranking via LLM-as-a-Judge + Elo rating
- **Tasks**: Execution-centric coding tasks
- **Models evaluated**: Multiple LLMs
- **Per-task data available**: Yes -- Elo rankings
- **Links**: Referenced at [Emergent Mind](https://www.emergentmind.com/topics/autocodearena)

### CodeArena (Collective Evaluation Platform)
- **What it evaluates**: Collective LLM code generation evaluation and alignment
- **Tasks**: Multi-round code evaluation
- **Models evaluated**: Multiple LLMs
- **Per-task data available**: Yes
- **Links**: [Paper (ACL 2025 Demo)](https://aclanthology.org/2025.acl-demo.48/)

---

## Additional / Niche Benchmarks

| Benchmark | Focus | Tasks | Paper/Link |
|-----------|-------|-------|------------|
| **CodeMind** | Code reasoning (independent/dependent/specification) | Multi-dimensional reasoning tasks | [arXiv](https://arxiv.org/abs/2402.09664) |
| **CodeScope** | Multilingual multitask multidimensional evaluation | Understanding + generation | Referenced in survey papers |
| **CoNaLa** | NL-to-code (StackOverflow mined) | ~2,800 curated examples | [Paper](https://arxiv.org/abs/1805.08949) |
| **STEPWISE-CODEX-Bench** | Multi-function comprehension + execution reasoning | ByteDance benchmark | Referenced in Awesome-Code-Benchmark |
| **GitBug-Actions** | Reproducible bug-fix benchmarks via GitHub Actions | Continuous collection | [Paper](https://arxiv.org/abs/2310.15642) |
| **CodeFuse-CommitEval** | Commit message + code change inconsistency detection | Multi-task evaluation | [Paper](https://arxiv.org/abs/2511.19875) |

---

## Meta-Resources

- **Awesome-Code-Benchmark**: Comprehensive curated list of all coding benchmarks. [GitHub](https://github.com/tongye98/Awesome-Code-Benchmark)
- **Awesome-Code-LLM**: Curated list of code LLM research including benchmarks. [GitHub](https://github.com/huybery/Awesome-Code-LLM) | [GitHub (codefuse-ai)](https://github.com/codefuse-ai/Awesome-Code-LLM)
- **LLM Agent Benchmark List**: [GitHub](https://github.com/zhangxjohn/LLM-Agent-Benchmark-List)
- **250 LLM Benchmarks & Evaluation Datasets**: [GitHub](https://github.com/VyetGokyra/awaresome_LLM_eval_benchmark)
- **llm-stats.com**: Aggregated benchmark scores. [Website](https://llm-stats.com/benchmarks)
- **Artificial Analysis**: LLM benchmark aggregation. [Website](https://artificialanalysis.ai/leaderboards/models)

---

## Summary: Benchmarks with Confirmed Per-Model Per-Task Data Availability

The following benchmarks have confirmed public multi-model, per-task result matrices (or the infrastructure to assemble them):

| Benchmark | # Tasks | # Models Evaluated | Data Format |
|-----------|---------|-------------------|-------------|
| HumanEval / HumanEval+ | 164 | 100+ | pass@k per problem |
| MBPP / MBPP+ | 399-974 | 100+ | pass@k per problem |
| BigCodeBench | 1,140 | 60+ | Leaderboard + dataset |
| LiveCodeBench | 1,055+ | 50+ | Leaderboard + HF dataset |
| SWE-bench (all variants) | 300-2,294 | 50+ | Per-instance resolution |
| SWE-PolyBench | 2,110 | Multiple | Evaluation harness |
| Aider Leaderboard | 133/225 | 30+ | Pass rates per task |
| EvalPlus | 164+399 | 100+ | Leaderboard |
| BFCL (Gorilla) | Multi-category | 60+ | Leaderboard |
| InfiBench | 234 | 100+ | Per-question scores |
| APPS | 5,000 test | 20+ | Public dataset |
| CodeContests | 13K+ | Multiple | Public dataset |
| MultiPL-E | HumanEval/MBPP x 18 langs | Many | Big Code Leaderboard |
| Copilot Arena | Continuous | 10+ | Elo ratings |
| AppWorld | 750 | Multiple | Leaderboard |
| AgentBench | 8 environments | 29 | Per-environment scores |
| CRUXEval | 800 | Multiple | Per-task pass rates |
| DebugBench | 4,253 | Multiple | Per-bug-type results |
| ResearchCodeBench | 212 | 30+ | Execution-based per-task |
| PaperBench | 8,316 | Multiple | Per-rubric scores |
| CodeEditorBench | 7,961 | 17 | Per-task results |
| EDIT-Bench | 540 | 40 | Per-task pass@1 |
| Defects4C | 248+102 | 24 | Per-bug repair rates |

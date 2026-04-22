# Contributing to torch_measure

Thanks for your interest in contributing! `torch_measure` is developed by the
[AIMS Foundations](https://github.com/aims-foundations) as the software component of our
broader Measurement Science stack. Contributions from outside the lab are very
welcome.

* [Project Overview](#project-overview)
* [Ways to Contribute](#ways-to-contribute)
* [Getting Involved](#getting-involved)
* [Development Environment Setup](#development-environment-setup)
* [Submitting a Pull Request](#submitting-a-pull-request)
* [Citation and Attribution](#citation-and-attribution)

## Project Overview

See the [README](README.md) for what `torch_measure` is and why it exists.
For contributors: the core psychometric machinery (IRT and factor models,
standard metrics, the main fitters) is largely in place. The active frontier
is the broader work listed in [Ways to Contribute](#ways-to-contribute) —
benchmark diagnostics, harness integrations, predictive-evaluation tools,
validity and reliability tooling.

The package is organized flat-by-domain under `src/torch_measure/` — each
domain (`models/`, `metrics/`, `fitting/`, `data/`, `datasets/`, `cat/`,
`viz/`) has a matching `tests/test_<domain>/` directory. Most contributions
add one file per feature in the appropriate domain, with a mirrored test
file. If your contribution introduces a new capability area (a new
submodule, a new response type, a cross-cutting workflow), open a
[Discord](https://discord.gg/F6xbEwvvhb) thread before you write much code
so we can agree on layout, naming, and the right abstractions.

All new public APIs need:

* A numpy-style docstring with a short usage example
* A corresponding entry in the Sphinx docs under `docs/source/`
* At least one test, with appropriate pytest markers (`slow`, `gpu`,
  `network`) for expensive cases

## Ways to Contribute

Contributions are welcome across the full stack of AI measurement, not just
psychometrics in PyTorch. Useful categories include:

**Measurement models and methods**
* New probabilistic models (such as IRT, factor)
* New psychometric metrics (reliability, information, scalability, DIF, …)
* New fitting methods (MLE variants, EM, JML, SVI, …)

**Benchmark work**
* Data loaders for new benchmarks
* Benchmark diagnostics — DIF detection, item exposure analysis, contamination
  checks, item quality audits
* Integrations with evaluation harnesses (HELM, `lm-eval-harness`, Inspect, …)

**Predictive evaluation**
* Capability inference from sparse or partial evaluation data
* Cross-benchmark transfer and prediction of performance on unseen items
* Uncertainty quantification for benchmark scores and leaderboard comparisons

**Validity and reliability tooling**
* Construct-validity checks — does a benchmark measure what it claims?
* Reliability and calibration estimators for new response types (pairwise,
  graded, LLM-judge, trace-based)

**Infrastructure and docs**
* Bug fixes, tests, performance and GPU-kernel optimizations
* Documentation, tutorials, and worked examples on real benchmark data

Not everything happens through a pull request — feel free to drop into our
[Discord](https://discord.gg/F6xbEwvvhb) to discuss ideas first, especially
for contributions in the newer categories above where we're still figuring out
the right abstractions.

## Getting Involved

**Find something to work on.** Browse
[open issues](https://github.com/aims-foundations/torch_measure/issues) and
comment on one you'd like to take so we don't duplicate work. If nothing
looks like a good fit, say hello on Discord and we'll help find something
that matches your interests.

**Ask questions.** [Discord](https://discord.gg/F6xbEwvvhb) is best for
quick questions, design discussion, and general chat. Open a GitHub issue
for bug reports, feature requests, and anything that should have a
permanent record.

## Development Environment Setup

```bash
git clone https://github.com/aims-foundations/torch_measure
cd torch_measure
pip install -e ".[dev,test,docs]"
pre-commit install
pytest
```

The test suite uses pytest markers to gate expensive tests:

* `slow` — long-running tests
* `gpu` — requires a CUDA device
* `network` — requires network access (e.g. HuggingFace downloads)

Run only the fast, local, CPU tests with:

```bash
pytest -m "not slow and not gpu and not network"
```

## Submitting a Pull Request

1. Fork the repo and create a feature branch off `main`.
2. Make your changes. Before opening the PR, verify:
   - [ ] `pytest` passes
   - [ ] New public APIs have docstrings and tests
   - [ ] Docs under `docs/source/` updated if you changed user-facing behavior
   - [ ] Commit messages are short, single-line, and describe the change
   - [ ] No committed data files — data belongs in
     [measurement-db](https://huggingface.co/datasets/aims-foundations/measurement-db)
3. Open a pull request against `main`.
4. CI will run linting and tests on the PR. A maintainer will review; most
   PRs need one approval before merge.
5. We squash-merge PRs by default.

## Citation and Attribution

If your contribution is substantial (e.g. a new model family or a major
refactor) and leads to inclusion in a paper derived from `torch_measure`, we
are glad to discuss author credit. See the citation block in the
[README](README.md) for the current software citation.

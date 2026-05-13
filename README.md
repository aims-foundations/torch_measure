# torch_measure

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Discord](https://img.shields.io/badge/Discord-join%20chat-5865F2.svg)](https://discord.gg/F6xbEwvvhb)

**PyTorch-native toolkit for predictive evaluation of AI systems.**

Benchmark scores increasingly gate deployment decisions but rarely predict how a model will behave in production. `torch_measure` treats evaluation itself as a predictive modeling problem: latent-variable models infer a system's capability directly from sparse benchmark observations and predict its performance on unseen tasks. Built on PyTorch, with GPU-accelerated IRT, factor models, amortized inference, adaptive testing, and tabular baselines.

## Installation

With **pip**:

```bash
pip install torch_measure
```

With **[uv](https://docs.astral.sh/uv/)** (faster; drop-in replacement for pip):

```bash
uv pip install torch_measure        # into the active environment
uv add torch_measure                # into a uv-managed project
```

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details, or drop by our [Discord](https://discord.gg/F6xbEwvvhb) to chat.

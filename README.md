# torch_measure

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Discord](https://img.shields.io/badge/Discord-join%20chat-5865F2.svg)](https://discord.gg/F6xbEwvvhb)

**PyTorch-native measurement science toolkit for AI evaluation.**

Benchmark scores increasingly gate deployment decisions but rarely predict how a model will behave in production. `torch_measure` brings the measurement-science apparatus — item response theory, adaptive testing, psychometric metrics, and factor models — to the PyTorch ecosystem, so AI evaluations can be designed and interpreted with the rigor the stakes now demand.

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

With optional dependencies (same syntax for both — just prefix `uv` if desired):

```bash
pip install torch_measure[all]          # Everything
pip install torch_measure[bayesian]     # Pyro-based Bayesian IRT
pip install torch_measure[data]         # HuggingFace data loaders
pip install torch_measure[viz]          # Visualization
```

## Quick Start

```python
import torch
from torch_measure.models import Rasch
from torch_measure.data import ResponseMatrix

# Create a binary response matrix (models x items)
responses = torch.bernoulli(torch.rand(50, 200))
rm = ResponseMatrix(responses)

# Fit a Rasch (1PL) model
model = Rasch(n_subjects=rm.n_rows, n_items=rm.n_cols)
model.fit(rm.data, method="mle")

# Get estimated abilities and difficulties
abilities = model.ability          # (50,) subject ability parameters
difficulties = model.difficulty    # (200,) item difficulty parameters

# Predict response probabilities
probs = model.predict()            # (50, 200) predicted P(correct)
```

### Adaptive Testing

```python
from torch_measure.cat import AdaptiveTester

# Efficiently estimate a new model's ability using fewer items
tester = AdaptiveTester(model, strategy="fisher")
estimated_ability = tester.run(responses=new_model_responses, budget=50)
```

### Psychometric Metrics

```python
from torch_measure.metrics import tetrachoric_correlation, infit_statistics, expected_calibration_error

# Compute tetrachoric correlation matrix
corr = tetrachoric_correlation(rm.data)

# Evaluate model fit
infit = infit_statistics(predicted_probs, rm.data)

# Calibration quality
ece = expected_calibration_error(predicted_probs, rm.data)
```

## Features

| Module | Description |
|---|---|
| `torch_measure.models` | IRT (Rasch, 2PL, 3PL, Amortized, Many-Facet), Beta IRT (BetaRasch, Beta2PL), factor models, rotation |
| `torch_measure.cat` | Computerized Adaptive Testing with Fisher information selection |
| `torch_measure.fitting` | MLE, EM, JML, and Bayesian SVI parameter estimation |
| `torch_measure.metrics` | Tetrachoric correlation, Mokken scalability, infit/outfit, ECE, DIF |
| `torch_measure.data` | Response matrices, masking strategies, HuggingFace/HELM loaders |
| `torch_measure.viz` | Response heatmaps, ICCs, information plots, academic styling |

## Why torch_measure?

AI benchmark scores increasingly decide which models get deployed, but rarely predict how those models will behave in production. Measurement science — item response theory, adaptive testing, reliability, validity — has answered "how much should I trust this score?" for decades in education, psychology, and clinical assessment. `torch_measure` brings that apparatus to the PyTorch ecosystem, so evaluation can be done with the same rigor as training.

- **GPU-accelerated**: All models are PyTorch `nn.Module`s — train on GPU, use autograd.
- **Amortized inference**: Predict item parameters from embeddings without per-item calibration.
- **Built for LLM-era data**: Scales to large benchmark matrices, handles missing responses, composes with modern ML pipelines.
- **Composable**: Mix IRT, factor models, and adaptive testing freely.
- **Research-ready**: Powers 6+ published papers from [AIMS Foundations](https://github.com/aims-foundations).

## Citation

If you use `torch_measure` in your research, please cite:

```bibtex
@software{torch_measure,
  title={torch\_measure: PyTorch-native Measurement Science Toolkit},
  author={AIMS Foundations},
  url={https://github.com/aims-foundations/torch_measure},
  year={2026}
}
```

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details, or drop by our [Discord](https://discord.gg/F6xbEwvvhb) to chat.

## License

MIT License. See [LICENSE](LICENSE) for details.

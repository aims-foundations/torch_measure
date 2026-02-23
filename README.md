# torch_measure

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**PyTorch-native measurement science toolkit for AI evaluation.**

`torch_measure` brings Item Response Theory (IRT), Computerized Adaptive Testing (CAT), psychometric metrics, and factor models to the PyTorch ecosystem — GPU-accelerated, differentiable, and designed for modern AI benchmark analysis.

## Installation

```bash
pip install torch-measure
```

With optional dependencies:

```bash
pip install torch-measure[all]          # Everything
pip install torch-measure[bayesian]     # Pyro-based Bayesian IRT
pip install torch-measure[data]         # HuggingFace data loaders
pip install torch-measure[viz]          # Visualization
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

- **GPU-accelerated**: All models are PyTorch `nn.Module`s — train on GPU, use autograd
- **Amortized inference**: Predict item parameters from embeddings without per-item calibration
- **Modern AI evaluation**: Built for LLM benchmarks, not just educational testing
- **Composable**: Mix IRT models, factor models, and adaptive testing freely
- **Research-ready**: Powers 6+ published papers from the [AIMS Foundation](https://github.com/aims-foundation)

## Citation

If you use `torch_measure` in your research, please cite:

```bibtex
@software{torch_measure,
  title={torch\_measure: PyTorch-native Measurement Science Toolkit},
  author={AIMS Foundation},
  url={https://github.com/aims-foundation/torch_measure},
  year={2026}
}
```

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

## License

MIT License. See [LICENSE](LICENSE) for details.

# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Measurement models: IRT, factor models, network models, and rotation utilities."""

from torch_measure.models._base import IRTModel
from torch_measure.models._network_base import NetworkModel
from torch_measure.models._predictor import Predictor, cartesian_query, predict_dense
from torch_measure.models.amortized import AmortizedIRT
from torch_measure.models.beta_rasch import BetaRasch
from torch_measure.models.beta_twopl import BetaTwoPL
from torch_measure.models.bifactor import Bifactor
from torch_measure.models.bradley_terry import BradleyTerry
from torch_measure.models.caimira import CAIMIRA
from torch_measure.models.cold_start_lookup import ColdStartLookupPredictor
from torch_measure.models.ggm import GaussianGraphicalModel
from torch_measure.models.ising import IsingModel
from torch_measure.models.llm_judge import LLMJudge
from torch_measure.models.logistic_fm import LogisticFM
from torch_measure.models.multifacet import MultiFacetRasch
from torch_measure.models.multifacet_twopl import MultiFacet2PL
from torch_measure.models.ncf import NCF
from torch_measure.models.rasch import Rasch
from torch_measure.models.rotation import bifactor_rotation, promax_rotation, varimax_rotation
from torch_measure.models.tabpfn_predictor import TabPFNPredictor
from torch_measure.models.testlet import TestletRasch, build_testlet_map
from torch_measure.models.threepl import ThreePL
from torch_measure.models.twopl import TwoPL

__all__ = [
    "Predictor",
    "IRTModel",
    "NetworkModel",
    "cartesian_query",
    "predict_dense",
    "IsingModel",
    "GaussianGraphicalModel",
    "BradleyTerry",
    "ColdStartLookupPredictor",
    "Rasch",
    "TwoPL",
    "ThreePL",
    "BetaRasch",
    "BetaTwoPL",
    "AmortizedIRT",
    "CAIMIRA",
    "TabPFNPredictor",
    "MultiFacetRasch",
    "MultiFacet2PL",
    "TestletRasch",
    "LogisticFM",
    "Bifactor",
    "build_testlet_map",
    "varimax_rotation",
    "promax_rotation",
    "bifactor_rotation",
    "NCF",
    "LLMJudge",
]


# ---- Backward-compat shim (2026-05-22 PR #2 v2 Lane C) ----------------------
#
# LLMJudgeIRT and build_difficulty_prompt were moved from
# ``torch_measure.models`` to ``torch_measure.experimental`` because the
# module documents a convergent negative result (see
# ``torch_measure/experimental/llm_judge_irt.py`` module docstring). The
# names below are intentionally NOT in ``__all__`` and not eagerly imported,
# so wildcard imports and Sphinx autodoc no longer surface them. PEP 562
# module ``__getattr__`` resolves them on demand with a single
# DeprecationWarning so existing call sites (notably
# ``tests/test_submission/test_nan_safety.py``) keep working until they
# migrate. Remove this shim in a future release.
def __getattr__(name):
    if name in {"LLMJudgeIRT", "build_difficulty_prompt"}:
        import warnings

        warnings.warn(
            f"{name} has moved to torch_measure.experimental; "
            "import from torch_measure.experimental instead. "
            "The shim in torch_measure.models will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        from torch_measure.experimental import llm_judge_irt

        return getattr(llm_judge_irt, name)
    raise AttributeError(f"module 'torch_measure.models' has no attribute {name!r}")

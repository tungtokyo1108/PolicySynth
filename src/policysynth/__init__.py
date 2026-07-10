"""PolicySynth — decision-aligned synthetic data for decision support systems.

A synthetic customer population is only useful to a DSS if it leads managers to
the *same decisions* as real data would. PolicySynth is a generator built for
that goal, and ships with Strategy Simulation Fidelity (SSF), the metric that
measures it.

Quickstart
----------
    from policysynth import PolicySynth, ssf_score

    gen = PolicySynth().fit(df, target="Churn", value="MonthlyCharges")
    synthetic = gen.sample(5000)

    result = ssf_score(df, synthetic, scorer=my_churn_model, value_col="MonthlyCharges")
    print(result["ssf"])   # 1.0 == identical go/no-go decisions

Reference
---------
Dang, T., Phung, T.H., Nguyen, S.L., Nguyen, T. "Strategy Simulation Fidelity:
Aligning Synthetic Customer Populations with Real Decisions in Decision Support
Systems." Decision Support Systems.
"""
from .config import PolicySynthConfig
from .generator import PolicySynth
from .ssf import ssf_score, Strategy, ROIConfig, default_strategy_family
from .evaluate import (
    three_axis_report,
    membership_inference_auc,
    novelty_rate,
)
from .datasets import load_telco_sample

__version__ = "0.1.0"

__all__ = [
    "PolicySynth",
    "PolicySynthConfig",
    "ssf_score",
    "Strategy",
    "ROIConfig",
    "default_strategy_family",
    "three_axis_report",
    "membership_inference_auc",
    "novelty_rate",
    "load_telco_sample",
]

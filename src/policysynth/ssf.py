"""Strategy Simulation Fidelity (SSF) — the paper's core evaluation criterion.

SSF measures what statistical-fidelity metrics miss: *does the synthetic
population lead to the same business decisions as the real one?*

For a family of parameterised campaign strategies, we compute the go/no-go
decision (expected ROI > 0) on the real and on the synthetic population and
report the fraction on which they agree:

    SSF = (1 / K) * sum_k  1[ decision_real(k) == decision_synth(k) ]

SSF = 1.0 means a manager running what-if analysis on the synthetic data
would make exactly the decisions they'd make on real data. It is generator-
agnostic: use it to score PolicySynth *or any other synthesizer*.

This module is intentionally dependency-light (numpy + sklearn) so it can be
dropped into any evaluation harness.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Optional

import numpy as np
import pandas as pd


@dataclass
class Strategy:
    segment: str        # value-tier to target: "high" | "mid" | "low" | "all"
    discount: float     # retention offer size, as a fraction (e.g. 0.10)
    threshold: float    # only contact customers with P(target) >= threshold


def default_strategy_family() -> list[Strategy]:
    """The 48-strategy grid used in the paper."""
    fam = []
    for seg, d, p in product(["high", "mid", "low", "all"],
                             [0.05, 0.10, 0.15, 0.20],
                             [0.30, 0.50, 0.70]):
        fam.append(Strategy(seg, d, p))
    return fam


@dataclass
class ROIConfig:
    """Economics of a retention campaign (defaults follow the paper)."""
    retention_offer_cost: float = 15.0   # $ cost per contacted customer
    success_rate: float = 0.30           # P(retain | offered & would churn)
    value_recoverable: float = 0.30      # fraction of value saved on success
    fp_redemption: float = 0.30          # non-churners who redeem the discount


def _value_tiers(value: np.ndarray) -> np.ndarray:
    """Label each customer high/mid/low by value tercile."""
    t1, t2 = np.quantile(value, [1 / 3, 2 / 3])
    tier = np.full(len(value), "low", dtype=object)
    tier[value > t1] = "mid"
    tier[value > t2] = "high"
    return tier


def _strategy_roi(prob: np.ndarray, value: np.ndarray, tier: np.ndarray,
                  s: Strategy, econ: ROIConfig) -> float:
    mask = np.ones(len(prob), bool) if s.segment == "all" else (tier == s.segment)
    flag = mask & (prob >= s.threshold)
    if flag.sum() == 0:
        return 0.0
    rev_saved = (prob[flag] * value[flag] * econ.value_recoverable *
                 econ.success_rate * (1 - s.discount)).sum()
    waste = ((1 - prob[flag]) * value[flag] * s.discount * econ.fp_redemption).sum()
    cost = flag.sum() * econ.retention_offer_cost
    return float(rev_saved - waste - cost)


def ssf_score(real: pd.DataFrame,
              synthetic: pd.DataFrame,
              scorer: Callable[[pd.DataFrame], np.ndarray],
              value_col: str,
              strategy_family: Optional[list[Strategy]] = None,
              econ: Optional[ROIConfig] = None) -> dict:
    """Compute Strategy Simulation Fidelity between real and synthetic data.

    Parameters
    ----------
    real, synthetic : DataFrame
        The two populations to compare.
    scorer : callable
        Your production churn/response scorer: takes a DataFrame, returns
        P(positive) per row. This is the decision policy SSF is measured
        against — pass the *same* model you use in production.
    value_col : str
        Column holding customer value (drives segmentation and ROI).
    strategy_family : list[Strategy], optional
        Defaults to the paper's 48-strategy grid.
    econ : ROIConfig, optional
        Campaign economics.

    Returns
    -------
    dict with keys:
        ssf                 : float in [0, 1] — the headline number
        n_strategies        : int
        n_agree             : int
        mean_roi_gap_pct    : float — average |ROI_real - ROI_syn| / |ROI_real|
        per_strategy        : list of dicts (one row per strategy)
    """
    fam = strategy_family or default_strategy_family()
    econ = econ or ROIConfig()

    p_r, v_r = scorer(real), pd.to_numeric(real[value_col]).values.astype(float)
    p_s, v_s = scorer(synthetic), pd.to_numeric(synthetic[value_col]).values.astype(float)
    tier_r, tier_s = _value_tiers(v_r), _value_tiers(v_s)

    rows, n_agree, gaps = [], 0, []
    for s in fam:
        roi_r = _strategy_roi(p_r, v_r, tier_r, s, econ)
        roi_s = _strategy_roi(p_s, v_s, tier_s, s, econ)
        dec_r, dec_s = int(roi_r > 0), int(roi_s > 0)
        agree = dec_r == dec_s
        n_agree += int(agree)
        gaps.append(abs(roi_r - roi_s) / max(abs(roi_r), 1.0))
        rows.append(dict(segment=s.segment, discount=s.discount,
                         threshold=s.threshold, roi_real=roi_r, roi_syn=roi_s,
                         decision_real=dec_r, decision_syn=dec_s, agree=agree))
    K = len(fam)
    return dict(ssf=n_agree / max(K, 1), n_strategies=K, n_agree=n_agree,
                mean_roi_gap_pct=float(np.mean(gaps) * 100), per_strategy=rows)

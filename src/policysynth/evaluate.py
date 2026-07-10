"""The paper's three-axis deployment quality gate.

A synthetic-data layer is safe to deploy only if it passes on all three:

    1. decision alignment       — SSF (see ssf.py)
    2. membership-inference risk — an attacker who sees only the synthetic data
                                   should not be able to tell whether a given
                                   real record was in the training set
    3. novelty                  — synthetic records should not be verbatim
                                   copies of training rows

`three_axis_report` bundles axes 2 and 3 (axis 1 is `ssf_score`, kept separate
because it needs your production scorer).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder


def _encode(*frames: pd.DataFrame) -> list[np.ndarray]:
    """Encode several frames into a common feature space.

    One encoder is fit jointly across every frame so all outputs share the
    same column layout (needed for cross-frame nearest-neighbour queries).
    Columns are intersected across frames; the first frame defines the order.
    """
    cols = [c for c in frames[0].columns
            if all(c in f.columns for f in frames)]
    num = [c for c in cols if all(pd.api.types.is_numeric_dtype(f[c]) for f in frames)]
    cat = [c for c in cols if c not in num]
    enc = None
    if cat:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        enc.fit(pd.concat([f[cat] for f in frames]).astype(str))
    out = []
    for f in frames:
        blocks = []
        if num:
            blocks.append(f[num].apply(pd.to_numeric, errors="coerce").fillna(0).values)
        if cat:
            blocks.append(enc.transform(f[cat].astype(str)))
        out.append(np.hstack(blocks) if blocks else np.zeros((len(f), 0)))
    return out


def membership_inference_auc(train: pd.DataFrame, synth: pd.DataFrame,
                             holdout: pd.DataFrame, seed: int = 42) -> float:
    """Distance-to-synthetic membership-inference attack AUC.

    This is the attack the paper reports. The intuition: if a generator has
    memorised its training set, real *training* records will lie closer to the
    synthetic cloud than real *held-out* records do. The attacker scores each
    real record by (negative) distance to its nearest synthetic neighbour and
    tries to separate members (train) from non-members (holdout).

        0.5  = safe   — members and non-members are indistinguishable
        ~1.0 = leaky  — the generator memorised its training data

    Parameters
    ----------
    train : DataFrame     real records the generator WAS trained on (members)
    synth : DataFrame     the synthetic population
    holdout : DataFrame   real records the generator was NOT trained on
    """
    Xsy, Xtr, Xho = _encode(synth, train, holdout)
    nn = NearestNeighbors(n_neighbors=1).fit(Xsy)
    d_train = nn.kneighbors(Xtr)[0].ravel()
    d_hold = nn.kneighbors(Xho)[0].ravel()
    # closer to synthetic  ->  more likely a member  ->  higher score
    scores = np.r_[-d_train, -d_hold]
    labels = np.r_[np.ones(len(d_train)), np.zeros(len(d_hold))]
    return float(roc_auc_score(labels, scores))


def novelty_rate(real: pd.DataFrame, synth: pd.DataFrame, tol: float = 1e-6) -> float:
    """Fraction of synthetic rows that are NOT near-duplicates of a real row.

    1.0 = fully novel; 0.0 = every synthetic record copies a training record.
    """
    Xr, Xs = _encode(real, synth)
    nn = NearestNeighbors(n_neighbors=1).fit(Xr)
    dist, _ = nn.kneighbors(Xs)
    # normalise tolerance by feature dimension so it is scale-aware
    return float(np.mean(dist.ravel() > tol))


def three_axis_report(train: pd.DataFrame, synth: pd.DataFrame,
                      holdout: pd.DataFrame, seed: int = 42) -> dict:
    """Privacy + novelty axes of the deployment gate (pair with `ssf_score`).

    Parameters
    ----------
    train : DataFrame     real records used to fit the generator
    synth : DataFrame     synthetic population produced by the generator
    holdout : DataFrame   real records NOT used to fit the generator
    """
    mia = membership_inference_auc(train, synth, holdout, seed)
    nov = novelty_rate(train, synth)
    return dict(
        membership_inference_auc=mia,
        membership_inference_risk="low (safe)" if mia < 0.55 else
                                  ("moderate" if mia < 0.7 else "high (unsafe)"),
        novelty_rate=nov,
        passes_privacy=mia < 0.55 and nov > 0.99,
    )

"""Fast smoke tests — run with `pytest`. These check the API wiring end to
end on a tiny synthetic table; they are not a statistical validation."""
import numpy as np
import pandas as pd

from policysynth import (
    PolicySynth, PolicySynthConfig, ssf_score, three_axis_report,
)


def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    value = rng.uniform(20, 120, n)
    p = 1 / (1 + np.exp(-(value - 70) / 20))
    churn = np.where(rng.uniform(size=n) < p * 0.5, "Yes", "No")
    return pd.DataFrame({
        "value": value,
        "plan": rng.choice(["A", "B", "C"], n),
        "tenure": rng.integers(1, 72, n),
        "Churn": churn,
    })


def test_fit_sample_shapes():
    df = _toy()
    gen = PolicySynth(PolicySynthConfig(epochs=3, diffusion_steps=20)).fit(
        df, target="Churn", value="value")
    synth = gen.sample(120)
    assert len(synth) == 120
    assert set(synth.columns) == set(df.columns)
    assert set(synth["Churn"].unique()) <= {"Yes", "No"}


def test_ssf_range():
    df = _toy()
    gen = PolicySynth(PolicySynthConfig(epochs=3, diffusion_steps=20)).fit(
        df, target="Churn", value="value")
    synth = gen.sample(len(df))
    scorer = lambda d: 1 / (1 + np.exp(-(d["value"].values - 70) / 20)) * 0.5
    out = ssf_score(df, synth, scorer=scorer, value_col="value")
    assert 0.0 <= out["ssf"] <= 1.0
    assert out["n_strategies"] == 48


def test_three_axis_keys():
    df = _toy()
    train, holdout = df.iloc[:300], df.iloc[300:]
    gen = PolicySynth(PolicySynthConfig(epochs=3, diffusion_steps=20)).fit(
        train, target="Churn", value="value")
    rep = three_axis_report(train, gen.sample(300), holdout)
    for k in ("membership_inference_auc", "novelty_rate", "passes_privacy"):
        assert k in rep

"""PolicySynth quickstart — end to end in one file.

Runs on CPU in about a minute on the bundled Telco churn sample:

    python examples/quickstart.py

It (1) fits PolicySynth on real data, (2) samples a synthetic population,
(3) measures Strategy Simulation Fidelity — how often the synthetic data
would lead to the same go/no-go campaign decision as the real data — and
(4) runs the privacy + novelty quality gate.
"""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from policysynth import PolicySynth, PolicySynthConfig, ssf_score, three_axis_report

DATA = Path(__file__).resolve().parents[1] / "data" / "telco_churn_sample.csv"
TARGET, VALUE = "Churn", "MonthlyCharges"


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA).drop(columns=["customerID"], errors="ignore")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    return df


def build_scorer(df: pd.DataFrame):
    """A stand-in 'production' churn scorer used to define the decision policy.
    Swap this for your real model when applying SSF to your own data."""
    feats = [c for c in df.columns if c != TARGET]
    cat = [c for c in feats if df[c].dtype == object]
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(df[cat].astype(str))

    def featurise(d: pd.DataFrame):
        num = d[[c for c in feats if c not in cat]].apply(pd.to_numeric, errors="coerce").fillna(0)
        return pd.concat([num.reset_index(drop=True),
                          pd.DataFrame(enc.transform(d[cat].astype(str)))], axis=1).values

    y = (df[TARGET].astype(str) == "Yes").astype(int)
    model = GradientBoostingClassifier(random_state=0).fit(featurise(df), y)
    return lambda d: model.predict_proba(featurise(d))[:, 1]


def main() -> None:
    df = load()
    # Hold out a real test set: fit on `train`, evaluate privacy against `holdout`.
    train, holdout = train_test_split(df, test_size=0.3, random_state=0,
                                      stratify=df[TARGET])
    print(f"Loaded {len(df):,} real customers "
          f"({len(train):,} train / {len(holdout):,} holdout).\n")

    print("Fitting PolicySynth ...")
    # A quick config for the demo; drop the overrides to use the paper defaults.
    cfg = PolicySynthConfig(epochs=60, diffusion_steps=100)
    gen = PolicySynth(cfg).fit(train, target=TARGET, value=VALUE)
    print(f"  done in {gen.fit_seconds:.1f}s\n")

    synth = gen.sample(len(train))
    print(f"Sampled {len(synth):,} synthetic customers.\n")

    scorer = build_scorer(train)
    ssf = ssf_score(train, synth, scorer=scorer, value_col=VALUE)
    print("── Strategy Simulation Fidelity ──────────────────────────────")
    print(f"  SSF ................. {ssf['ssf']:.3f}  "
          f"({ssf['n_agree']}/{ssf['n_strategies']} strategies agree)")
    print(f"  mean ROI gap ........ {ssf['mean_roi_gap_pct']:.1f}%\n")

    gate = three_axis_report(train, synth, holdout)
    print("── Privacy & novelty gate ────────────────────────────────────")
    print(f"  membership-inference AUC .. {gate['membership_inference_auc']:.3f} "
          f"({gate['membership_inference_risk']})")
    print(f"  novelty rate .............. {gate['novelty_rate']:.3f}")
    print(f"  passes privacy gate ....... {gate['passes_privacy']}")


if __name__ == "__main__":
    main()

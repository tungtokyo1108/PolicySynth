"""PolicySynth quickstart — the whole method end to end in one file.

Runs on CPU in about a minute:

    pip install -e .
    python examples/quickstart.py

It (1) fits PolicySynth on real data, (2) samples a synthetic population,
(3) measures Strategy Simulation Fidelity — how often the synthetic data
would lead to the same go/no-go campaign decision as the real data — and
(4) runs the privacy + novelty quality gate.

To apply this to YOUR data, replace `load_telco_sample()` with your own
DataFrame and `build_scorer()` with your production churn/response model.
Nothing else here is dataset-specific.
"""
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from policysynth import (
    PolicySynth, PolicySynthConfig, ROIConfig,
    ssf_score, three_axis_report, load_telco_sample,
)

TARGET, VALUE = "Churn", "MonthlyCharges"


def build_scorer(df: pd.DataFrame):
    """A stand-in 'production' churn scorer that defines the decision policy.
    Swap this for your real model when you apply SSF to your own data."""
    feats = [c for c in df.columns if c != TARGET]
    cat = [c for c in feats if df[c].dtype == object]
    num_cols = [c for c in feats if c not in cat]
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(df[cat].astype(str))

    def featurise(d: pd.DataFrame):
        num = d[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0).reset_index(drop=True)
        oh = pd.DataFrame(enc.transform(d[cat].astype(str)))
        return pd.concat([num, oh], axis=1).values

    y = (df[TARGET].astype(str) == "Yes").astype(int)
    model = GradientBoostingClassifier(random_state=0).fit(featurise(df), y)
    return lambda d: model.predict_proba(featurise(d))[:, 1]


def main() -> None:
    df = load_telco_sample()
    # Hold out a real test set: fit on `train`, evaluate privacy against `holdout`.
    train, holdout = train_test_split(df, test_size=0.3, random_state=0,
                                      stratify=df[TARGET])
    print(f"Loaded {len(df):,} real customers "
          f"({len(train):,} train / {len(holdout):,} holdout).\n")

    print("Fitting PolicySynth ...")
    cfg = PolicySynthConfig(epochs=60, diffusion_steps=100)  # quick demo config
    gen = PolicySynth(cfg).fit(train, target=TARGET, value=VALUE)
    print(f"  done in {gen.fit_seconds:.1f}s\n")

    synth = gen.sample(len(train))
    print(f"Sampled {len(synth):,} synthetic customers.\n")

    # Economics tuned so the strategy family straddles the ROI=0 boundary —
    # otherwise every strategy is trivially GO or NO-GO and SSF saturates at 1.0
    # for ANY synthetic data, telling you nothing. See the SSF docs for why.
    econ = ROIConfig(retention_offer_cost=8.0, success_rate=0.35, value_recoverable=0.6)
    scorer = build_scorer(train)
    ssf = ssf_score(train, synth, scorer=scorer, value_col=VALUE, econ=econ)
    go = sum(r["decision_real"] for r in ssf["per_strategy"])
    print("── Strategy Simulation Fidelity ──────────────────────────────")
    print(f"  SSF ................. {ssf['ssf']:.3f}  "
          f"({ssf['n_agree']}/{ssf['n_strategies']} strategies agree)")
    print(f"  decision spread ..... {go} GO / {ssf['n_strategies'] - go} NO-GO on real data")
    print(f"  mean ROI gap ........ {ssf['mean_roi_gap_pct']:.1f}%")
    if go == 0 or go == ssf["n_strategies"]:
        print("  ! all real decisions fall on one side — SSF is not discriminating here;"
              "\n    adjust ROIConfig so strategies straddle ROI=0.")
    print()

    gate = three_axis_report(train, synth, holdout)
    print("── Privacy & novelty gate ────────────────────────────────────")
    print(f"  membership-inference AUC .. {gate['membership_inference_auc']:.3f} "
          f"({gate['membership_inference_risk']})")
    print(f"  novelty rate .............. {gate['novelty_rate']:.3f}")
    print(f"  passes privacy gate ....... {gate['passes_privacy']}")


if __name__ == "__main__":
    main()

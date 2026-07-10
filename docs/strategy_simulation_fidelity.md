# Strategy Simulation Fidelity (SSF)

SSF answers the question statistical-fidelity metrics ignore: **does the
synthetic population lead to the same business decisions as the real one?**

## Definition

Let `𝒮` be a family of `K` parameterised campaign strategies. For each strategy
`s`, compute the expected ROI on the real population and on the synthetic one,
and reduce each to a binary go/no-go decision `1[ROI > 0]`. SSF is the fraction
of strategies on which the two decisions agree:

```
SSF = (1/K) · Σ_{s ∈ 𝒮}  1[ decision_real(s) == decision_synth(s) ]
```

- **SSF = 1.0** — a manager running what-if analysis on the synthetic data makes
  exactly the decisions they'd make on real data.
- **SSF = 0.5** — the synthetic data provides no decision information beyond
  chance.

SSF is **generator-agnostic**: use it to score PolicySynth *or any other
synthesizer*.

## The ROI value function

Each strategy `s = (segment, discount, threshold)` targets customers in a
value segment whose predicted probability clears `threshold`. For the flagged
set, ROI is:

```
ROI(s) = revenue_saved − discount_waste − campaign_cost

revenue_saved  = Σ  p_i · value_i · value_recoverable · success_rate · (1 − discount)
discount_waste = Σ  (1 − p_i) · value_i · discount · fp_redemption
campaign_cost  = n_flagged · retention_offer_cost
```

where `p_i` is the scorer's probability for customer `i`. All constants live in
`ROIConfig`:

```python
from policysynth import ROIConfig
econ = ROIConfig(
    retention_offer_cost=15.0,   # $ per contacted customer
    success_rate=0.30,           # P(retain | offered & would churn)
    value_recoverable=0.30,      # fraction of value saved on success
    fp_redemption=0.30,          # non-churners who redeem the discount
)
```

## Configuring the strategy family

The default is a 48-strategy grid (4 segments × 4 discounts × 3 thresholds). To
use your own:

```python
from policysynth import Strategy, ssf_score

family = [Strategy(segment="high", discount=0.10, threshold=0.5),
          Strategy(segment="all",  discount=0.20, threshold=0.7)]

ssf_score(real, synth, scorer=my_model, value_col="MonthlyCharges",
          strategy_family=family, econ=econ)
```

## ⚠ Make sure SSF is *discriminating*

SSF is only meaningful when strategies fall on **both sides** of the ROI = 0
boundary. If your economics make every strategy trivially profitable (or trivially
unprofitable), then every real decision is GO (or every one is NO-GO), and SSF is
**1.0 for any synthetic data at all** — including random noise. The number then
tells you nothing about generator quality.

Check the `per_strategy` output:

```python
res = ssf_score(real, synth, scorer=model, value_col="MonthlyCharges", econ=econ)
go = sum(r["decision_real"] for r in res["per_strategy"])
assert 0 < go < res["n_strategies"], "economics don't straddle ROI=0 — SSF is uninformative"
```

Tune `ROIConfig` (lower `retention_offer_cost`, raise `success_rate` /
`value_recoverable`) until the decision spread is mixed. This is a property of
your campaign economics, not a bug — but it must hold for SSF to mean anything.

## Direction vs. magnitude

SSF certifies the **sign** of ROI (which campaigns to run), not its magnitude.
PolicySynth's ROI *magnitude* estimates diverge from real outcomes by ~70–80% on
this corpus, so:

- ✅ Use SSF + PolicySynth for **screening and ranking** campaigns.
- ⚠ Apply the documented volume-correction procedure before using ROI magnitudes
  for **budget allocation or executive sign-off**.

`ssf_score` reports `mean_roi_gap_pct` alongside `ssf` precisely so this gap is
visible, not hidden.

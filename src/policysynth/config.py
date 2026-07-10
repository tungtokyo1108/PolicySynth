"""Configuration for PolicySynth.

A single dataclass holds every knob. Defaults are the paper's settings
(Dang et al., *Decision Support Systems*) and work out of the box; you only
need to touch these to reproduce a specific experiment or tune for your data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class PolicySynthConfig:
    # ── Diffusion backbone ────────────────────────────────────────────────
    diffusion_steps: int = 100          # T — number of denoising steps
    hidden_dim: int = 384               # width of the denoiser MLP
    epochs: int = 200                   # training epochs
    batch_size: int = 256
    learning_rate: float = 2e-4

    # ── A2 · decision conditioning ────────────────────────────────────────
    # Conditions the generator on a differentiable surrogate of your churn/
    # response scorer, so decision-relevant structure is preserved. This is
    # the mechanism behind PolicySynth's seed-to-seed stability.
    use_decision_conditioning: bool = True
    surrogate_epochs: int = 30
    surrogate_hidden: int = 128
    condition_weight: float = 0.5       # weight of the auxiliary decision loss

    # ── A3 · value-stratified oversampling ────────────────────────────────
    # Oversamples top-decile "value" customers so the high-value tail — the
    # part decisions hinge on — is not washed out by the majority.
    use_value_stratified: bool = True
    value_oversample_factor: float = 2.0
    value_top_quantile: float = 0.90    # customers above this value get upweighted

    # ── A4 · tiered differential privacy ──────────────────────────────────
    # Per-parameter-group gradient clipping + Gaussian noise, with a smaller
    # privacy budget (more noise) on the layers that touch raw input/output.
    use_tiered_dp: bool = True
    dp_clip_norm: float = 1.0
    epsilon_tier1: float = 0.5          # input/output layers (most sensitive)
    epsilon_tier2: float = 2.0          # early hidden layers
    epsilon_tier3: float = 8.0          # deep hidden layers (least sensitive)

    # ── Reproducibility / hardware ────────────────────────────────────────
    seed: int = 42
    device: str = "auto"                # "auto" | "cpu" | "cuda" | "mps"

    def replace(self, **kw) -> "PolicySynthConfig":
        """Return a copy with fields overridden (handy for ablations)."""
        from dataclasses import replace
        return replace(self, **kw)

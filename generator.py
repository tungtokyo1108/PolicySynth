"""PolicySynth — a decision-conditioned diffusion generator for tabular data.

Reference implementation of the generator from:

  Dang, T., Phung, T.H., Nguyen, S.L., Nguyen, T.
  "Strategy Simulation Fidelity: Aligning Synthetic Customer Populations
   with Real Decisions in Decision Support Systems." (Decision Support Systems)

The method is a conditional diffusion model with three additions that
distinguish it from a plain tabular diffusion baseline:

  A2  decision conditioning     — a frozen surrogate of your churn/response
                                  scorer supplies an auxiliary loss that keeps
                                  the label-correlated joint intact
  A3  value-stratified sampling — oversample the high-value tail
  A4  tiered differential privacy — layer-group-wise clip + noise

Its practical payoff is *stability*: go/no-go recommendations move little
from one retraining seed to the next. See the paper for the full evaluation.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "PolicySynth requires PyTorch. Install with `pip install torch`."
    ) from e

from .config import PolicySynthConfig
from .data import TabularEncoder


def _resolve_device(name: str) -> "torch.device":
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _Denoiser(nn.Module):
    """epsilon-prediction network with sinusoidal time embedding + condition."""

    def __init__(self, in_dim: int, cond_dim: int, hidden: int, t_emb: int = 64):
        super().__init__()
        self.t_emb = t_emb
        self.proj_t = nn.Sequential(nn.Linear(t_emb, hidden), nn.SiLU(),
                                    nn.Linear(hidden, hidden))
        self.net = nn.Sequential(
            nn.Linear(in_dim + hidden + cond_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, in_dim),
        )

    def _sinusoidal(self, t):
        half = self.t_emb // 2
        freqs = torch.exp(torch.linspace(0, math.log(10000), half, device=t.device) * -1)
        ang = t.float().unsqueeze(-1) * freqs
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    def forward(self, x_t, t, cond):
        te = self.proj_t(self._sinusoidal(t))
        h = torch.cat([x_t, te, cond], dim=-1)
        return self.net(h)


class _Surrogate(nn.Module):
    """Differentiable surrogate of the decision scorer (A2)."""

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


class PolicySynth:
    """Decision-conditioned synthetic tabular data generator.

    Parameters
    ----------
    config : PolicySynthConfig, optional
        Hyper-parameters. Defaults reproduce the paper.

    Example
    -------
    >>> gen = PolicySynth()
    >>> gen.fit(df, target="Churn", value="MonthlyCharges")
    >>> synthetic = gen.sample(5000)
    """

    def __init__(self, config: Optional[PolicySynthConfig] = None):
        self.cfg = config or PolicySynthConfig()
        self.device = _resolve_device(self.cfg.device)
        self.encoder: Optional[TabularEncoder] = None
        self.target: Optional[str] = None
        self.value: Optional[str] = None
        self._target_prior: float = 0.5
        self.fit_seconds: float = 0.0

    # ── public API ────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame, target: str,
            value: Optional[str] = None,
            categorical: Optional[list[str]] = None,
            numeric: Optional[list[str]] = None) -> "PolicySynth":
        """Fit the generator.

        Parameters
        ----------
        df : DataFrame
            Real training data.
        target : str
            Binary decision/outcome column (e.g. "Churn"). Its label drives
            the decision-conditioning mechanism.
        value : str, optional
            Numeric column proxying customer value/CLV (e.g. "MonthlyCharges").
            Used for value-stratified oversampling. If None, oversampling is
            disabled and every record is weighted equally.
        categorical, numeric : list[str], optional
            Override column-type inference for the feature columns.
        """
        import time
        t0 = time.time()
        self._set_seed(self.cfg.seed)

        self.target, self.value = target, value
        y = self._binarise(df[target])
        self._target_prior = float(np.mean(y))

        if categorical is not None or numeric is not None:
            enc = TabularEncoder(categorical=categorical or [], numeric=numeric or [])
            enc.fit(df)
        else:
            enc = TabularEncoder.infer(df, target=target).fit(df)
        self.encoder = enc
        X = enc.transform(df)
        self.input_dim = X.shape[1]

        # value → 3-way tier one-hot for conditioning; weights for oversampling
        val = self._value_vector(df)
        vtier = self._value_tier(val)
        weights = self._oversample_weights(val)

        self.cond_dim = 2 + 3   # target one-hot (2) + value tier one-hot (3)
        self.net = _Denoiser(self.input_dim, self.cond_dim, self.cfg.hidden_dim).to(self.device)
        self.surrogate = _Surrogate(self.input_dim, self.cfg.surrogate_hidden).to(self.device)
        self._build_param_groups()

        # diffusion schedule
        betas = torch.linspace(1e-4, 0.02, self.cfg.diffusion_steps)
        self.alphas_cum = torch.cumprod(1.0 - betas, dim=0).to(self.device)

        if self.cfg.use_decision_conditioning:
            self._fit_surrogate(X, y)
        self._train_diffusion(X, y, vtier, weights)

        self.fit_seconds = time.time() - t0
        return self

    @torch.no_grad()
    def sample(self, n: int, target_rate: Optional[float] = None) -> pd.DataFrame:
        """Generate ``n`` synthetic records as a DataFrame.

        target_rate : float, optional
            Fraction of records with the positive target label. Defaults to
            the prior observed during fit.
        """
        if self.encoder is None:
            raise RuntimeError("Call fit() before sample().")
        rate = self._target_prior if target_rate is None else target_rate
        tgt = (torch.rand(n, device=self.device) < rate).long()
        vt = torch.randint(0, 3, (n,), device=self.device)
        cond = self._cond_vec(tgt, vt)

        x = torch.randn(n, self.input_dim, device=self.device)
        for step in reversed(range(self.cfg.diffusion_steps)):
            t = torch.full((n,), step, device=self.device, dtype=torch.long)
            ac = self.alphas_cum[t].unsqueeze(-1)
            eps = self.net(x, t, cond)
            x0 = (x - torch.sqrt(1 - ac) * eps) / torch.sqrt(ac)
            x0 = x0.clamp(-3, 3)
            if step > 0:
                acp = self.alphas_cum[t - 1].unsqueeze(-1)
                x = torch.sqrt(acp) * x0 + torch.sqrt(1 - acp) * eps
            else:
                x = x0
        df = self.encoder.inverse_transform(x.cpu().numpy())
        # re-attach the decision label used for conditioning
        df[self.target] = np.where(tgt.cpu().numpy() == 1, *self._label_pair())
        return df

    # ── internals ─────────────────────────────────────────────────────────
    def _train_diffusion(self, X, y, vtier, weights):
        cfg = self.cfg
        Xt = torch.from_numpy(X).float().to(self.device)
        yt = torch.from_numpy(y).long().to(self.device)
        vt = torch.from_numpy(vtier).long().to(self.device)
        probs = torch.from_numpy(weights / weights.sum()).float().to(self.device)

        opt = torch.optim.Adam(self.net.parameters(), lr=cfg.learning_rate)
        n, bs, T = len(Xt), cfg.batch_size, cfg.diffusion_steps
        for _ in range(cfg.epochs):
            idx = torch.multinomial(probs, n, replacement=True)
            for k in range(0, n, bs):
                ib = idx[k:k + bs]
                if len(ib) < 2:
                    continue
                xb, yb, vb = Xt[ib], yt[ib], vt[ib]
                cond = self._cond_vec(yb, vb)
                t = torch.randint(0, T, (xb.shape[0],), device=self.device)
                ac = self.alphas_cum[t].unsqueeze(-1)
                noise = torch.randn_like(xb)
                x_t = torch.sqrt(ac) * xb + torch.sqrt(1 - ac) * noise
                pred = self.net(x_t, t, cond)
                loss = F.mse_loss(pred, noise)
                if cfg.use_decision_conditioning:
                    x0 = (x_t - torch.sqrt(1 - ac) * pred) / torch.sqrt(ac)
                    logit = self.surrogate(x0).squeeze(-1)
                    loss = loss + cfg.condition_weight * \
                        F.binary_cross_entropy_with_logits(logit, yb.float())
                opt.zero_grad()
                loss.backward()
                self._apply_tiered_dp()
                opt.step()

    def _fit_surrogate(self, X, y):
        Xt = torch.from_numpy(X).float().to(self.device)
        yt = torch.from_numpy(y).float().to(self.device)
        opt = torch.optim.Adam(self.surrogate.parameters(), lr=1e-3)
        n, bs = len(Xt), self.cfg.batch_size
        for _ in range(self.cfg.surrogate_epochs):
            idx = torch.randperm(n, device=self.device)
            for k in range(0, n, bs):
                ib = idx[k:k + bs]
                logit = self.surrogate(Xt[ib]).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logit, yt[ib])
                opt.zero_grad(); loss.backward(); opt.step()
        for p in self.surrogate.parameters():
            p.requires_grad = False

    def _build_param_groups(self):
        layers = [m for m in self.net.modules() if isinstance(m, nn.Linear)]
        self.param_groups = []
        for i, layer in enumerate(layers):
            if i == 0 or i == len(layers) - 1:
                tier = 1
            elif i < len(layers) // 2:
                tier = 2
            else:
                tier = 3
            self.param_groups.append((tier, list(layer.parameters())))

    def _apply_tiered_dp(self):
        if not self.cfg.use_tiered_dp:
            return
        eps = {1: self.cfg.epsilon_tier1, 2: self.cfg.epsilon_tier2,
               3: self.cfg.epsilon_tier3}
        clip = self.cfg.dp_clip_norm
        for tier, params in self.param_groups:
            gsq = sum(float(p.grad.detach().pow(2).sum()) for p in params if p.grad is not None)
            gnorm = math.sqrt(gsq) + 1e-12
            scale = min(1.0, clip / gnorm)
            sigma = clip / max(eps[tier], 1e-3)
            for p in params:
                if p.grad is not None:
                    p.grad.mul_(scale).add_(torch.randn_like(p.grad) * sigma * 1e-3)

    def _cond_vec(self, tgt, vtier):
        c1 = F.one_hot(tgt.long(), num_classes=2).float()
        c2 = F.one_hot(vtier.long(), num_classes=3).float()
        return torch.cat([c1, c2], dim=-1)

    # ── value / label helpers (generalised — no hard-coded columns) ───────
    def _value_vector(self, df) -> np.ndarray:
        if self.value is None:
            return np.ones(len(df), dtype=np.float32)
        return pd.to_numeric(df[self.value], errors="coerce").fillna(0.0).values.astype(np.float32)

    def _value_tier(self, val: np.ndarray) -> np.ndarray:
        if self.value is None or np.allclose(val, val[0]):
            return np.zeros(len(val), dtype=np.int64)
        t1, t2 = np.quantile(val, [1 / 3, 2 / 3])
        tier = np.zeros(len(val), dtype=np.int64)
        tier[val > t1] = 1
        tier[val > t2] = 2
        return tier

    def _oversample_weights(self, val: np.ndarray) -> np.ndarray:
        w = np.ones(len(val), dtype=np.float32)
        if self.cfg.use_value_stratified and self.value is not None:
            thr = np.quantile(val, self.cfg.value_top_quantile)
            w[val >= thr] = self.cfg.value_oversample_factor
        return w

    def _binarise(self, s: pd.Series) -> np.ndarray:
        self._labels = self._detect_labels(s)
        pos = self._labels[1]
        return (s.astype(str).values == str(pos)).astype(np.float32)

    def _detect_labels(self, s: pd.Series):
        uniq = pd.Series(s.astype(str).unique())
        for pos in ("Yes", "yes", "1", "True", "true", "churn", "Churn"):
            if pos in set(uniq):
                neg = [u for u in uniq if u != pos]
                return (neg[0] if neg else "No", pos)
        vals = sorted(uniq.tolist())
        return (vals[0], vals[-1])

    def _label_pair(self):
        neg, pos = self._labels
        return pos, neg   # np.where(cond, pos, neg)

    def _set_seed(self, seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

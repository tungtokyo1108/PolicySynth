"""Generic tabular encoder/decoder for PolicySynth.

Turns an arbitrary mixed-type DataFrame into a real-valued matrix the
diffusion model can train on, and back again. No column names are
hard-coded — this is what makes PolicySynth usable on your own data,
not just the paper's Telco corpus.

Encoding scheme (deliberately simple and dependency-light):
  * numeric columns  → min-max scaled to [0, 1]
  * categorical cols → one-hot
On decode, one-hot blocks are argmax'd back to labels and numeric
columns are inverse-scaled and rounded to the original dtype.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class TabularEncoder:
    categorical: list[str] = field(default_factory=list)
    numeric: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._cat_levels: dict[str, list] = {}
        self._num_min: dict[str, float] = {}
        self._num_max: dict[str, float] = {}
        self._int_cols: set[str] = set()
        self._columns: list[str] = []

    # ── inference of column types ─────────────────────────────────────────
    @classmethod
    def infer(cls, df: pd.DataFrame, target: str, max_cardinality: int = 40
              ) -> "TabularEncoder":
        """Auto-detect categorical vs numeric columns (target excluded)."""
        cats, nums = [], []
        for col in df.columns:
            if col == target:
                continue
            s = df[col]
            if pd.api.types.is_numeric_dtype(s) and s.nunique() > 12:
                nums.append(col)
            elif s.nunique() <= max_cardinality:
                cats.append(col)
            else:
                nums.append(col)
        return cls(categorical=cats, numeric=nums)

    # ── fit / transform ───────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame) -> "TabularEncoder":
        self._columns = self.numeric + self.categorical
        for col in self.numeric:
            s = pd.to_numeric(df[col], errors="coerce")
            self._num_min[col] = float(s.min())
            self._num_max[col] = float(s.max())
            if pd.api.types.is_integer_dtype(df[col].dropna()):
                self._int_cols.add(col)
        for col in self.categorical:
            self._cat_levels[col] = sorted(df[col].astype(str).unique().tolist())
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        blocks = []
        for col in self.numeric:
            s = pd.to_numeric(df[col], errors="coerce").fillna(
                np.mean([self._num_min[col], self._num_max[col]]))
            lo, hi = self._num_min[col], self._num_max[col]
            rng = (hi - lo) or 1.0
            blocks.append(((s.values - lo) / rng).reshape(-1, 1))
        for col in self.categorical:
            levels = self._cat_levels[col]
            idx = {lvl: i for i, lvl in enumerate(levels)}
            oh = np.zeros((len(df), len(levels)), dtype=np.float32)
            for i, v in enumerate(df[col].astype(str).values):
                oh[i, idx.get(v, 0)] = 1.0
            blocks.append(oh)
        return np.hstack(blocks).astype(np.float32) if blocks else np.zeros((len(df), 0))

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    # ── decode ────────────────────────────────────────────────────────────
    def inverse_transform(self, X: np.ndarray) -> pd.DataFrame:
        out = {}
        cur = 0
        for col in self.numeric:
            lo, hi = self._num_min[col], self._num_max[col]
            rng = (hi - lo) or 1.0
            vals = np.clip(X[:, cur], 0.0, 1.0) * rng + lo
            if col in self._int_cols:
                vals = np.rint(vals).astype(int)
            out[col] = vals
            cur += 1
        for col in self.categorical:
            levels = self._cat_levels[col]
            width = len(levels)
            block = X[:, cur:cur + width]
            picks = block.argmax(axis=1)
            out[col] = [levels[p] for p in picks]
            cur += width
        return pd.DataFrame(out)[self._columns]

    @property
    def dim(self) -> int:
        n = len(self.numeric)
        for col in self.categorical:
            n += len(self._cat_levels[col])
        return n

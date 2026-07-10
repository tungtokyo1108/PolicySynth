"""Bundled sample datasets, so the quickstart runs with zero downloads.

    >>> from policysynth.datasets import load_telco_sample
    >>> df = load_telco_sample()

The Telco Customer Churn sample is a public IBM benchmark (7,043 rows). Swap it
for your own DataFrame the moment you move past the tutorial — nothing in
PolicySynth is specific to this table.
"""
from __future__ import annotations

from importlib import resources

import pandas as pd

__all__ = ["load_telco_sample"]


def load_telco_sample() -> pd.DataFrame:
    """Return the bundled Telco Customer Churn sample as a tidy DataFrame.

    ``customerID`` is dropped and ``TotalCharges`` is coerced to numeric
    (the raw file stores a handful of blanks as strings), so the frame is
    ready to hand straight to :meth:`policysynth.PolicySynth.fit`.
    """
    with resources.files(__package__).joinpath("telco_churn_sample.csv").open() as fh:
        df = pd.read_csv(fh)
    df = df.drop(columns=["customerID"], errors="ignore")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    return df

"""Scoring metrics: Gini, KS, PSI, risk grades, calibration, approval-rate curve."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def gini(y, p, sample_weight=None) -> float:
    y, p = np.asarray(y), np.asarray(p)
    ok = ~(pd.isna(y) | pd.isna(p))
    if len(np.unique(y[ok])) < 2:
        return float("nan")
    w = None if sample_weight is None else np.asarray(sample_weight)[ok]
    return float(2 * roc_auc_score(y[ok], p[ok], sample_weight=w) - 1)


def ks(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def _shares(values: np.ndarray, edges: np.ndarray | None, cats: np.ndarray | None) -> np.ndarray:
    nan = np.isnan(values)
    if cats is not None:
        counts = np.array([(values[~nan] == c).sum() for c in cats] + [nan.sum()], dtype=float)
    else:
        counts = np.r_[np.bincount(np.searchsorted(edges, values[~nan], side="right"), minlength=len(edges) + 1), nan.sum()]
    return counts / max(counts.sum(), 1)


def psi(reference, current, bins: int = 10, max_categories: int = 20) -> float:
    """Population Stability Index. Quantile bins of the reference; low-cardinality values are categories;
    missing values form their own bin."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref_ok = ref[~np.isnan(ref)]
    uniq = np.unique(ref_ok)
    if len(uniq) <= max_categories:
        cats = np.union1d(uniq, np.unique(cur[~np.isnan(cur)]))
        r, c = _shares(ref, None, cats), _shares(cur, None, cats)
    else:
        edges = np.unique(np.quantile(ref_ok, np.linspace(0, 1, bins + 1))[1:-1])
        r, c = _shares(ref, edges, None), _shares(cur, edges, None)
    r, c = np.where(r == 0, 1e-4, r), np.where(c == 0, 1e-4, c)
    return float(np.sum((r - c) * np.log(r / c)))


def fit_grade_edges(train_scores, n_grades: int = 10) -> np.ndarray:
    """Quantile risk-grade boundaries fitted on train predictions only (grade 1 = lowest PD)."""
    q = np.unique(np.quantile(np.asarray(train_scores, dtype=float), np.linspace(0, 1, n_grades + 1)))
    q[0], q[-1] = -np.inf, np.inf
    return q


def apply_grade_edges(scores, edges) -> np.ndarray:
    return np.searchsorted(edges[1:-1], np.asarray(scores, dtype=float), side="right") + 1


def risk_grade_table(y, p, grades, amount=None) -> pd.DataFrame:
    d = pd.DataFrame({"grade": grades, "y": y, "pd": p, "amount": 1.0 if amount is None else amount})
    t = d.groupby("grade").agg(loans=("y", "size"), avg_pd=("pd", "mean"), min_pd=("pd", "min"), max_pd=("pd", "max"),
                               bad_rate=("y", "mean"), bads=("y", "sum")).reset_index()
    t["share"] = t["loans"] / t["loans"].sum()
    t["cum_share"] = t["share"].cumsum()
    t["cum_bad_rate"] = t["bads"].cumsum() / t["loans"].cumsum()
    t["cum_bad_capture"] = t["bads"][::-1].cumsum()[::-1] / t["bads"].sum()   # share of bads in this grade and riskier
    return t


def calibration_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    d = pd.DataFrame({"y": y, "p": p})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), n_bins, labels=False)
    return d.groupby("bin").agg(mean_pd=("p", "mean"), bad_rate=("y", "mean"), n=("y", "size")).reset_index()


def approval_curve(y, p, rates=None) -> pd.DataFrame:
    """Approve the lowest-PD share of applications; bad rate among the approved."""
    rates = np.linspace(0.05, 1.0, 96) if rates is None else np.asarray(rates)
    order = np.argsort(p, kind="stable")
    ys = np.asarray(y)[order]
    cum_bad = np.cumsum(ys)
    k = np.maximum(np.round(rates * len(ys)).astype(int), 1)
    return pd.DataFrame({"approval_rate": rates, "bad_rate": cum_bad[k - 1] / k})


def gini_by_period(df: pd.DataFrame, period_col: str, target: str, scores: list[str], min_bads: int = 20) -> pd.DataFrame:
    rows = []
    for per, g in df.groupby(period_col, observed=True):
        if g[target].sum() < min_bads:
            continue
        rows.append({period_col: per, "n": len(g), "bad_rate": g[target].mean(), **{s: gini(g[target], g[s]) for s in scores}})
    return pd.DataFrame(rows)

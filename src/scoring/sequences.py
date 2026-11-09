"""Raw events -> GRU tokens.

A token is everything that happened to one loan within one minute (e.g. payment + closure).
Every field is computed from information known at the token time only: running loan state
(principal left, paid share, prolongations, current due date) and client portfolio state.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

COMPONENTS = ["main", "percent", "penalty"]
EVENTS = ["issue", "payment", "prolongation", "closure", "cancel", "autodebit_fail"]
CATEGORIES = ["loan", "prolongation", "recurrent"]
DAY = pd.Timedelta(days=1)


def slog(x):
    """Signed log1p: compresses amounts/days and keeps the sign."""
    x = np.asarray(x, dtype="float64")
    return np.sign(x) * np.log1p(np.abs(x))


def _days(a, b) -> np.ndarray:
    return ((a - b) / DAY).to_numpy(dtype=float)


def build_tokens(events: pd.DataFrame, loans: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (meta, features): one row per token, sorted by client and time."""
    contract = loans[["loan_id", "amount", "term", "issue_time", "first_due_date", "loan_no"]].rename(columns={"amount": "loan_sum"})
    rows = events.drop(columns=["amount"]).merge(contract, on="loan_id")
    rows = rows.sort_values(["loan_id", "event_time"], kind="mergesort").reset_index(drop=True)

    # running per-loan state on the row level
    rows["plan_end_after"] = rows["plan_end_after"].where(rows["event"].isin(["issue", "prolongation"]))
    rows["plan_end_after"] = rows.groupby("loan_id", sort=False)["plan_end_after"].ffill()
    rows["plan_end_before"] = rows.groupby("loan_id", sort=False)["plan_end_after"].shift(1)
    rows["cum_main"] = rows.groupby("loan_id", sort=False)["amt_main"].cumsum()
    rows["cum_paid"] = rows[["amt_main", "amt_percent", "amt_penalty"]].sum(axis=1).groupby(rows["loan_id"], sort=False).cumsum()
    rows["n_prol"] = rows["event"].eq("prolongation").astype(int).groupby(rows["loan_id"], sort=False).cumsum()

    # aggregate rows into tokens (loan, minute)
    rows["tmin"] = rows["event_time"].dt.floor("min")
    for e in EVENTS:
        rows[f"ev_{e}"] = rows["event"].eq(e).astype("int8")
    for c in CATEGORIES:
        rows[f"cat_{c}"] = rows["category"].eq(c).astype("int8")
    agg = {"client_id": "first", "event_time": "max", "plan_end_before": "first", "plan_end_after": "last",
           "cum_main": "last", "cum_paid": "last", "n_prol": "last", "loan_sum": "first", "term": "first",
           "issue_time": "first", "first_due_date": "first", "loan_no": "first"}
    agg |= {f"amt_{c}": "sum" for c in COMPONENTS}
    agg |= {c: "max" for c in rows.columns if c.startswith(("ev_", "cat_"))}
    tok = rows.groupby(["loan_id", "tmin"], sort=False).agg(agg).reset_index()
    tok = tok.sort_values(["client_id", "event_time", "loan_id"], kind="mergesort").reset_index(drop=True)

    # per-client portfolio state at the token
    gc = tok.groupby("client_id", sort=False)
    delta = tok["ev_issue"].astype(int) - tok["ev_closure"].astype(int)
    tok["open_loans"] = delta.groupby(tok["client_id"], sort=False).cumsum().clip(lower=0)
    tok["dt_prev_days"] = _days(tok["event_time"], gc["event_time"].shift(1))

    t = tok["event_time"]
    plan_before = tok["plan_end_before"].fillna(tok["plan_end_after"])
    loan_sum = tok["loan_sum"].where(tok["loan_sum"] > 0)
    feats = pd.DataFrame({
        "log_loan_sum": np.log1p(tok["loan_sum"].to_numpy()),
        "log_loan_term": np.log1p(tok["term"].to_numpy()),
        "loan_age": slog(_days(t, tok["issue_time"])),
        "dpd_at_token": slog(_days(t, plan_before)),            # > 0: overdue vs the current due date
        "days_to_due_after": slog(_days(tok["plan_end_after"], t)),
        "ext_days": slog(_days(tok["plan_end_after"], plan_before)),  # due-date shift made by this token
        "late_vs_first_plan": slog(_days(t, tok["first_due_date"])),
        "principal_left_ratio": (1 - tok["cum_main"] / loan_sum).clip(-1, 2).to_numpy(),
        "paid_ratio": (tok["cum_paid"] / loan_sum).clip(-1, 10).to_numpy(),
        "n_prol": np.log1p(tok["n_prol"].to_numpy()),
        "open_loans": np.log1p(tok["open_loans"].to_numpy()),
        "loan_no": np.log1p(tok["loan_no"].to_numpy()),
        "dt_prev": slog(np.nan_to_num(tok["dt_prev_days"].to_numpy(), nan=-1.0)),
        "dow_sin": np.sin(2 * np.pi * t.dt.day_of_week / 7), "dow_cos": np.cos(2 * np.pi * t.dt.day_of_week / 7),
        "hour_sin": np.sin(2 * np.pi * t.dt.hour / 24), "hour_cos": np.cos(2 * np.pi * t.dt.hour / 24),
    })
    for c in COMPONENTS:
        feats[f"amt_{c}"] = slog((tok[f"amt_{c}"] / loan_sum).fillna(0).to_numpy() * 100)
    for c in [c for c in tok.columns if c.startswith(("ev_", "cat_"))]:
        feats[c] = tok[c].to_numpy()
    feats = feats.astype("float32").fillna(0.0)
    meta = tok[["client_id", "loan_id", "event_time"]].copy()
    return meta, feats


@dataclass
class SequenceStore:
    """Flat token matrix + per-application slice [seq_start, seq_end) of tokens strictly before the application."""

    x: np.ndarray            # (n_tokens, n_fields) standardized
    t: np.ndarray            # token time, seconds
    seq_start: np.ndarray    # per application
    seq_end: np.ndarray
    app_t: np.ndarray        # application time, seconds
    fields: list[str]
    mu: np.ndarray
    sd: np.ndarray

    @property
    def lengths(self) -> np.ndarray:
        return self.seq_end - self.seq_start


def make_store(meta: pd.DataFrame, feats: pd.DataFrame, apps: pd.DataFrame, fit_before: pd.Timestamp) -> SequenceStore:
    """Index tokens per application. Standardization statistics use tokens before ``fit_before`` only."""
    tsec = ((meta["event_time"] - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).to_numpy(dtype=np.int64)
    clients = meta["client_id"].to_numpy()
    starts = np.r_[0, np.flatnonzero(np.diff(clients)) + 1]
    ends = np.r_[starts[1:], len(meta)]
    first = pd.Series(starts, index=clients[starts])
    last = pd.Series(ends, index=clients[starts])
    app_t = ((apps["application_date"] - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).to_numpy(dtype=np.int64)
    s = first.reindex(apps["client_id"]).fillna(0).to_numpy(dtype=np.int64)
    e = last.reindex(apps["client_id"]).fillna(0).to_numpy(dtype=np.int64)
    seq_end = s.copy()
    for i in range(len(apps)):            # tokens are sorted by time inside a client
        seq_end[i] = s[i] + np.searchsorted(tsec[s[i]:e[i]], app_t[i], side="left")   # strictly before the application
    x = feats.to_numpy(dtype=np.float32)
    fit = tsec < (fit_before - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)
    mu, sd = x[fit].mean(0), x[fit].std(0) + 1e-6
    binary = [i for i, c in enumerate(feats.columns) if c.startswith(("ev_", "cat_"))]
    mu[binary], sd[binary] = 0.0, 1.0
    return SequenceStore(x=((x - mu) / sd).astype(np.float32), t=tsec, seq_start=s, seq_end=seq_end, app_t=app_t,
                         fields=list(feats.columns), mu=mu, sd=sd)

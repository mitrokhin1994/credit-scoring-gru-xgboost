"""Tabular features for every application, strictly point-in-time.

Internal history features are derived from the raw event stream only (never from generator internals):
events -> per-loan summary -> aggregates over previous loans closed before the application.
External snapshots (bureau, application form, digital footprint) are joined as they are.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS = [30, 60, 90, 180, 365, None]          # None = whole history
HALF_LIFE_DAYS = 60.0
DAY = pd.Timedelta(days=1)

GROUPS = {
    "hist": "Internal credit history",
    "bki": "Credit bureau",
    "app": "Application form and loan request",
    "dig": "Digital footprint",
    "coll": "Collections",
}


def _days(a, b) -> pd.Series | np.ndarray:
    d = (a - b) / DAY
    return d.astype(float) if isinstance(d, pd.Series) else np.asarray(d, dtype=float)


def loan_summary(loans: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """One row per loan: contract + outcome derived from its events (known once the loan is closed)."""
    ev = events
    pay = ev[ev["event"].isin(["payment", "cancel"])]
    is_pay = pay["event"].eq("payment").to_numpy()
    close = ev.loc[ev["event"].eq("closure"), ["loan_id", "event_time"]].rename(columns={"event_time": "close_time"})
    due = (ev.loc[ev["event"].isin(["issue", "prolongation"])].groupby("loan_id")["plan_end_after"].max()
           .rename("final_due"))
    n_prol = ev[ev["event"].eq("prolongation")].groupby("loan_id").size().rename("n_prol")
    p = pay.merge(close, on="loan_id", how="left")
    at_close = p["event_time"].eq(p["close_time"]).to_numpy()
    agg = pd.DataFrame({
        "loan_id": p["loan_id"].to_numpy(),
        "pay": is_pay.astype(int),
        "partial": (is_pay & p["category"].eq("loan").to_numpy() & ~at_close).astype(int),
        "cancel": (~is_pay).astype(int),
        "night": (is_pay & (p["event_time"].dt.hour.to_numpy() < 6)).astype(int),
        "amt_main": p["amt_main"].to_numpy(), "amt_percent": p["amt_percent"].to_numpy(),
        "amt_penalty": p["amt_penalty"].to_numpy(),
    }).groupby("loan_id").sum()
    s = (loans[["loan_id", "client_id", "loan_no", "issue_time", "amount", "term", "first_due_date"]]
         .merge(close, on="loan_id", how="left").merge(due, on="loan_id", how="left")
         .merge(n_prol, on="loan_id", how="left").merge(agg, on="loan_id", how="left"))
    s["n_prol"] = s["n_prol"].fillna(0)
    s[list(agg.columns)] = s[list(agg.columns)].fillna(0)
    close_day, due_day = s["close_time"].dt.normalize(), s["final_due"].dt.normalize()
    s["dpd"] = _days(close_day, due_day).clip(lower=0)
    s["early_days"] = _days(due_day, close_day).clip(lower=0)
    s["duration"] = _days(s["close_time"], s["issue_time"])
    s["night_share"] = s["night"] / s["pay"].replace(0, np.nan)
    return s


def _windowed(pairs: pd.DataFrame, w: int | None) -> pd.DataFrame:
    sub = pairs if w is None else pairs[pairs["age_days"] <= w]
    sfx = "all" if w is None else f"{w}d"
    out = sub.groupby("application_id").agg(
        loans_cnt=("loan_id", "size"), amount_sum=("amount", "sum"), amount_mean=("amount", "mean"),
        amount_max=("amount", "max"), amount_min=("amount", "min"), term_mean=("term", "mean"),
        dpd_max=("dpd", "max"), dpd_mean=("dpd", "mean"), overdue_share=("overdue", "mean"),
        dpd5_share=("dpd5", "mean"), dpd15_share=("dpd15", "mean"), dpd30_cnt=("dpd30", "sum"),
        prol_sum=("n_prol", "sum"), prol_share=("prolonged", "mean"), prol_max=("n_prol", "max"),
        early_share=("early", "mean"), early_days_mean=("early_days", "mean"),
        penalty_sum=("amt_penalty", "sum"), partial_mean=("partial", "mean"), cancel_sum=("cancel", "sum"),
        night_share_mean=("night_share", "mean"), duration_mean=("duration", "mean"), payments_mean=("pay", "mean"),
        percent_sum=("amt_percent", "sum"),
    )
    out["penalty_ratio"] = out["penalty_sum"] / out["amount_sum"]
    out["percent_ratio"] = out.pop("percent_sum") / out["amount_sum"]
    zero = ["loans_cnt", "amount_sum", "dpd30_cnt", "prol_sum", "penalty_sum", "cancel_sum"]
    return out.add_prefix("hist_").add_suffix(f"_{sfx}"), [f"hist_{c}_{sfx}" for c in zero]


def history_features(apps: pd.DataFrame, summary: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """~210 internal-history features. Uses only loans closed and events posted before the application."""
    a = apps[["application_id", "client_id", "application_date", "loan_no", "requested_amount", "requested_term"]]
    app_ids = pd.Index(a["application_id"], name="application_id")
    pairs = a.merge(summary, on="client_id", how="inner")
    pairs = pairs[pairs["close_time"] < pairs["application_date"]]
    pairs = pairs.sort_values(["application_id", "issue_time"], kind="mergesort").reset_index(drop=True)
    pairs["age_days"] = _days(pairs["application_date"], pairs["issue_time"])
    pairs["overdue"] = (pairs["dpd"] > 0).astype(int)
    pairs["dpd5"], pairs["dpd15"], pairs["dpd30"] = ((pairs["dpd"] > x).astype(int) for x in (5, 15, 29))
    pairs["prolonged"] = (pairs["n_prol"] > 0).astype(int)
    pairs["early"] = (pairs["early_days"] > 0).astype(int)

    blocks, zero_cols = [], []
    for w in WINDOWS:
        b, z = _windowed(pairs, w)
        blocks.append(b.reindex(app_ids))
        zero_cols += z
    x: dict[str, pd.Series] = {}   # every extra feature is an index-aligned Series

    g = pairs.groupby("application_id")
    last, prev = g.nth(-1).set_index("application_id"), g.nth(-2).set_index("application_id")
    for src, name in {"amount": "amount", "term": "term", "dpd": "dpd", "n_prol": "prol", "early_days": "early_days",
                      "partial": "partial", "night_share": "night_share", "duration": "duration", "cancel": "cancel"}.items():
        x[f"hist_last_{name}"] = last[src]
    x["hist_last_penalty_ratio"] = last["amt_penalty"] / last["amount"]
    x["hist_last_percent_ratio"] = last["amt_percent"] / last["amount"]
    x["hist_trend_dpd_last2"] = last["dpd"] - prev["dpd"]
    x["hist_trend_prol_last2"] = last["n_prol"] - prev["n_prol"]
    x["hist_trend_early_last2"] = last["early_days"] - prev["early_days"]
    x["hist_trend_amount_last2"] = last["amount"] / prev["amount"]
    x["hist_amount_growth_all"] = last["amount"] / g["amount"].first()

    # the last three loans (a window in loans rather than in days)
    last3 = pairs[g.cumcount(ascending=False) < 3].groupby("application_id")
    for src, name in {"dpd": "dpd", "n_prol": "prol", "early_days": "early_days", "night_share": "night_share"}.items():
        x[f"hist_{name}_mean_last3"] = last3[src].mean()

    # slope of DPD over the loan index and time-weighted means (half-life 60 days)
    pairs["i"] = g.cumcount().astype(float)
    pairs["w"] = 0.5 ** (pairs["age_days"] / HALF_LIFE_DAYS)
    for c in ["dpd", "n_prol", "overdue", "early_days"]:
        pairs[f"w_{c}"] = pairs["w"] * pairs[c]
    pairs["i2"], pairs["idpd"] = pairs["i"] ** 2, pairs["i"] * pairs["dpd"]
    sm = pairs.groupby("application_id")[["i", "i2", "dpd", "idpd", "w", "w_dpd", "w_n_prol", "w_overdue", "w_early_days"]].sum()
    n = g.size()
    x["hist_dpd_slope"] = ((sm["idpd"] - sm["i"] * sm["dpd"] / n) / (sm["i2"] - sm["i"] ** 2 / n)).where(n >= 3)
    for c, name in [("w_dpd", "dpd"), ("w_n_prol", "prol"), ("w_overdue", "overdue"), ("w_early_days", "early_days")]:
        x[f"hist_tw_{name}_mean"] = sm[c] / sm["w"]

    # pauses between loans
    pairs["gap"] = _days(pairs["issue_time"], g["close_time"].shift(1))
    x["hist_gap_mean_days"] = pairs.groupby("application_id")["gap"].mean()
    x["hist_gap_max_days"] = pairs.groupby("application_id")["gap"].max()
    x["hist_overdue_share_of_prolonged"] = pairs[pairs["prolonged"].eq(1)].groupby("application_id")["overdue"].mean()

    # recency (days before the application)
    app_date = a.set_index("application_id")["application_date"]
    for name, s in {
        "hist_days_since_last_close": g["close_time"].max(),
        "hist_days_since_last_issue": g["issue_time"].max(),
        "hist_days_since_first_loan": g["issue_time"].min(),
        "hist_days_since_last_overdue": pairs[pairs["overdue"].eq(1)].groupby("application_id")["close_time"].max(),
        "hist_days_since_last_prol": pairs[pairs["prolonged"].eq(1)].groupby("application_id")["close_time"].max(),
    }.items():
        x[name] = _days(app_date.reindex(s.index), s)
    x["hist_loans_per_month"] = n / (x["hist_days_since_first_loan"] / 30.44).clip(lower=1)

    # payments, cancellations, prolongations in calendar windows before the application
    ev = a[["application_id", "client_id", "application_date"]].merge(
        events[events["event"].isin(["payment", "cancel", "prolongation"])], on="client_id")
    ev = ev[ev["event_time"] < ev["application_date"]]
    is_pay = ev["event"].eq("payment").to_numpy()
    pe = pd.DataFrame({"application_id": ev["application_id"].to_numpy(),
                       "age": _days(ev["application_date"], ev["event_time"]).to_numpy(), "pay": is_pay,
                       "amount": np.where(is_pay, ev["amount"].to_numpy(), 0.0),
                       "night": is_pay & (ev["event_time"].dt.hour.to_numpy() < 6),
                       "cancel": ev["event"].eq("cancel").to_numpy(), "prol": ev["event"].eq("prolongation").to_numpy(),
                       "penalty": ev["amt_penalty"].to_numpy()})
    for w in (7, 30, 90):
        sub = pe[pe["age"] <= w].groupby("application_id")
        x[f"hist_pay_cnt_{w}d"] = sub["pay"].sum()
        if w >= 30:
            x[f"hist_pay_sum_{w}d"] = sub["amount"].sum()
    sub = pe[pe["age"] <= 90].groupby("application_id")
    for c in ["night", "cancel", "prol", "penalty"]:
        x[f"hist_{c}_events_90d"] = sub[c].sum()

    feats = pd.concat(blocks + [pd.DataFrame(x).reindex(app_ids)], axis=1)
    zero_cols += [c for c in x if c.startswith(("hist_pay_", "hist_night_events", "hist_cancel_events",
                                                 "hist_prol_events", "hist_penalty_events"))]
    feats = feats.fillna({c: 0 for c in zero_cols})

    # current application (known at application time)
    ai = a.set_index("application_id")
    cur = pd.DataFrame({
        "app_loan_number": ai["loan_no"], "app_requested_amount": ai["requested_amount"],
        "app_requested_term": ai["requested_term"],
        "app_amount_to_last_ratio": ai["requested_amount"] / feats["hist_last_amount"],
        "app_amount_to_max_ratio": ai["requested_amount"] / feats["hist_amount_max_all"],
        "app_amount_to_mean_ratio": ai["requested_amount"] / feats["hist_amount_mean_all"],
    })
    return pd.concat([feats, cur], axis=1).copy().reset_index()


def build_features(apps: pd.DataFrame, loans: pd.DataFrame, events: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    hist = history_features(apps, loan_summary(loans, events), events)
    feats = hist.merge(external, on="application_id", how="left", validate="one_to_one")
    return feats.assign(app_amount_to_income=feats["app_requested_amount"] / feats["app_income"],
                        bki_debt_to_income=feats["bki_total_debt"] / (12 * feats["app_income"]))


def feature_catalog(columns) -> pd.DataFrame:
    cols = [c for c in columns if c.split("_")[0] in GROUPS]
    cat = pd.DataFrame({"feature": cols, "group": [c.split("_")[0] for c in cols]})
    cat["group_name"] = cat["group"].map(GROUPS)
    return cat


def pit_recompute(app_ids, apps: pd.DataFrame, loans: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Recompute history features of given applications on history truncated at the application time.

    If the batch calculation used anything from the future (or mixed rows between applications),
    the two results would differ.
    """
    out = []
    a_idx = apps.set_index("application_id")
    for app_id in app_ids:
        row = a_idx.loc[[app_id]].reset_index()
        cid, t = row["client_id"].iat[0], row["application_date"].iat[0]
        ev = events[(events["client_id"] == cid) & (events["event_time"] < t)]
        ln = loans[(loans["client_id"] == cid) & (loans["application_date"] < t)]
        out.append(history_features(row, loan_summary(ln, ev), ev))
    return pd.concat(out, ignore_index=True)

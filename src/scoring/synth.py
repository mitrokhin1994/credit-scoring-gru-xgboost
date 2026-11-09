"""Synthetic portfolio of repeat borrowers of a short-term consumer lender.

Generative story
----------------
* Every client has a latent credit risk ``z`` that evolves from loan to loan:
  client-specific drift + random walk + rare shocks (job loss, recovery). ``z`` is never observed.
* Every loan's behaviour is a noisy observation of the current ``z``: early repayment,
  prolongations, overdue and its duration, partial payments, reversed payments, payment hours.
  Habits that are unrelated to risk (some people always pay early, some pay at night) add noise.
* On top of ``z`` a hidden *liquidity stress* ``l`` (AR(1) over loans) raises the default risk. It barely
  changes loan-level outcomes but shows in the micro-dynamics inside a loan: the repayment goes through
  a failed auto-debit in the morning of the due date and a partial auto-debit collection, payments move
  to late evening, prolongations happen at the very last moment, an early partial payment slides towards
  the due date. A loan-level feature factory (counts, sums, DPD, shares) cannot see this; a GRU over
  raw tokens can - the same reason the approach pays off on real data.
* The lender stores only the raw event stream: issue / payment / prolongation / closure / cancel /
  autodebit_fail.
* Default of the next loan (NPL30d90) depends on the *current* ``z`` and ``l`` plus external credit
  stress ``e``, the loan terms and a macro factor.
* Credit-bureau snapshots are driven mostly by ``e``; application-form fields by the static part
  of the risk; digital footprint is weak and mostly noise. A few deliberate traps (leakage,
  drifting fields, near-empty and constant fields, duplicates) test the feature-selection pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import HISTORY_START, OOT_START, SAMPLE_END, SAMPLE_START, SEED

DAILY_RATE = 0.008      # interest per day, share of principal
PENALTY_RATE = 0.001    # penalty per day of overdue, share of principal
TERMS = np.array([7, 10, 14, 21, 30])
TERM_P = np.array([0.15, 0.15, 0.30, 0.20, 0.20])
MAX_PROL = 5


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _day(ts: str | pd.Timestamp) -> float:
    return (pd.Timestamp(ts) - HISTORY_START) / pd.Timedelta(days=1)


def _round_to(x, step=500):
    return np.round(np.asarray(x, dtype=float) / step) * step


@dataclass
class SynthConfig:
    n_clients: int = 45_000
    seed: int = SEED
    max_loans: int = 40
    # latent risk dynamics (per loan)
    drift_sd: float = 0.12
    rw_sd: float = 0.30
    shock_p: float = 0.04
    recovery_p: float = 0.03
    # hidden liquidity stress, AR(1) over loans, visible only in micro-dynamics of payments
    liq_rho: float = 0.8
    liq_effect: float = 1.1      # added to the default logit
    # behaviour model
    loan_noise_sd: float = 0.85  # idiosyncratic, unobservable risk of a particular loan
    a_over: float = -0.7         # intercept of "goes overdue at due date"
    b_prol: float = -0.9         # intercept of "prolongs instead of repaying"
    m_dpd: float = float(np.log(9.0))  # log-median overdue duration, days
    dpd_risk_coef: float = 0.5
    write_off_p: float = 0.6     # share of 90+ overdue that is never repaid
    churn_base: float = 0.25
    gap_mean: float = 18.0       # days between closure and the next application


class _Events:
    """Column buffer for the raw event stream."""

    def __init__(self):
        self.parts: list[dict[str, np.ndarray]] = []

    def add(self, loan_id, t, event, main=0.0, percent=0.0, penalty=0.0, category="", plan_end=np.nan):
        n = len(loan_id)
        if n == 0:
            return
        f = lambda v: np.broadcast_to(np.asarray(v, dtype=float), (n,)).copy()
        self.parts.append({
            "loan_id": np.asarray(loan_id, dtype=np.int64), "t": f(t), "event": np.full(n, event, dtype=object),
            "amt_main": f(main), "amt_percent": f(percent), "amt_penalty": f(penalty),
            "category": np.full(n, category, dtype=object), "plan_end_after": f(plan_end),
        })

    def frame(self) -> pd.DataFrame:
        cols = self.parts[0].keys()
        return pd.DataFrame({c: np.concatenate([p[c] for p in self.parts]) for c in cols})


def _clock(rng, n, p_night):
    """Fraction of day: night hours (0-5) with probability p_night, otherwise daytime."""
    night = rng.random(n) < p_night
    hour = np.where(night, rng.integers(0, 6, n), np.clip(np.round(rng.normal(14.5, 3.5, n)), 7, 23))
    return (hour + rng.integers(0, 60, n) / 60.0) / 24.0


def _settle_clock(rng, liq, hour_habit):
    """Hour of a payment that settles an obligation (closure, prolongation): the more liquidity stress,
    the closer to the end of the day the money is found. Kept within 07-24h (not counted as 'night')."""
    hour = np.clip(13.0 + 3.6 * liq + 1.0 * hour_habit + rng.normal(0, 1.1, len(liq)), 7.0, 23.95)
    return hour / 24.0


def _clients(cfg: SynthConfig, rng) -> pd.DataFrame:
    n = cfg.n_clients
    age = np.clip(np.round(19 + rng.gamma(3.0, 6.0, n)), 21, 75)
    education = rng.choice(5, n, p=[0.06, 0.34, 0.30, 0.26, 0.04])
    employment = rng.choice(5, n, p=[0.60, 0.14, 0.10, 0.09, 0.07])  # employed, self-empl., unempl., retired, student
    children = np.minimum(rng.poisson(0.8, n), 5)
    region = rng.integers(1, 86, n)
    region_effect = rng.normal(0, 0.25, 87)[region]
    log_income = (10.55 + 0.012 * (age - 35) - 0.0004 * (age - 35) ** 2 + 0.07 * education
                  + np.array([0.0, 0.10, -0.60, -0.35, -0.50])[employment] + rng.normal(0, 0.42, n))
    z_static = (-0.45 * (log_income - 10.5) + 0.35 * (age < 25) - 0.06 * education
                + np.array([0.0, 0.15, 0.55, -0.10, 0.25])[employment] + region_effect + 0.05 * children)
    z0 = z_static + 0.85 * rng.normal(size=n)
    device_os = np.where(rng.random(n) < sigmoid(-0.9 + 0.9 * (log_income - 10.5) - 0.3 * z0), 1, 0)  # 1 = iOS
    # the lender grows: half of the clients arrive uniformly, half with a density rising in time
    span = _day(SAMPLE_END) - 30
    first_day = np.where(rng.random(n) < 0.5, rng.uniform(0, span, n), span * np.sqrt(rng.random(n)))
    return pd.DataFrame({
        "client_id": np.arange(1, n + 1),
        "age0": age, "gender": rng.integers(0, 2, n), "education": education, "employment": employment,
        "marital": rng.choice(4, n, p=[0.38, 0.42, 0.15, 0.05]), "children": children, "region": region,
        "log_income": log_income, "device_os": device_os,
        "housing": rng.choice(3, n, p=[0.45, 0.35, 0.20]), "has_car": (rng.random(n) < sigmoid(-1 + 1.2 * (log_income - 10.5))).astype(int),
        "phone_age_m": np.round(np.clip(rng.gamma(2.0, 30.0, n) * np.exp(-0.15 * z0), 1, 240)),
        "z0": z0, "e": 0.35 * z0 + 0.9 * rng.normal(size=n), "drift": rng.normal(0, cfg.drift_sd, n),
        "early_habit": rng.normal(0, 0.6, n), "night_habit": rng.normal(0, 0.8, n), "prol_habit": rng.normal(0, 0.5, n),
        "hour_habit": rng.normal(0, 1.0, n), "first_day": first_day,
    })


def _simulate_loans(cfg: SynthConfig, cl: pd.DataFrame, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(cl)
    end_day, wave_day = _day(SAMPLE_END), _day("2025-10-01")
    z = cl["z0"].to_numpy().copy()
    e, drift = cl["e"].to_numpy(), cl["drift"].to_numpy()
    early_habit, night_habit, prol_habit, hour_habit = (
        cl[c].to_numpy() for c in ("early_habit", "night_habit", "prol_habit", "hour_habit"))
    log_income = cl["log_income"].to_numpy()
    t_app = np.floor(cl["first_day"].to_numpy()) + _clock(rng, n, sigmoid(-2.4 + 0.5 * z + night_habit))
    active = np.ones(n, bool)
    liq = rng.normal(size=n)          # liquidity stress at the first loan
    limit = _round_to(rng.uniform(3000, 10000, n))
    prev_amount = np.full(n, np.nan)
    ev, loans, next_id = _Events(), [], 1

    for k in range(1, cfg.max_loans + 1):
        idx = np.flatnonzero(active & (t_app <= end_day))
        m = len(idx)
        if m == 0:
            break
        if k > 1:
            shock = ((rng.random(m) < cfg.shock_p) * rng.uniform(0.8, 2.0, m)
                     - (rng.random(m) < cfg.recovery_p) * rng.uniform(0.5, 1.5, m))
            z[idx] += drift[idx] + cfg.rw_sd * rng.normal(size=m) + shock
            liq[idx] = cfg.liq_rho * liq[idx] + np.sqrt(1 - cfg.liq_rho ** 2) * rng.normal(size=m)
        zi, ei, li = z[idx], e[idx], liq[idx]
        lid = np.arange(next_id, next_id + m)
        next_id += m
        p_night = sigmoid(-2.4 + 0.55 * zi + night_habit[idx])

        # --- contract
        app = t_app[idx]
        amount = np.clip(np.minimum(_round_to(limit[idx] * rng.uniform(0.55, 1.0, m) * np.exp(0.06 * zi)), limit[idx]), 2000, 50000)
        term = rng.choice(TERMS, m, p=TERM_P)
        term = np.where(rng.random(m) < 0.4 * sigmoid(zi - 1.5), 30, term)
        issue = app + rng.uniform(5, 90, m) / 1440
        issue_day = np.floor(issue)
        first_due = issue_day + term
        stretch = np.where(np.isnan(prev_amount[idx]), 0.0, np.log(amount / prev_amount[idx]))
        macro = 0.10 * np.sin(2 * np.pi * (issue - 45) / 365.25) + 0.22 * np.clip((issue - wave_day) / 92, 0, 1)
        # interactions an additive scorecard cannot express: risky clients asking for more than before,
        # long terms on top of a high external debt load, external debt hurting low incomes most (DTI logic)
        inter = (1.2 * np.clip(stretch, 0, None) * sigmoid(3 * (zi - 0.5))
                 + 0.5 * (term >= 21) * sigmoid(3 * (ei - 0.5))
                 + 0.8 * sigmoid(2.5 * (ei - 0.3)) * sigmoid(-2.5 * (log_income[idx] - 10.3)))
        r_known = zi + 0.45 * ei + cfg.liq_effect * li + 0.35 * stretch + 0.12 * (term == 30) + inter + macro
        r = r_known + rng.normal(0, cfg.loan_noise_sd, m)
        ev.add(lid, issue, "issue", plan_end=first_due + 0.5)

        # --- optional early partial repayment before the first due date
        pp = (rng.random(m) < sigmoid(-1.7 - 0.5 * zi + 0.4 * early_habit[idx])) & (term >= 3)
        # under liquidity stress the money is scraped together right before the due date
        pp_day = issue_day + 1 + np.floor(sigmoid(rng.normal(size=m) + 1.2 * li) * (term - 2))
        pp_main = np.where(pp, _round_to(amount * rng.uniform(0.2, 0.5, m), 100), 0.0)
        ev.add(lid[pp], pp_day[pp] + _clock(rng, pp.sum(), p_night[pp]), "payment", main=pp_main[pp], category="loan")
        principal = amount - pp_main

        # --- due-date cycles: repay / prolong / go overdue
        due, period_start = first_due.astype(float), issue_day.copy()
        status = np.zeros(m, np.int8)            # 0 open, 1 repaid on time / early, 2 overdue
        n_prol = np.zeros(m, int)
        close = np.full(m, np.nan)
        over_due, over_cycle = np.full(m, np.nan), np.full(m, -1)
        for i in range(MAX_PROL + 1):
            io = np.flatnonzero(status == 0)
            if len(io) == 0:
                break
            over = rng.random(len(io)) < sigmoid(cfg.a_over + r[io] + 0.25 * i)
            p_prol = sigmoid(cfg.b_prol + 0.6 * zi[io] + 0.35 * ei[io] + prol_habit[idx][io] + 0.15 * i) if i < MAX_PROL else 0.0
            prol = ~over & (rng.random(len(io)) < p_prol)
            rep = ~over & ~prol

            j = io[rep]
            # under liquidity stress the repayment goes through the auto-debit route on the due date:
            # failed attempt in the morning -> partial collection at midday -> the rest paid manually in the evening
            adp = rng.random(len(j)) < sigmoid(-2.2 + 1.6 * li[j])
            early = np.where(adp, 0, np.minimum(term[j] - 1, rng.poisson(np.exp(1.1 - 0.45 * zi[j] + early_habit[idx][j]))))
            cday = due[j] - early
            cday = np.where(pp[j] & (cday <= pp_day[j]), pp_day[j] + 1, cday)
            ct = cday + _settle_clock(rng, li[j], hour_habit[idx][j])
            ja, da = j[adp], cday[adp]
            if len(ja):
                rec_t = da + rng.uniform(11, 15, len(ja)) / 24
                ev.add(lid[ja], da + rng.uniform(7, 9, len(ja)) / 24, "autodebit_fail", category="recurrent")
                rec_amt = _round_to(principal[ja] * rng.uniform(0.3, 0.7, len(ja)), 100)
                ev.add(lid[ja], rec_t, "payment", main=rec_amt, category="recurrent")
                principal[ja] -= rec_amt
                ct[adp] = np.minimum(np.maximum(ct[adp], rec_t + rng.uniform(1, 3, len(ja)) / 24), da + 0.999)
            close[j] = ct
            status[j] = 1

            j = io[prol]
            lead = np.where(rng.random(len(j)) < sigmoid(-0.8 + 1.5 * li[j]), 0,     # stress: prolong at the last moment
                            np.minimum(term[j] - 1, rng.poisson(np.exp(0.6 - 0.3 * zi[j]))))
            pday = np.maximum(due[j] - lead, period_start[j] + 1)
            pt = pday + _settle_clock(rng, li[j], hour_habit[idx][j])
            new_due = due[j] + term[j]
            ev.add(lid[j], pt, "payment", percent=np.round(principal[j] * DAILY_RATE * (pday - period_start[j])), category="prolongation")
            ev.add(lid[j], pt, "prolongation", plan_end=new_due + 0.5)
            due[j], period_start[j] = new_due, pday
            n_prol[j] += 1

            j = io[over]
            status[j], over_due[j], over_cycle[j] = 2, due[j], i

        # --- overdue: duration, partial payments, write-off
        dpd = np.zeros(m)
        jo = np.flatnonzero(status == 2)
        d = np.clip(np.ceil(np.exp(cfg.m_dpd + cfg.dpd_risk_coef * r[jo] + 1.0 * rng.normal(size=len(jo)))), 1, 400)
        dpd[jo] = d
        wo = np.zeros(m, bool)
        wo[jo] = (d >= 90) & (rng.random(len(jo)) < cfg.write_off_p)
        cured = jo[~wo[jo]]
        close[cured] = over_due[cured] + dpd[cured] + _settle_clock(rng, li[cured], hour_habit[idx][cured])
        horizon = np.minimum(dpd[jo], 120)
        n_part = np.where(horizon > 1, rng.poisson(0.10 * horizon ** 0.65), 0)
        rep_ix = np.repeat(jo, n_part)
        if len(rep_ix):
            h = np.repeat(horizon, n_part)
            off = 1 + np.floor(rng.random(len(rep_ix)) * (h - 1))
            pm = _round_to(amount[rep_ix] * rng.uniform(0.05, 0.15, len(rep_ix)), 100)
            ev.add(lid[rep_ix], over_due[rep_ix] + off + _clock(rng, len(rep_ix), p_night[rep_ix]), "payment",
                   main=pm, penalty=np.round(amount[rep_ix] * PENALTY_RATE * off), category="loan")
            part_main = np.bincount(np.searchsorted(jo, rep_ix), weights=pm, minlength=len(jo))
            principal[jo] = np.maximum(principal[jo] - part_main, 0)
        # auto-debit attempts on the first overdue days fail; sometimes a part of the debt is collected
        n_try = np.minimum(dpd[jo], 3).astype(int)
        rt = np.repeat(jo, n_try)
        if len(rt):
            off = np.arange(len(rt)) - np.repeat(np.cumsum(n_try) - n_try, n_try)
            ev.add(lid[rt], over_due[rt] + off + rng.uniform(7, 9, len(rt)) / 24, "autodebit_fail", category="recurrent")
        coll = jo[(rng.random(len(jo)) < 0.3) & (n_try >= 2)]
        if len(coll):
            off = 1 + np.floor(rng.random(len(coll)) * (np.minimum(dpd[coll], 3) - 1))
            amt = _round_to(principal[coll] * rng.uniform(0.1, 0.4, len(coll)), 100)
            ev.add(lid[coll], over_due[coll] + off + rng.uniform(11, 15, len(coll)) / 24, "payment", main=amt, category="recurrent")
            principal[coll] = np.maximum(principal[coll] - amt, 0)

        # --- closing payment + closure
        jc = np.flatnonzero(~np.isnan(close))
        days_used = np.maximum(np.floor(close[jc]) - period_start[jc], 1)
        rest = np.maximum(principal[jc], 0)
        ev.add(lid[jc], close[jc], "payment", main=rest, percent=np.round(rest * DAILY_RATE * days_used),
               penalty=np.round(amount[jc] * PENALTY_RATE * dpd[jc]), category="loan")
        ev.add(lid[jc], close[jc], "closure")

        # --- reversed (cancelled) payments: a failed attempt to pay
        last_day = np.where(np.isnan(close), over_due + 30, np.floor(close))
        jx = np.flatnonzero((rng.random(m) < sigmoid(-3.0 + 0.6 * zi)) & (last_day - issue_day >= 3))
        if len(jx):
            px = issue_day[jx] + 1 + np.floor(rng.random(len(jx)) * (last_day[jx] - issue_day[jx] - 2))
            pt = px + _clock(rng, len(jx), p_night[jx])
            amt = _round_to(amount[jx] * rng.uniform(0.1, 0.5, len(jx)), 100)
            ev.add(lid[jx], pt, "payment", main=amt, category="loan")
            ev.add(lid[jx], pt + rng.uniform(1, 24, len(jx)) / 24, "cancel", main=-amt, category="loan")

        # --- targets
        npl = (status == 2) & (dpd >= 30) & (over_due + 30 <= issue_day + 90)
        overdue_90d = np.where(status == 2, np.clip(issue_day + 90 - over_due, 0, dpd), 0)
        loans.append(pd.DataFrame({
            "loan_id": lid, "client_id": cl["client_id"].to_numpy()[idx], "loan_no": k, "app_day": app, "issue_day_f": issue,
            "amount": amount, "term": term, "first_due_day": first_due, "limit": limit[idx],
            "npl30d90": npl.astype(int),
            **{f"fpd{x}": ((status == 2) & (over_cycle == 0) & (dpd >= x)).astype(int) for x in (10, 15, 30)},
            "z": zi, "liq": li, "r": r, "r_known": r_known, "dpd": dpd, "n_prol": n_prol,
            "written_off": wo.astype(int),
            "coll_calls_90d": rng.poisson(0.3 * overdue_90d + 0.2),
        }))

        # --- next application of the same client
        closed = ~np.isnan(close)
        cont = closed & (rng.random(m) >= cfg.churn_base + 0.12 * (dpd > 0))
        nxt = np.floor(close + 0.25 + rng.exponential(cfg.gap_mean * np.exp(0.3 * rng.normal(size=m)))) + _clock(rng, m, p_night)
        nxt = np.maximum(nxt, close + 1 / 24)
        t_app[idx] = np.where(cont, nxt, np.inf)
        active[idx[~cont]] = False
        limit[idx] = np.clip(_round_to(limit[idx] * np.where(dpd == 0, 1.3, np.where(dpd <= 5, 1.0, 0.7))), 2000, 50000)
        prev_amount[idx] = amount

    return pd.concat(loans, ignore_index=True), ev.frame()


def _to_ts(days) -> pd.Series:
    return (HISTORY_START + pd.to_timedelta(np.asarray(days, dtype=float), unit="D")).floor("min")


def _external_snapshots(apps: pd.DataFrame, cl: pd.DataFrame, rng) -> pd.DataFrame:
    """Bureau, application form and digital footprint at application time (+ deliberate traps)."""
    a = apps.merge(cl, on="client_id", how="left")
    n = len(a)
    z, e, inc = a["z"].to_numpy(), a["e"].to_numpy(), a["log_income"].to_numpy() - 10.5
    t = a["app_day"].to_numpy()
    years = (t - a["first_day"].to_numpy()) / 365.25
    age = a["age0"].to_numpy() + np.floor(years)
    N = lambda s=1.0: rng.normal(0, s, n)
    P = lambda lam: rng.poisson(np.clip(lam, 0, None))
    out = {"application_id": a["loan_id"].to_numpy()}

    # ---------------- credit bureau ----------------
    hit = rng.random(n) >= 0.05 + 0.10 * (age < 23)
    inq30 = P(np.exp(0.2 + 0.45 * e + 0.25 * z + 0.2 * a["liq"].to_numpy() + N(0.1)))
    inq90 = inq30 + P(np.exp(0.5 + 0.35 * e))
    inq180 = inq90 + P(np.exp(0.6 + 0.30 * e))
    mfo_open = P(np.exp(-0.3 + 0.6 * e + 0.2 * z))
    bank_open = P(np.exp(0.2 - 0.25 * e + 0.4 * inc))
    cards = P(np.exp(-0.1 + 0.3 * inc - 0.1 * e))
    total_debt = np.round(np.exp(10.2 + 0.6 * inc + 0.25 * e + N(0.7)) * ((mfo_open + bank_open) > 0), -2)
    s_status = 0.9 * e + 0.4 * z + N(0.8)
    worst12 = np.digitize(s_status, [0.6, 1.3, 1.8, 2.3])
    worst24 = np.maximum(worst12, np.digitize(s_status + 0.3 + N(0.5), [0.6, 1.3, 1.8, 2.3]))
    worst6 = np.minimum(worst12, np.digitize(s_status - 0.3 + N(0.5), [0.6, 1.3, 1.8, 2.3]))
    monthly_payment = total_debt * rng.uniform(0.04, 0.10, n)
    bki_score = np.clip(np.round(650 - 70 * e - 25 * z + N(35)), 300, 850)
    bki = {
        "bki_inq_7d": rng.binomial(inq30, 0.3), "bki_inq_30d": inq30, "bki_inq_90d": inq90, "bki_inq_180d": inq180,
        "bki_inq_mfo_30d": rng.binomial(inq30, 0.6), "bki_inq_bank_90d": rng.binomial(inq90, 0.35),
        "bki_mfo_open_cnt": mfo_open, "bki_mfo_closed_12m": P(np.exp(0.8 + 0.3 * e)), "bki_bank_open_cnt": bank_open,
        "bki_cards_cnt": cards, "bki_consumer_loans_cnt": P(np.exp(-0.2 + 0.2 * e + 0.3 * inc)),
        "bki_card_utilization": np.where(cards > 0, rng.beta(2 + 2 * sigmoid(e), 3), np.nan),
        "bki_total_debt": total_debt, "bki_total_debt_k": np.round(total_debt / 1000),  # duplicate in other units
        "bki_mfo_debt": np.round(total_debt * rng.beta(1 + 2 * sigmoid(e), 3) * (mfo_open > 0), -2),
        "bki_overdue_debt": np.round(total_debt * (rng.random(n) < sigmoid(-2.2 + 1.1 * e + 0.3 * z)) * rng.uniform(0.05, 0.6, n), -2),
        "bki_worst_status_6m": worst6, "bki_worst_status_12m": worst12, "bki_worst_status_24m": worst24,
        "bki_max_overdue_days_12m": np.select([worst12 == 0, worst12 == 1, worst12 == 2, worst12 == 3],
                                              [0, rng.integers(1, 30, n), rng.integers(30, 60, n), rng.integers(60, 90, n)],
                                              rng.integers(90, 365, n)),
        "bki_overdue_cnt_12m": np.where(worst12 > 0, 1 + P(0.5 * worst12 + 0.3 * sigmoid(e)), 0),
        "bki_overdue_30_cnt_24m": np.where(worst24 >= 2, 1 + P(0.4 * worst24), 0),
        "bki_months_since_overdue": np.where(worst24 > 0, np.round(rng.exponential(8.0 / (1 + worst12), n), 1), np.nan),
        "bki_closed_with_overdue_share": rng.beta(1 + 3 * sigmoid(e + 0.3 * z), 5),
        "bki_dti": np.clip(monthly_payment / np.exp(inc + 10.5), 0, 3),
        "bki_payment_load": np.round(monthly_payment, -1),
        "bki_score": bki_score,
        "bki_history_len_m": np.round(np.clip((age - 18) * 12 * rng.uniform(0.15, 0.8, n), 0, None)),
        "bki_new_accounts_30d": P(np.exp(-1.0 + 0.5 * e)), "bki_new_accounts_90d": P(np.exp(-0.4 + 0.5 * e)),
        "bki_mortgage_flag": (rng.random(n) < sigmoid(-2.5 + 0.8 * inc + 0.03 * (age - 35))).astype(int),
        "bki_auto_loan_flag": (rng.random(n) < sigmoid(-2.2 + 0.6 * inc)).astype(int),
        "bki_avg_loan_amount": np.round(np.exp(10.5 + 0.5 * inc + N(0.5)), -2),
        "bki_limit_total": np.round(cards * np.exp(11 + 0.6 * inc + N(0.5)), -3),
        "bki_active_share": rng.beta(2 + sigmoid(e), 3),
        "bki_restructured_flag": (rng.random(n) < sigmoid(-4 + 1.0 * e)).astype(int),
        "bki_court_cases_flag": (rng.random(n) < sigmoid(-4.5 + 0.8 * e + 0.3 * z)).astype(int),
        "bki_guarantor_flag": (rng.random(n) < 0.03).astype(int),
        "bki_months_since_last_loan": np.round(rng.exponential(3 + 4 * sigmoid(-e), n), 1),
        "bki_mfo_share_of_debt": rng.beta(1 + 2 * sigmoid(e), 2 + inc.clip(-1, 1)),
        "bki_inq_growth_30_180": inq30 / (inq180 + 1),
    }
    # vendor changed the score scale in Oct-2025 -> field drifts (must be caught by PSI)
    bki["bki_score_v2"] = np.where(t < _day("2025-10-01"), bki_score, np.round((bki_score - 300) / 5.5, 1))
    out.update({c: np.where(hit, v, np.nan) for c, v in bki.items()})
    out["bki_no_hit_flag"] = (~hit).astype(int)

    # ---------------- application form ----------------
    hour = np.floor((t % 1) * 24)
    channel = np.where(rng.random(n) < sigmoid(-1.8 + 0.3 * z), 2, np.where(rng.random(n) < 0.55, 1, 0))  # web, app, partner
    out.update({
        "app_age": age, "app_age_months": age * 12 + np.floor((years % 1) * 12),  # near-duplicate of app_age
        "app_gender": a["gender"].to_numpy(), "app_marital_status": a["marital"].to_numpy(),
        "app_children_cnt": a["children"].to_numpy(), "app_education": a["education"].to_numpy(),
        "app_employment_type": a["employment"].to_numpy(),
        "app_employment_months": np.where(np.isin(a["employment"], [0, 1]), np.round(rng.gamma(2, 18, n) * np.exp(-0.1 * z)), 0),
        "app_income": np.round(np.exp(inc + 10.5 + N(0.05)), -2),
        "app_region_code": a["region"].to_numpy(), "app_city_size": rng.choice(4, n, p=[0.3, 0.3, 0.25, 0.15]),
        "app_housing_type": a["housing"].to_numpy(), "app_has_car": a["has_car"].to_numpy(),
        "app_phone_age_months": a["phone_age_m"].to_numpy() + np.floor(years * 12),
        "app_email_domain_type": rng.choice(4, n, p=[0.55, 0.25, 0.15, 0.05]),
        "app_address_match_flag": (rng.random(n) < sigmoid(1.2 - 0.25 * z)).astype(int),
        "app_hour": hour, "app_weekday": np.floor(t + HISTORY_START.day_of_week) % 7,
        "app_channel": channel, "app_device_os": np.where(rng.random(n) < 0.12, 2, a["device_os"].to_numpy()),
        "app_credit_limit": a["limit"].to_numpy(),                     # limit offered at the application
        "app_amount_to_limit": a["amount"].to_numpy() / a["limit"].to_numpy(),
        "app_country_code": np.where(rng.random(n) < 0.0005, 2, 1),         # quasi-constant
        "app_employer_inn_flag": np.where(rng.random(n) < 0.02, (rng.random(n) < 0.8).astype(float), np.nan),  # ~98% missing
    })

    # ---------------- digital footprint ----------------
    month_idx = np.floor(t / 30.44)
    new_source = (t >= _day("2025-11-01")) & (rng.random(n) < 0.35)
    # digital footprint: weak signals of the current risk, most fields are close to noise
    dig = {
        "dig_form_fill_time_sec": np.round(np.exp(5.0 + N(0.4) - 0.05 * z)),
        "dig_form_edits_cnt": P(np.exp(0.8 + 0.05 * z)),
        "dig_copy_paste_flag": (rng.random(n) < sigmoid(-2.0 + 0.1 * z)).astype(int),
        "dig_sessions_7d": P(np.exp(0.9 + 0.08 * z + 0.08 * e)),
        "dig_sessions_30d": P(np.exp(1.9 + 0.06 * z + 0.06 * e)),
        "dig_slider_moves_cnt": P(np.exp(1.0 + 0.06 * z)),
        "dig_max_amount_viewed_ratio": np.clip(a["amount"].to_numpy() / a["limit"].to_numpy() * rng.uniform(1.0, 1.4, n)
                                               + 0.1 * sigmoid(0.3 * z), 0, 2),
        "dig_typing_speed_cpm": np.round(rng.normal(200 - 1.5 * z, 40)),
        "dig_failed_otp_cnt": P(0.1 + 0.04 * sigmoid(z)),
        "dig_push_enabled_flag": (rng.random(n) < sigmoid(0.4 - 0.05 * z)).astype(int),
        "dig_page_views_cnt": P(np.exp(2.0 + 0.03 * z)),
        "dig_hours_since_last_session": np.round(np.exp(3.0 - 0.06 * z + N(1.0)), 1),
        "dig_device_price_class": np.clip(np.round(3 + 2 * inc + N(1.0)), 1, 5),
        "dig_screen_class": np.clip(np.round(3 + 1.5 * inc + N(1.2)), 1, 5),
        "dig_battery_level": np.round(rng.uniform(5, 100, n)),
        "dig_vpn_flag": (rng.random(n) < 0.03 + 0.02 * sigmoid(z)).astype(int),
        "dig_emulator_flag": (rng.random(n) < 0.0008).astype(int),            # quasi-constant
        "dig_app_version": np.round(3.0 + 0.1 * month_idx + rng.choice([0, 0, -0.1, -0.2], n), 1),  # grows every month
        "dig_marketing_source": np.where(new_source, 7, rng.choice(6, n, p=[0.35, 0.25, 0.15, 0.1, 0.1, 0.05])),
        "dig_promo_flag": (rng.random(n) < 0.2).astype(int),
        "dig_referral_flag": (rng.random(n) < 0.08).astype(int),
        "dig_time_on_offer_sec": np.round(np.exp(3.5 + N(0.8))),
        "dig_scroll_depth": rng.beta(4, 2, n),
        "dig_night_session_share": rng.beta(1 + 2 * sigmoid(0.8 * a["night_habit"].to_numpy() + 0.3 * z), 5),
        "dig_contacts_permission_flag": (rng.random(n) < 0.4).astype(int),
        "dig_geo_permission_flag": (rng.random(n) < sigmoid(0.2 - 0.05 * z)).astype(int),
        "dig_os_version": np.round(rng.normal(12, 2, n)),
        "dig_app_installs_age_days": np.round(rng.exponential(200, n) + 365 * years),
        "dig_sms_bank_cnt_30d": P(np.exp(1.5 + 0.2 * inc)),
        "dig_sms_mfo_cnt_30d": P(np.exp(0.5 + 0.3 * e + 0.05 * z)),
        "dig_calculator_opens_cnt": P(np.exp(0.7 + 0.04 * z)),
        "dig_help_page_views": np.full(n, rng.poisson(0.5)),          # constant field (feature-store artefact)
        "dig_chat_messages_cnt": P(np.exp(-0.5 + 0.08 * z)),
        "dig_social_score": np.where(rng.random(n) < 0.03, np.round(500 - 60 * z + N(80)), np.nan),   # ~97% missing
        "dig_selfie_quality": np.where(rng.random(n) < 0.045, rng.uniform(0, 1, n), np.nan),          # ~95.5% missing
    }
    out.update(dig)

    # "calls in 90 days" from the collection system, counted from the report date instead of the
    # application date -> covers the 90 days AFTER the issue of the scored loan (leakage trap)
    out["coll_calls_cnt_90d"] = a["coll_calls_90d"].to_numpy()
    return pd.DataFrame(out)


@dataclass
class SynthData:
    loans: pd.DataFrame
    events: pd.DataFrame
    applications: pd.DataFrame
    external: pd.DataFrame
    latent: pd.DataFrame


def generate(cfg: SynthConfig | None = None) -> SynthData:
    cfg = cfg or SynthConfig()
    rng = np.random.default_rng(cfg.seed)
    cl = _clients(cfg, rng)
    loans, ev = _simulate_loans(cfg, cl, rng)

    ev = ev.merge(loans[["loan_id", "client_id"]], on="loan_id", how="left")
    ev["event_time"] = _to_ts(ev["t"])
    ev["plan_end_after"] = _to_ts(ev["plan_end_after"]).where(ev["plan_end_after"].notna())
    ev["amount"] = ev[["amt_main", "amt_percent", "amt_penalty"]].sum(axis=1)
    order = {"issue": 0, "autodebit_fail": 1, "payment": 2, "cancel": 3, "prolongation": 4, "closure": 5}
    ev = (ev.assign(_o=ev["event"].map(order)).sort_values(["client_id", "event_time", "loan_id", "_o"], kind="mergesort")
          .drop(columns=["t", "_o"]).reset_index(drop=True))
    ev = ev[["client_id", "loan_id", "event_time", "event", "amount", "amt_main", "amt_percent", "amt_penalty",
             "category", "plan_end_after"]]

    loans["application_date"] = _to_ts(loans["app_day"])
    loans["issue_time"] = _to_ts(loans["issue_day_f"])
    loans["first_due_date"] = HISTORY_START + pd.to_timedelta(loans["first_due_day"], unit="D")
    sample = loans[(loans["loan_no"] >= 2) & loans["application_date"].between(SAMPLE_START, SAMPLE_END)].copy()

    apps = sample[["loan_id", "client_id", "application_date", "loan_no", "amount", "term",
                   "npl30d90", "fpd30", "fpd15", "fpd10"]].rename(
        columns={"loan_id": "application_id", "amount": "requested_amount", "term": "requested_term"})
    apps.insert(4, "segment", np.select([apps["loan_no"].eq(2), apps["loan_no"].eq(3)], ["L2", "L3"], "L4+"))
    apps["is_oot"] = (apps["application_date"] >= OOT_START).astype(int)
    external = _external_snapshots(sample, cl, rng)
    latent = sample[["loan_id", "z", "liq", "r_known", "r"]].rename(columns={"loan_id": "application_id"})
    loans_out = loans[["loan_id", "client_id", "loan_no", "application_date", "issue_time", "amount", "term", "first_due_date"]]
    return SynthData(loans=loans_out, events=ev, applications=apps.reset_index(drop=True),
                     external=external.reset_index(drop=True), latent=latent.reset_index(drop=True))


def save(data: SynthData, out_dir) -> None:
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data.loans.to_parquet(out / "loans.parquet", index=False)
    data.events.to_parquet(out / "events.parquet", index=False)
    data.applications.to_parquet(out / "applications.parquet", index=False)
    data.external.to_parquet(out / "external_features.parquet", index=False)
    data.latent.to_parquet(out / "_latent.parquet", index=False)


def load(out_dir) -> SynthData:
    from pathlib import Path
    p = Path(out_dir)
    return SynthData(loans=pd.read_parquet(p / "loans.parquet"), events=pd.read_parquet(p / "events.parquet"),
                     applications=pd.read_parquet(p / "applications.parquet"),
                     external=pd.read_parquet(p / "external_features.parquet"),
                     latent=pd.read_parquet(p / "_latent.parquet"))

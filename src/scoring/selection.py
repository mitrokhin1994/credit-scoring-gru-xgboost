"""Feature selection funnel for a boosting scorecard.

1. technical: near-empty and (quasi-)constant fields                       (train)
2. leakage:   univariate Gini too good to be true -> manual review         (train)
3. stability: PSI between the first and the later months of development    (dates only, no target)
4. power:     IV on valid with bins fitted on train                        (train -> valid)
5. evidence:  XGBoost gain (several seeds), mean |SHAP|, permutation ΔGini; noise canaries set the floor
6. MRMR:      relevance (normalized evidence) / redundancy (|Spearman| with already chosen)
7. rank ensemble of all sources -> correlation pruning -> saturation curve picks the final size

The OOT period is never used here: it stays an honest holdout for the final comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier

from .metrics import gini, psi


@dataclass
class SelectionConfig:
    max_missing: float = 0.95
    max_top_share: float = 0.995
    leak_gini: float = 0.5
    max_psi: float = 0.25
    min_iv: float = 0.02
    n_seeds: int = 3
    n_canaries: int = 10
    perm_repeats: int = 2
    top_n: int = 40
    max_corr: float = 0.8
    max_candidates: int = 40
    saturation_grid: tuple = (5, 8, 10, 12, 15, 20, 25, 30, 40)
    saturation_tol: float = 0.003   # on the cross-validated curve
    seed: int = 42
    n_jobs: int = 8


# ------------------------------------------------------------------ binning / WoE / IV
class TreeBinner:
    """Supervised binning of one numeric feature with a shallow decision tree; missing values get their own bin.
    WoE = ln(share of goods / share of bads) with additive smoothing (higher WoE = lower risk)."""

    def __init__(self, max_bins: int = 8, min_bin_share: float = 0.05, smoothing: float = 0.5):
        self.max_bins, self.min_bin_share, self.smoothing = max_bins, min_bin_share, smoothing

    def fit(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y)
        ok = ~np.isnan(x)
        self.edges_ = np.array([])
        if ok.sum() >= 50 and len(np.unique(x[ok])) > 1:
            tree = DecisionTreeClassifier(max_leaf_nodes=self.max_bins, random_state=0,
                                          min_samples_leaf=max(int(self.min_bin_share * ok.sum()), 1))
            tree.fit(x[ok].reshape(-1, 1), y[ok])
            self.edges_ = np.sort(tree.tree_.threshold[tree.tree_.feature >= 0])
        b = self.bins(x)
        self.woe_ = self._woe(b, y)
        return self

    def bins(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.where(np.isnan(x), -1, np.searchsorted(self.edges_, x, side="left"))

    def _woe(self, b, y) -> dict:
        s = self.smoothing
        t = pd.DataFrame({"b": b, "y": y}).groupby("b")["y"].agg(["sum", "count"])
        bad, good = t["sum"] + s, t["count"] - t["sum"] + s
        return dict(zip(t.index, np.log((good / good.sum()) / (bad / bad.sum()))))

    def woe(self, x) -> np.ndarray:
        return pd.Series(self.bins(x)).map(self.woe_).fillna(0.0).to_numpy()

    def table(self, x, y) -> pd.DataFrame:
        b = self.bins(x)
        t = pd.DataFrame({"bin": b, "y": np.asarray(y)}).groupby("bin")["y"].agg(n="count", bad_rate="mean").reset_index()
        lo = np.r_[-np.inf, self.edges_]
        hi = np.r_[self.edges_, np.inf]
        t["range"] = [("missing" if i < 0 else f"({lo[i]:.4g}; {hi[i]:.4g}]") for i in t["bin"]]
        t["woe"] = t["bin"].map(self.woe_)
        return t


def information_value(bins: np.ndarray, y, smoothing: float = 0.5) -> float:
    t = pd.DataFrame({"b": bins, "y": np.asarray(y)}).groupby("b")["y"].agg(["sum", "count"])
    bad = (t["sum"] + smoothing) / (t["sum"] + smoothing).sum()
    good = (t["count"] - t["sum"] + smoothing) / (t["count"] - t["sum"] + smoothing).sum()
    return float(max(((good - bad) * np.log(good / bad)).sum(), 0.0))


# ------------------------------------------------------------------ stages
def technical_report(train: pd.DataFrame, features, cfg: SelectionConfig) -> pd.DataFrame:
    rows = []
    for f in features:
        s = train[f]
        vc = s.value_counts(normalize=True, dropna=True)
        rows.append({"feature": f, "missing_share": s.isna().mean(), "n_unique": s.nunique(),
                     "top_share": float(vc.iloc[0]) if len(vc) else 1.0})   # share of the most frequent value
    t = pd.DataFrame(rows)
    t["drop_reason"] = np.select([t["missing_share"] > cfg.max_missing, t["n_unique"] <= 1, t["top_share"] > cfg.max_top_share],
                                 ["missing", "constant", "quasi_constant"], "")
    return t


def stability_report(dev: pd.DataFrame, features, date_col: str, ref_months: int = 3) -> pd.DataFrame:
    """Max monthly PSI against the first ``ref_months`` months of the development period."""
    month = dev[date_col].dt.to_period("M")
    months = sorted(month.unique())
    ref = month.isin(months[:ref_months]).to_numpy()
    later = months[ref_months:]
    rows = []
    for f in features:
        v = dev[f].to_numpy(dtype=float)
        vals = [psi(v[ref], v[(month == m).to_numpy()]) for m in later]
        rows.append({"feature": f, "psi_max": max(vals), "psi_last": vals[-1]})
    return pd.DataFrame(rows)


def xgb_evidence(Xtr, ytr, Xva, yva, cfg: SelectionConfig, shap_rows: int = 5000) -> tuple[pd.DataFrame, list]:
    """Gain and mean |SHAP| averaged over seeds, permutation ΔGini on valid for the first seed's model."""
    feats = list(Xtr.columns)
    gain, shap, models = pd.Series(0.0, index=feats), pd.Series(0.0, index=feats), []
    xs = Xva.sample(min(shap_rows, len(Xva)), random_state=cfg.seed)
    for k in range(cfg.n_seeds):
        m = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=4, min_child_weight=50, subsample=0.8,
                              colsample_bytree=0.5, reg_lambda=10, tree_method="hist", eval_metric="auc",
                              early_stopping_rounds=100, random_state=cfg.seed + k, n_jobs=cfg.n_jobs)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        g = pd.Series(m.get_booster().get_score(importance_type="gain")).reindex(feats).fillna(0.0)
        gain += g / g.sum() / cfg.n_seeds
        contrib = m.get_booster().predict(xgb.DMatrix(xs), pred_contribs=True, iteration_range=(0, m.best_iteration + 1))
        shap += pd.Series(np.abs(contrib[:, :-1]).mean(0), index=feats) / cfg.n_seeds
        models.append(m)
    m = models[0]
    rng = np.random.default_rng(cfg.seed)
    base = gini(yva, m.predict_proba(Xva)[:, 1])
    perm = {}
    for f in feats:
        drops = []
        for _ in range(cfg.perm_repeats):
            xp = Xva.copy()
            xp[f] = rng.permutation(xp[f].to_numpy())
            drops.append(base - gini(yva, m.predict_proba(xp)[:, 1]))
        perm[f] = float(np.mean(drops))
    ev = pd.DataFrame({"feature": feats, "gain_importance": gain.to_numpy(), "mean_abs_shap": shap.to_numpy(),
                       "permutation_importance": pd.Series(perm).reindex(feats).to_numpy()})
    return ev, models


def _minmax(s: pd.Series) -> pd.Series:
    s = s.fillna(0.0).clip(lower=0.0)
    return s / s.max() if s.max() > 0 else s * 0


def mrmr(X: pd.DataFrame, evidence: pd.DataFrame, max_features: int, sample_rows: int = 20000, seed: int = 42,
         eps: float = 1e-6) -> pd.DataFrame:
    """Hybrid MRMR: relevance = mean of normalized gain/SHAP/permutation/IV, redundancy = mean over the already
    selected of |Spearman|; greedy by relevance / redundancy."""
    ev = evidence.set_index("feature")
    rel = pd.concat([_minmax(ev[c]) for c in ["gain_importance", "mean_abs_shap", "permutation_importance", "valid_iv"]],
                    axis=1).mean(axis=1)
    cand = [f for f in rel.index if rel[f] > 0]
    xs = X[cand].sample(min(sample_rows, len(X)), random_state=seed)
    corr = xs.rank().corr().abs().fillna(0.0)
    selected, red_at, score_at = [], {}, {}
    remaining = set(cand)
    for _ in range(min(max_features, len(cand))):
        best, best_s, best_r = None, -np.inf, 0.0
        for f in sorted(remaining):
            r = float(corr.loc[f, selected].mean()) if selected else 0.0
            s = rel[f] / (r + eps) if selected else rel[f]
            if s > best_s:
                best, best_s, best_r = f, s, r
        selected.append(best)
        remaining.remove(best)
        red_at[best], score_at[best] = best_r, best_s
    out = pd.DataFrame({"feature": rel.index, "mrmr_relevance": rel.to_numpy()})
    out["mrmr_rank"] = out["feature"].map({f: i + 1 for i, f in enumerate(selected)})
    out["mrmr_redundancy"] = out["feature"].map(red_at)
    return out


def rank_ensemble(table: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Sum of 1/rank over sources where the feature is in the top-N (gain, SHAP, permutation, IV, MRMR)."""
    t = table.copy()
    for col, label in [("gain_importance", "gain"), ("mean_abs_shap", "shap"), ("permutation_importance", "perm"),
                       ("valid_iv", "iv")]:
        pos = t[col] > 0
        t[f"{label}_rank"] = t[col].where(pos).rank(ascending=False, method="min")
    t["mrmr_rank_"] = t["mrmr_rank"]
    rank_cols = ["gain_rank", "shap_rank", "perm_rank", "iv_rank", "mrmr_rank_"]
    in_top = t[rank_cols].le(top_n)
    t["top_sources"] = in_top.sum(axis=1)
    t["ensemble_score"] = (1.0 / t[rank_cols]).where(in_top, 0.0).sum(axis=1)
    return t.drop(columns="mrmr_rank_").sort_values(["ensemble_score", "top_sources"], ascending=False).reset_index(drop=True)


def correlation_prune(X: pd.DataFrame, ordered, max_corr: float, limit: int, sample_rows: int = 20000, seed: int = 42):
    xs = X[list(ordered)].sample(min(sample_rows, len(X)), random_state=seed).rank()
    corr = xs.corr().abs().fillna(0.0)
    kept, log = [], []
    for f in ordered:
        worst = corr.loc[f, kept].idxmax() if kept else None
        r = float(corr.loc[f, worst]) if kept else 0.0
        if r < max_corr:
            kept.append(f)
        log.append({"feature": f, "max_corr_with_kept": r, "most_correlated": worst, "kept": r < max_corr})
        if len(kept) >= limit:
            break
    return kept, pd.DataFrame(log)


def saturation_curve(dev: pd.DataFrame, y, clients, ordered, grid, n_folds: int = 3, seed: int = 42,
                     n_jobs: int = 8) -> pd.DataFrame:
    """Mean out-of-fold Gini of a fixed XGBoost on the top-n features; folds by client on train+valid.
    A single valid sample is too noisy (SE of Gini ~0.01) to compare feature sets that differ by 0.005."""
    from .gru import client_folds
    y = np.asarray(y)
    fold = client_folds(np.asarray(clients), n_folds, seed)
    rows = []
    for n in sorted({g for g in grid if g < len(ordered)} | {len(ordered)}):   # always end with all candidates
        cols = list(ordered[:n])
        tr_g, va_g = [], []
        for k in range(n_folds):
            tr, va = fold != k, fold == k
            m = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=4, min_child_weight=50, subsample=0.8,
                                  colsample_bytree=0.8, reg_lambda=10, tree_method="hist", max_bin=128, eval_metric="auc",
                                  early_stopping_rounds=50, random_state=seed, n_jobs=n_jobs)
            m.fit(dev.loc[tr, cols], y[tr], eval_set=[(dev.loc[va, cols], y[va])], verbose=False)
            tr_g.append(gini(y[tr], m.predict_proba(dev.loc[tr, cols])[:, 1]))
            va_g.append(gini(y[va], m.predict_proba(dev.loc[va, cols])[:, 1]))
        rows.append({"n_features": n, "train_gini": np.mean(tr_g), "valid_gini": np.mean(va_g), "valid_gini_std": np.std(va_g)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ orchestration
@dataclass
class SelectionResult:
    selected: list
    funnel: pd.DataFrame
    tables: dict = field(default_factory=dict)
    binners: dict = field(default_factory=dict)


def select_features(train: pd.DataFrame, valid: pd.DataFrame, features, target: str, date_col: str,
                    cfg: SelectionConfig | None = None, log=print, client_col: str = "client_id") -> SelectionResult:
    cfg = cfg or SelectionConfig()
    ytr, yva = train[target].to_numpy(), valid[target].to_numpy()
    funnel, tables = [("Initial set", len(features))], {}

    tech = technical_report(train, features, cfg)
    tables["technical"] = tech
    feats = tech.loc[tech["drop_reason"].eq(""), "feature"].tolist()
    funnel.append(("Technical filters (missing, constant)", len(feats)))
    log(f"technical: {len(features)} -> {len(feats)}")

    binners = {f: TreeBinner().fit(train[f], ytr) for f in feats}
    uni = pd.DataFrame({"feature": feats,
                        "train_gini": [abs(gini(ytr, binners[f].woe(train[f]))) for f in feats],
                        "valid_gini": [abs(gini(yva, binners[f].woe(valid[f]))) for f in feats]})
    uni["leak_flag"] = uni["train_gini"] > cfg.leak_gini
    tables["leakage"] = uni
    feats = uni.loc[~uni["leak_flag"], "feature"].tolist()
    funnel.append(("Suspected leakage (|Gini| > 0.5)", len(feats)))
    log(f"leakage: -> {len(feats)}  flagged: {uni.loc[uni['leak_flag'], 'feature'].tolist()}")

    dev = pd.concat([train, valid])
    stab = stability_report(dev, feats, date_col)
    iv = pd.DataFrame({"feature": feats,
                       "train_iv": [information_value(binners[f].bins(train[f]), ytr) for f in feats],
                       "valid_iv": [information_value(binners[f].bins(valid[f]), yva) for f in feats],
                       "n_bins": [len(binners[f].woe_) for f in feats]})
    tables["stability"], tables["iv"] = stab, iv
    feats = stab.loc[stab["psi_max"] <= cfg.max_psi, "feature"].tolist()
    funnel.append((f"Stability (PSI ≤ {cfg.max_psi})", len(feats)))
    log(f"stability: -> {len(feats)}  dropped: {stab.loc[stab['psi_max'] > cfg.max_psi, 'feature'].tolist()}")

    strong = set(iv.loc[iv["valid_iv"] >= cfg.min_iv, "feature"])
    feats = [f for f in feats if f in strong]
    funnel.append((f"Predictive power (valid IV ≥ {cfg.min_iv})", len(feats)))
    log(f"iv: -> {len(feats)}")

    # model evidence with noise canaries: a real feature has to beat pure noise
    rng = np.random.default_rng(cfg.seed)
    canaries = [f"canary_{i:02d}" for i in range(cfg.n_canaries)]
    Xtr, Xva = train[feats].copy(), valid[feats].copy()
    for i, c in enumerate(canaries):
        gen = (lambda n: rng.normal(size=n)) if i % 2 == 0 else (lambda n: rng.uniform(size=n))
        Xtr[c], Xva[c] = gen(len(Xtr)), gen(len(Xva))
    ev, _ = xgb_evidence(Xtr, ytr, Xva, yva, cfg)
    ev = ev.merge(iv[["feature", "valid_iv"]], on="feature", how="left")
    floor = ev[ev["feature"].isin(canaries)][["gain_importance", "mean_abs_shap", "permutation_importance"]].max()
    beats = (ev[["gain_importance", "mean_abs_shap", "permutation_importance"]] > floor).sum(axis=1)
    ev["beats_canaries"] = (beats >= 2) & ~ev["feature"].isin(canaries)
    tables["evidence"] = ev
    tables["canary_floor"] = floor
    feats = ev.loc[ev["beats_canaries"], "feature"].tolist()
    funnel.append(("Importance above noise canaries", len(feats)))
    log(f"canaries: -> {len(feats)}")

    ev_f = ev[ev["feature"].isin(feats)]
    mr = mrmr(train[feats], ev_f, max_features=min(cfg.top_n, len(feats)), seed=cfg.seed)
    table = rank_ensemble(ev_f.merge(mr, on="feature"), cfg.top_n)
    tables["ensemble"] = table
    ordered = table.loc[table["ensemble_score"] > 0, "feature"].tolist()
    kept, corr_log = correlation_prune(train, ordered, cfg.max_corr, cfg.max_candidates, seed=cfg.seed)
    tables["correlation"] = corr_log
    funnel.append((f"Rank ensemble + correlation (|ρ| < {cfg.max_corr})", len(kept)))
    log(f"ensemble+corr: -> {len(kept)}")

    dev = pd.concat([train, valid], ignore_index=True)
    sat = saturation_curve(dev, dev[target], dev[client_col], kept, cfg.saturation_grid, seed=cfg.seed, n_jobs=cfg.n_jobs)
    tables["saturation"] = sat
    best = sat["valid_gini"].max()
    n_final = int(sat.loc[sat["valid_gini"] >= best - cfg.saturation_tol, "n_features"].min())
    selected = kept[:n_final]
    funnel.append(("Saturation curve (final set)", len(selected)))
    log(f"saturation: best valid gini {best:.4f}, chosen n={n_final}")
    return SelectionResult(selected=selected, funnel=pd.DataFrame(funnel, columns=["stage", "n_features"]),
                           tables=tables, binners={f: binners[f] for f in binners})

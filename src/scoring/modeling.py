"""XGBoost tuning (Optuna, penalized valid Gini), final fit, bootstrap comparison, WoE-logit scorecard baseline."""
from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

from .metrics import gini
from .selection import TreeBinner

EARLY_STOPPING = 50


def penalized_gini(train_gini: float, valid_gini: float, gap_free: float = 0.03, penalty: float = 0.5) -> float:
    """Valid Gini minus a penalty for the train-valid gap above ``gap_free`` (overfitting control)."""
    return valid_gini - penalty * max(0.0, train_gini - valid_gini - gap_free)


def make_xgb(params: dict, seed: int = 42, n_jobs: int = 8, n_estimators: int = 2000) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(objective="binary:logistic", eval_metric="auc", tree_method="hist", max_bin=128,
                             n_estimators=n_estimators, early_stopping_rounds=EARLY_STOPPING,
                             random_state=seed, n_jobs=n_jobs, **params)


def suggest_params(trial: optuna.Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "min_child_weight": trial.suggest_float("min_child_weight", 10, 300, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 30, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 100, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 10.0),
    }


def tune_xgb(Xtr, ytr, Xva, yva, n_trials: int = 60, seed: int = 42, n_jobs: int = 8,
             gap_free: float = 0.03, penalty: float = 0.5, parallel_trials: int = 4) -> optuna.Study:
    """TPE search. On ~40k rows XGBoost barely scales with threads, so trials run in parallel
    (``parallel_trials`` workers x ``n_jobs // parallel_trials`` threads each)."""
    threads = max(n_jobs // parallel_trials, 1)

    def objective(trial):
        m = make_xgb(suggest_params(trial), seed, threads)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        g_tr, g_va = gini(ytr, m.predict_proba(Xtr)[:, 1]), gini(yva, m.predict_proba(Xva)[:, 1])
        trial.set_user_attr("train_gini", g_tr)
        trial.set_user_attr("valid_gini", g_va)
        trial.set_user_attr("best_iteration", int(m.best_iteration))
        return penalized_gini(g_tr, g_va, gap_free, penalty)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=10))
    study.optimize(objective, n_trials=n_trials, n_jobs=parallel_trials)
    return study


def fit_final(params: dict, Xtr, ytr, Xva, yva, seed: int = 42, n_jobs: int = 8) -> xgb.XGBClassifier:
    m = make_xgb(params, seed, n_jobs)
    m.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], verbose=False)
    return m


def bootstrap_gini(y, scores: dict, groups, n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    """Cluster bootstrap by client: resample clients with replacement (weights), Gini of every score per draw."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(pd.Series(groups))
    y = np.asarray(y)
    rows = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[codes]
        rows.append({k: gini(y, v, sample_weight=w) for k, v in scores.items()})
    return pd.DataFrame(rows)


def summarize_bootstrap(boot: pd.DataFrame, point: dict, pairs) -> pd.DataFrame:
    rows = [{"metric": k, "value": point[k], "ci_low": boot[k].quantile(0.025), "ci_high": boot[k].quantile(0.975)}
            for k in boot.columns]
    for a, b in pairs:
        d = boot[a] - boot[b]
        rows.append({"metric": f"Δ {a} − {b}", "value": point[a] - point[b], "ci_low": d.quantile(0.025),
                     "ci_high": d.quantile(0.975), "p_value": float((d <= 0).mean())})
    return pd.DataFrame(rows)


class WoeLogitScorecard:
    """Classic scorecard: tree binning -> WoE -> logistic regression -> points (PDO scaling)."""

    def __init__(self, features, C: float = 0.5, pdo: float = 20.0, base_score: float = 600.0, base_odds: float = 50.0):
        self.features, self.C, self.pdo, self.base_score, self.base_odds = list(features), C, pdo, base_score, base_odds

    def fit(self, X: pd.DataFrame, y):
        self.binners_ = {f: TreeBinner().fit(X[f], y) for f in self.features}
        self.lr_ = LogisticRegression(C=self.C, max_iter=1000).fit(self._woe(X), y)
        return self

    def _woe(self, X) -> np.ndarray:
        return np.column_stack([self.binners_[f].woe(X[f]) for f in self.features])

    def predict_proba(self, X) -> np.ndarray:
        return self.lr_.predict_proba(self._woe(X))

    def points_table(self, X, y) -> pd.DataFrame:
        """Points per bin: score = offset - factor * ln(odds_bad); odds 50:1 good:bad -> 600, +20 points doubles odds."""
        factor = self.pdo / np.log(2)
        offset = self.base_score - factor * np.log(self.base_odds)
        n, a = len(self.features), self.lr_.intercept_[0]
        rows = []
        for j, f in enumerate(self.features):
            t = self.binners_[f].table(X[f], y)
            beta = self.lr_.coef_[0][j]
            # log-odds of bad = a + sum(beta_j * woe_j); WoE is ln(good/bad), so beta < 0 for a sensible feature
            t["points"] = np.round(offset / n - factor * (a / n + beta * t["woe"]))
            t.insert(0, "feature", f)
            rows.append(t)
        return pd.concat(rows, ignore_index=True)

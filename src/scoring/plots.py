"""Plotting helpers with one consistent style (light surface, hairline grid, fixed colors per entity)."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
# the same entity keeps its color in every chart
MODEL_COLORS = {"XGB base": BLUE, "XGB + GRU": ORANGE, "GRU": AQUA, "WoE scorecard": YELLOW, "WoE scorecard + GRU": VIOLET}
SPLIT_COLORS = {"train": "#86b6ef", "valid": "#3987e5", "oot": "#104281"}   # ordinal blue ramp
SEGMENT_COLORS = {"L2": BLUE, "L3": ORANGE, "L4+": AQUA}
DIVERGING = LinearSegmentedColormap.from_list("div", ["#104281", "#3987e5", "#f0efec", "#e66767", "#a8201f"])
SEQUENTIAL = LinearSegmentedColormap.from_list("seq", ["#f0efec", "#9ec5f4", "#3987e5", "#104281"])


def setup_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linestyle": "-", "grid.linewidth": 0.7, "axes.axisbelow": True,
        "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 10,
        "lines.linewidth": 2.0, "lines.markersize": 5, "legend.frameon": False, "legend.fontsize": 9,
        "patch.linewidth": 0,
    })


def save(fig, name: str, out_dir: Path | None):
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(out_dir) / f"{name}.png")
    return fig


def _shade_oot(ax, oot_start, label=True):
    lo, hi = ax.get_xlim()
    x0 = mpl.dates.date2num(pd.Timestamp(oot_start)) if isinstance(lo, float) else oot_start
    ax.axvspan(x0, hi, color="#eda100", alpha=0.07, lw=0)
    ax.axvline(x0, color=INK2, lw=0.9)
    if label:
        ax.text(x0, 1.0, "  OOT", transform=ax.get_xaxis_transform(), va="top", ha="left", color=INK2, fontsize=9)
    ax.set_xlim(lo, hi)


# ---------------------------------------------------------------- data overview
def monthly_overview(apps: pd.DataFrame, target: str, oot_start, out_dir=None):
    m = apps.assign(month=apps["application_date"].dt.to_period("M").dt.to_timestamp())
    vol = m.pivot_table(index="month", columns="segment", values="application_id", aggfunc="size").fillna(0)
    rate = m.groupby("month")[target].mean()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    bottom = np.zeros(len(vol))
    for s in ["L2", "L3", "L4+"]:
        a1.bar(vol.index, vol[s], bottom=bottom, width=24, color=SEGMENT_COLORS[s], label=s, edgecolor=SURFACE, linewidth=1)
        bottom += vol[s].to_numpy()
    a1.set_ylabel("Applications per month")
    a1.set_title("Issued loans by segment")
    a1.legend(ncol=3, loc="upper left")
    a2.plot(rate.index, rate.values, color=INK, marker="o")
    a2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    a2.set_ylabel(f"{target} rate")
    a2.set_title(f"Monthly default rate ({target})")
    for ax in (a1, a2):
        _shade_oot(ax, oot_start, label=ax is a1)
    fig.tight_layout()
    return save(fig, "01_monthly_overview", out_dir)


def client_timeline(events: pd.DataFrame, loans: pd.DataFrame, client_id: int, app_date=None, out_dir=None):
    ev = events[events["client_id"] == client_id]
    ln = loans[loans["client_id"] == client_id]
    close = ev[ev["event"] == "closure"].set_index("loan_id")["event_time"]
    fig, ax = plt.subplots(figsize=(11, 3.4))
    for _, r in ln.iterrows():
        end = close.get(r["loan_id"], ev["event_time"].max())
        ax.plot([r["issue_time"], end], [0, 0], color=AXIS, lw=7, solid_capstyle="round", zorder=1)
        ax.plot([r["first_due_date"]] * 2, [-0.18, 0.18], color=INK2, lw=1, zorder=2)
    styles = {"issue": (BLUE, "o", "issue"), "payment": (AQUA, "s", "payment"), "prolongation": (YELLOW, "D", "prolongation"),
              "closure": (GREEN, "^", "closure"), "autodebit_fail": (MAGENTA, "v", "failed auto-debit"),
              "cancel": (RED, "x", "payment reversal")}
    for e, (c, mk, lab) in styles.items():
        s = ev[ev["event"] == e]
        if len(s):
            y = {"issue": 0.45, "payment": -0.45, "prolongation": 0.9, "closure": 0.45, "autodebit_fail": -0.9,
                 "cancel": -0.9}[e]
            ring = {} if mk == "x" else {"edgecolor": SURFACE, "linewidth": 0.8}   # surface ring on filled markers
            ax.scatter(s["event_time"], np.full(len(s), y), color=c, marker=mk, s=42, label=lab, zorder=3, **ring)
    if app_date is not None:
        ax.axvline(app_date, color=ORANGE, lw=1.5)
        ax.text(app_date, 1.25, " scored application", color=INK, fontsize=9, va="center")
    ax.set_yticks([])
    ax.set_ylim(-1.3, 1.45)
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Client {client_id}: loans (grey bars, tick = due date) and events")
    ax.legend(ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.38))
    fig.tight_layout()
    return save(fig, "02_client_timeline", out_dir)


def sequence_lengths(lengths, segments, out_dir=None):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    d = pd.DataFrame({"len": lengths, "seg": segments})
    bins = np.arange(0, min(d["len"].quantile(0.995), 150) + 4, 3)
    for s in ["L2", "L3", "L4+"]:
        v = d.loc[d["seg"] == s, "len"]
        ax.hist(v, bins=bins, histtype="step", lw=2, color=SEGMENT_COLORS[s], label=f"{s}: median {v.median():.0f}")
    ax.set_xlabel("Tokens in history before the application")
    ax.set_ylabel("Applications")
    ax.set_title("Sequence length by segment")
    ax.legend()
    fig.tight_layout()
    return save(fig, "03_sequence_lengths", out_dir)


# ---------------------------------------------------------------- feature selection
def selection_funnel(funnel: pd.DataFrame, out_dir=None):
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    y = np.arange(len(funnel))[::-1]
    ax.barh(y, funnel["n_features"], color=BLUE, height=0.62)
    for yi, n in zip(y, funnel["n_features"]):
        ax.text(n + funnel["n_features"].max() * 0.01, yi, f"{n}", va="center", color=INK, fontsize=10)
    ax.set_yticks(y, funnel["stage"])
    ax.set_xlabel("Features")
    ax.grid(axis="y", visible=False)
    ax.set_title("Feature selection funnel")
    fig.tight_layout()
    return save(fig, "10_selection_funnel", out_dir)


def iv_vs_psi(iv: pd.DataFrame, stab: pd.DataFrame, min_iv: float, max_psi: float, annotate=(), out_dir=None):
    d = stab.merge(iv, on="feature", how="left")
    floor = 1e-3                                  # log scales: zero PSI / IV are drawn on the floor
    d["x"], d["y"] = d["psi_max"].clip(lower=floor), d["valid_iv"].fillna(0).clip(lower=floor)
    fig, ax = plt.subplots(figsize=(9, 5))
    ok = (d["psi_max"] <= max_psi) & (d["valid_iv"] >= min_iv)
    ax.scatter(d.loc[~ok, "x"], d.loc[~ok, "y"], s=16, color=AXIS, label="dropped", zorder=2)
    ax.scatter(d.loc[ok, "x"], d.loc[ok, "y"], s=18, color=BLUE, label="passed", zorder=3)
    ax.axvline(max_psi, color=INK2, lw=0.9)
    ax.axhline(min_iv, color=INK2, lw=0.9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.text(max_psi, floor, f"  PSI = {max_psi}", color=INK2, fontsize=8.5, va="bottom")
    ax.text(floor, min_iv, f" IV = {min_iv}", color=INK2, fontsize=8.5, va="bottom")
    for f in annotate:
        r = d[d["feature"] == f]
        if len(r):
            ax.annotate(f, (r["x"].iat[0], r["y"].iat[0]), xytext=(-6, 5), textcoords="offset points",
                        fontsize=8.5, color=INK, ha="right")
    ax.set_xlabel("PSI: max over development months vs first three months (log)")
    ax.set_ylabel("Valid IV, bins fitted on train (log)")
    ax.set_title("Stability vs predictive power")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return save(fig, "11_iv_vs_psi", out_dir)


def evidence_vs_canaries(ev: pd.DataFrame, floor: pd.Series, top: int = 30, out_dir=None):
    d = ev.sort_values("permutation_importance", ascending=False)
    real = d[~d["feature"].str.startswith("canary_")].head(top)
    best_canary = d[d["feature"].str.startswith("canary_")]["permutation_importance"].max()
    fig, ax = plt.subplots(figsize=(9, 7.5))
    y = np.arange(len(real))[::-1]
    ax.barh(y, real["permutation_importance"], color=np.where(real["beats_canaries"], BLUE, AXIS), height=0.65)
    ax.axvline(best_canary, color=ORANGE, lw=1.5)
    ax.text(best_canary, len(real) - 0.3, " best noise canary", color=INK, fontsize=9, va="bottom")
    ax.set_yticks(y, real["feature"], fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Permutation importance: valid Gini drop when shuffled")
    ax.set_title(f"Top {top} features by permutation importance")
    fig.tight_layout()
    return save(fig, "12_permutation_vs_canaries", out_dir)


def ensemble_heatmap(table: pd.DataFrame, top: int = 25, out_dir=None):
    t = table.head(top).set_index("feature")[["gain_rank", "shap_rank", "perm_rank", "iv_rank", "mrmr_rank"]]
    t.columns = ["gain", "SHAP", "permutation", "IV", "MRMR"]
    fig, ax = plt.subplots(figsize=(7.5, 0.3 * top + 1.6))
    vals = t.to_numpy(dtype=float)
    im = ax.imshow(np.where(np.isnan(vals), 99, vals), cmap=SEQUENTIAL.reversed(), vmin=1, vmax=60, aspect="auto")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            txt = "–" if np.isnan(v) else f"{v:.0f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="#ffffff" if (not np.isnan(v) and v <= 15) else INK)
    ax.set_xticks(range(t.shape[1]), t.columns)
    ax.set_yticks(range(t.shape[0]), t.index, fontsize=8.5)
    ax.grid(False)
    ax.set_title("Feature rank in each evidence source (1 = best)")
    fig.colorbar(im, ax=ax, shrink=0.6, label="rank")
    fig.tight_layout()
    return save(fig, "13_rank_ensemble", out_dir)


def saturation(sat: pd.DataFrame, n_final: int, out_dir=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sat["n_features"], sat["train_gini"], marker="o", color=SPLIT_COLORS["train"], label="train (mean over folds)")
    ax.plot(sat["n_features"], sat["valid_gini"], marker="o", color=SPLIT_COLORS["valid"], label="out-of-fold (mean over folds)")
    if "valid_gini_std" in sat:
        ax.fill_between(sat["n_features"], sat["valid_gini"] - sat["valid_gini_std"], sat["valid_gini"] + sat["valid_gini_std"],
                        color=SPLIT_COLORS["valid"], alpha=0.12, lw=0)
    ax.axvline(n_final, color=ORANGE, lw=1.5)
    ax.text(n_final, ax.get_ylim()[0], f"  chosen: {n_final}", color=INK, fontsize=9, va="bottom")
    ax.set_xlabel("Number of features (in rank-ensemble order)")
    ax.set_ylabel("Gini")
    ax.set_title("Saturation curve: 3 client folds on train+valid (band = ±1 std)")
    ax.legend()
    fig.tight_layout()
    return save(fig, "14_saturation_curve", out_dir)


def correlation_heatmap(X: pd.DataFrame, features, out_dir=None, name="15_correlation"):
    c = X[features].rank().corr()
    fig, ax = plt.subplots(figsize=(0.42 * len(features) + 3, 0.42 * len(features) + 2))
    im = ax.imshow(c, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(features)), features, rotation=90, fontsize=8)
    ax.set_yticks(range(len(features)), features, fontsize=8)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Spearman ρ")
    ax.set_title("Correlation of the final features")
    fig.tight_layout()
    return save(fig, name, out_dir)


# ---------------------------------------------------------------- GRU
def gru_learning_curves(histories, out_dir=None):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    for k, h in enumerate(histories):
        h = pd.DataFrame(h)
        a1.plot(h["epoch"], h["train_loss"], marker="o", color=CATEGORICAL[k], label=f"fold {k + 1}")
        a2.plot(h["epoch"], h["valid_gini"], marker="o", color=CATEGORICAL[k], label=f"fold {k + 1}")
    a1.set_title("Training loss (multi-task)")
    a2.set_title("NPL30d90 Gini on the fold's inner valid")
    for ax in (a1, a2):
        ax.set_xlabel("Epoch")
    a2.legend()
    fig.tight_layout()
    return save(fig, "20_gru_learning_curves", out_dir)


def score_distribution(score, y, title: str, xlabel: str, out_dir=None, name="21_score_distribution"):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bins = np.linspace(np.quantile(score, 0.002), np.quantile(score, 0.998), 60)
    ax.hist(score[y == 0], bins=bins, density=True, histtype="stepfilled", alpha=0.35, color=BLUE, label="good")
    ax.hist(score[y == 1], bins=bins, density=True, histtype="stepfilled", alpha=0.35, color=ORANGE, label="default")
    ax.hist(score[y == 0], bins=bins, density=True, histtype="step", lw=1.5, color=BLUE)
    ax.hist(score[y == 1], bins=bins, density=True, histtype="step", lw=1.5, color=ORANGE)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return save(fig, name, out_dir)


# ---------------------------------------------------------------- tuning
def optuna_history(studies: dict, out_dir=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, st in studies.items():
        df = st.trials_dataframe()
        c = MODEL_COLORS.get(name, BLUE)
        ax.scatter(df["number"], df["value"], s=14, color=c, alpha=0.45)
        ax.plot(df["number"], df["value"].cummax(), color=c, label=f"{name}: best {df['value'].max():.4f}")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Penalized valid Gini")
    lo = min(st.trials_dataframe()["value"].quantile(0.1) for st in studies.values())
    hi = max(st.trials_dataframe()["value"].max() for st in studies.values())
    ax.set_ylim(lo - 0.01, hi + 0.01)
    ax.set_title("Optuna (TPE): hyperparameter search history")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return save(fig, "30_optuna_history", out_dir)


def param_importance(study, out_dir=None, name="31_param_importance"):
    df = study.trials_dataframe()
    pcols = [c for c in df.columns if c.startswith("params_")]
    imp = {c.replace("params_", ""): abs(pd.Series(df[c]).rank().corr(df["value"].rank())) for c in pcols}
    s = pd.Series(imp).sort_values()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(s.index, s.values, color=BLUE, height=0.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("|Spearman ρ| of parameter with the objective")
    ax.set_title("Which hyperparameters matter (XGB + GRU)")
    fig.tight_layout()
    return save(fig, name, out_dir)


def boosting_curve(model, out_dir=None):
    r = model.evals_result()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for key, lab in [("validation_0", "train"), ("validation_1", "valid")]:
        auc = np.asarray(r[key]["auc"])
        ax.plot(np.arange(1, len(auc) + 1), 2 * auc - 1, color=SPLIT_COLORS[lab], label=lab)
    ax.axvline(model.best_iteration + 1, color=ORANGE, lw=1.5)
    ax.text(model.best_iteration + 1, ax.get_ylim()[0], f"  best iteration = {model.best_iteration + 1}", fontsize=9, va="bottom")
    ax.set_xlabel("Number of trees")
    ax.set_ylabel("Gini")
    ax.set_title("Learning curve of the final booster (early stopping on valid)")
    ax.legend()
    fig.tight_layout()
    return save(fig, "32_boosting_curve", out_dir)


# ---------------------------------------------------------------- comparison
def gini_comparison(metrics: pd.DataFrame, ci: pd.DataFrame | None = None, out_dir=None):
    """metrics: model x split Gini (columns train/valid/oot); ci: OOT CI per model (ci_low, ci_high)."""
    models = list(metrics.index)
    splits = ["train", "valid", "oot"]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    w = 0.8 / len(models)
    x = np.arange(len(splits))
    for k, mname in enumerate(models):
        vals = metrics.loc[mname, splits].to_numpy(dtype=float)
        pos = x - 0.4 + w / 2 + k * w
        ax.bar(pos, vals, width=w * 0.92, color=MODEL_COLORS.get(mname, CATEGORICAL[k]), label=mname)
        top = vals[-1]
        if ci is not None and mname in ci.index:
            lo, hi = ci.loc[mname, "ci_low"], ci.loc[mname, "ci_high"]
            ax.errorbar(pos[-1], vals[-1], yerr=[[vals[-1] - lo], [hi - vals[-1]]], color=INK, capsize=3, lw=1)
            top = hi
        ax.text(pos[-1], top + 0.006, f"{vals[-1]:.3f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(x, ["train", "valid", "OOT"])
    ax.set_ylabel("Gini")
    lo = np.nanmin(metrics[splits].to_numpy(dtype=float))
    ax.set_ylim(max(0, lo - 0.1), np.nanmax(metrics[splits].to_numpy(dtype=float)) + 0.06)
    ax.set_title("Gini by sample (OOT with 95% client-bootstrap CI)")
    ax.legend(ncol=len(models), loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    return save(fig, "40_gini_comparison", out_dir)


def delta_bootstrap(delta: np.ndarray, point: float, label: str, out_dir=None):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.hist(delta, bins=40, color=ORANGE, alpha=0.75)
    lo, hi = np.quantile(delta, [0.025, 0.975])
    for v in (lo, hi):
        ax.axvline(v, color=INK2, lw=1)
    ax.axvline(0, color=INK, lw=1.2)
    ax.axvline(point, color=ORANGE, lw=2)
    ax.set_title(f"{label}: ΔGini = {point:+.4f}, 95% CI [{lo:+.4f}; {hi:+.4f}]")
    ax.set_xlabel("OOT ΔGini (client bootstrap)")
    ax.set_ylabel("Draws")
    fig.tight_layout()
    return save(fig, "41_delta_gini_bootstrap", out_dir)


def gini_by_segment(df: pd.DataFrame, out_dir=None):
    """df: index segment, columns models."""
    models = list(df.columns)
    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.8 / len(models)
    x = np.arange(len(df))
    for k, mname in enumerate(models):
        pos = x - 0.4 + w / 2 + k * w
        ax.bar(pos, df[mname], width=w * 0.92, color=MODEL_COLORS.get(mname, CATEGORICAL[k]), label=mname)
    ax.set_xticks(x, df.index)
    ax.set_ylabel("OOT Gini")
    ax.set_ylim(max(0, df.to_numpy().min() - 0.1), df.to_numpy().max() + 0.04)
    ax.set_title("Quality by segment (OOT)")
    ax.legend(ncol=len(models), loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    return save(fig, "42_gini_by_segment", out_dir)


def roc_curves(curves: dict, out_dir=None, title="ROC on OOT", name="43_roc_oot", colors=None):
    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    for k, (lab, (y, p)) in enumerate(curves.items()):
        fpr, tpr, _ = roc_curve(y, p)
        g = 2 * np.trapezoid(tpr, fpr) - 1
        c = (colors or MODEL_COLORS).get(lab, CATEGORICAL[k])
        ax.plot(fpr, tpr, color=c, lw=1.8, label=f"{lab}: Gini {g:.3f}")
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return save(fig, name, out_dir)


def calibration(tables: dict, out_dir=None):
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    mx = 0
    for lab, t in tables.items():
        ax.plot(t["mean_pd"], t["bad_rate"], marker="o", color=SPLIT_COLORS.get(lab, BLUE), label=lab)
        mx = max(mx, t["mean_pd"].max(), t["bad_rate"].max())
    ax.plot([0, mx], [0, mx], color=AXIS, lw=1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Mean predicted PD in decile")
    ax.set_ylabel("Observed default rate")
    ax.set_title("Calibration by decile")
    ax.legend()
    fig.tight_layout()
    return save(fig, "50_calibration", out_dir)


def risk_grades(tables: dict, out_dir=None):
    fig, ax = plt.subplots(figsize=(9, 4))
    k = len(tables)
    w = 0.8 / k
    for i, (lab, t) in enumerate(tables.items()):
        pos = t["grade"].to_numpy() - 0.4 + w / 2 + i * w
        ax.bar(pos, t["bad_rate"], width=w * 0.92, color=SPLIT_COLORS.get(lab, CATEGORICAL[i]), label=lab)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xticks(range(1, int(max(t["grade"].max() for t in tables.values())) + 1))
    ax.set_xlabel("Risk grade (train PD quantiles; 1 = lowest risk)")
    ax.set_ylabel("Default rate")
    ax.set_title("Risk-grade monotonicity on every sample")
    ax.legend()
    fig.tight_layout()
    return save(fig, "51_risk_grades", out_dir)


def monthly_gini(df: pd.DataFrame, models, oot_start, out_dir=None):
    fig, ax = plt.subplots(figsize=(11, 4))
    for k, mname in enumerate(models):
        ax.plot(df["month"], df[mname], marker="o", color=MODEL_COLORS.get(mname, CATEGORICAL[k]), label=mname)
    _shade_oot(ax, oot_start)
    ax.set_ylabel("Gini")
    ax.set_title("Monthly Gini stability")
    ax.legend(ncol=len(models), loc="lower left")
    fig.tight_layout()
    return save(fig, "52_monthly_gini", out_dir)


def psi_bars(psi_s: pd.Series, out_dir=None, title="PSI of final-model features: train → OOT", name="53_feature_psi"):
    s = psi_s.sort_values()
    fig, ax = plt.subplots(figsize=(8.5, 0.28 * len(s) + 1.5))
    ax.barh(s.index, s.values, color=np.where(s.values > 0.1, YELLOW, BLUE), height=0.65)
    for v in (0.1, 0.25):
        ax.axvline(v, color=INK2, lw=0.9)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.set_xlabel("PSI (0.1 = moderate shift, 0.25 = significant)")
    ax.set_title(title)
    fig.tight_layout()
    return save(fig, name, out_dir)


# ---------------------------------------------------------------- interpretation
def shap_beeswarm(shap_values: np.ndarray, X: pd.DataFrame, top: int = 15, out_dir=None, max_points: int = 3000, seed: int = 0):
    order = np.argsort(np.abs(shap_values).mean(0))[::-1][:top]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(max_points, len(X)), replace=False)
    fig, ax = plt.subplots(figsize=(9, 0.42 * top + 1.6))
    for rank, j in enumerate(order[::-1]):
        sv = shap_values[idx, j]
        xv = X.iloc[idx, j].to_numpy(dtype=float)
        lo, hi = np.nanpercentile(xv, [5, 95]) if np.isfinite(xv).any() else (0, 1)
        cv = np.clip((xv - lo) / (hi - lo + 1e-12), 0, 1)
        # vertical jitter proportional to local density
        bins = np.round(sv / (np.ptp(sv) + 1e-12) * 60)
        _, inv, cnt = np.unique(bins, return_inverse=True, return_counts=True)
        jit = (rng.random(len(sv)) - 0.5) * np.minimum(cnt[inv] / cnt.max(), 1) * 0.8
        nan = np.isnan(xv)
        ax.scatter(sv[~nan], rank + jit[~nan], c=cv[~nan], cmap=DIVERGING, s=5, lw=0, vmin=0, vmax=1, rasterized=True)
        ax.scatter(sv[nan], rank + jit[nan], color=MUTED, s=5, lw=0, rasterized=True)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(range(len(order)), X.columns[order[::-1]], fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("SHAP (contribution to default log-odds)")
    ax.set_title("SHAP impact (colour = feature value: blue low, red high, grey missing)")
    fig.tight_layout()
    return save(fig, "60_shap_beeswarm", out_dir)


def shap_dependence(x: np.ndarray, sv: np.ndarray, feature: str, out_dir=None, name=None):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ok = np.isfinite(x)
    lo, hi = np.nanpercentile(x[ok], [0.5, 99.5])
    m = ok & (x >= lo) & (x <= hi)
    ax.scatter(x[m], sv[m], s=5, color=BLUE, alpha=0.35, lw=0, rasterized=True)
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_xlabel(feature)
    ax.set_ylabel("SHAP")
    ax.set_title(f"SHAP dependence: {feature}")
    fig.tight_layout()
    return save(fig, name or f"61_shap_dependence_{feature}", out_dir)


def approval_curves(curves: dict, at: float = 0.8, out_dir=None):
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for k, (lab, c) in enumerate(curves.items()):
        ax.plot(c["approval_rate"], c["bad_rate"], color=MODEL_COLORS.get(lab, CATEGORICAL[k]), label=lab)
        v = np.interp(at, c["approval_rate"], c["bad_rate"])
        ax.scatter([at], [v], color=MODEL_COLORS.get(lab, CATEGORICAL[k]), s=40, zorder=3, edgecolor=SURFACE, lw=1)
        ax.annotate(f"{v:.2%}", (at, v), xytext=(8, -4 if k else 6), textcoords="offset points", fontsize=9, color=INK)
    ax.axvline(at, color=AXIS, lw=1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.02))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Approval rate (lowest-PD applications approved first)")
    ax.set_ylabel("Default rate among approved")
    ax.set_title("Strategy curve on OOT: portfolio risk at a given approval rate")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return save(fig, "70_approval_curve", out_dir)

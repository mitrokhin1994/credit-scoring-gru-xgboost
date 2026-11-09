import numpy as np
import pytest

from scoring.metrics import apply_grade_edges, approval_curve, fit_grade_edges, gini, psi
from scoring.modeling import penalized_gini
from scoring.selection import TreeBinner, information_value


def test_gini_bounds():
    y = np.array([0, 0, 1, 1])
    assert gini(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert gini(y, [0.9, 0.8, 0.2, 0.1]) == -1.0


def test_psi_stable_vs_shifted():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=20000), rng.normal(size=20000)
    assert psi(a, b) < 0.01
    assert psi(a, b + 1.0) > 0.25
    cat = rng.integers(0, 5, 20000).astype(float)
    assert psi(cat, cat) == 0.0


def test_psi_counts_missing_as_bin():
    a = np.r_[np.zeros(900), np.full(100, np.nan)]
    b = np.r_[np.zeros(500), np.full(500, np.nan)]
    assert psi(a, b) > 0.25


def test_grades_follow_train_quantiles():
    rng = np.random.default_rng(1)
    s = rng.random(10000)
    edges = fit_grade_edges(s, 10)
    g = apply_grade_edges(s, edges)
    assert g.min() == 1 and g.max() == 10
    assert np.all(np.bincount(g)[1:] > 900)


def test_approval_curve_monotone_for_good_score():
    rng = np.random.default_rng(2)
    p = rng.random(5000)
    y = (rng.random(5000) < p).astype(int)
    c = approval_curve(y, p)
    assert c["bad_rate"].iloc[0] < c["bad_rate"].iloc[-1]
    assert abs(c["bad_rate"].iloc[-1] - y.mean()) < 1e-12


def test_penalized_gini():
    assert penalized_gini(0.60, 0.50, 0.03, 0.5) == pytest.approx(0.50 - 0.5 * 0.07)
    assert penalized_gini(0.52, 0.50, 0.03, 0.5) == pytest.approx(0.50)


def test_tree_binner_woe_and_iv():
    rng = np.random.default_rng(3)
    x = rng.normal(size=20000)
    y = (rng.random(20000) < 1 / (1 + np.exp(-(x - 1.5)))).astype(int)
    x[:500] = np.nan
    b = TreeBinner().fit(x, y)
    t = b.table(x, y)
    assert "missing" in set(t["range"])
    non_missing = t[t["bin"] >= 0].sort_values("bin")
    assert np.all(np.diff(non_missing["woe"]) < 0)          # higher x -> riskier -> lower WoE
    assert information_value(b.bins(x), y) > 0.3
    noise = rng.normal(size=20000)
    assert information_value(TreeBinner().fit(noise, y).bins(noise), y) < 0.02

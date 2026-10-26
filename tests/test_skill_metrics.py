"""Known-answer tests of the reference metric library shipped with the hypothesis-testing-stats skill."""
import importlib.util
import os

import numpy as np
import pytest
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
spec = importlib.util.spec_from_file_location("quant_metrics", os.path.join(ROOT, ".agents", "skills", "hypothesis-testing-stats", "scripts", "quant_metrics.py"))
qm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qm)
RNG = np.random.default_rng(0)


def test_drawdown_and_calmar():
    r = np.array([0.10, -0.20, 0.05, 0.10])
    assert qm.max_drawdown(r) == pytest.approx(-0.20)
    assert qm.max_drawdown_duration(r) == 3
    assert qm.max_drawdown(np.array([0.01, 0.02])) == 0.0


def test_sharpe_scales_with_periods():
    r = RNG.normal(0.001, 0.01, 1000)
    assert qm.sharpe(r, 252) == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(252))
    assert np.isfinite(qm.sortino(r, 252))


def test_var_cvar_ordering():
    r = RNG.normal(0, 0.01, 5000)
    assert qm.cvar_hist(r, 0.95) > qm.var_hist(r, 0.95) > 0
    assert qm.kupiec_pof(r < -qm.var_hist(r, 0.95))["p_value"] > 0.5


def test_qlike_minimum_at_truth():
    v = RNG.uniform(1, 2, 100)
    assert qm.qlike(v, v) == pytest.approx(0.0)
    assert qm.qlike(v, v * 1.5) > 0 and qm.qlike(v, v * 0.7) > 0


def test_r2_oos_and_dm():
    y = RNG.normal(size=500)
    assert qm.r2_oos(y, y, np.zeros(500)) == pytest.approx(1.0)
    la, lb = (RNG.normal(size=500) * 0.5) ** 2, (RNG.normal(size=500)) ** 2
    assert qm.diebold_mariano(la, lb)["p_one_sided"] < 0.01          # A has clearly lower loss
    assert qm.diebold_mariano(lb, la)["p_one_sided"] > 0.99
    same = qm.diebold_mariano(la, la + 1e-12 * RNG.normal(size=500))
    assert abs(same["mean_loss_diff"]) < 1e-9


def test_hac_mean_t_matches_scipy_without_lags():
    x = RNG.normal(0.1, 1, 400)
    assert qm.hac_mean_t(x, 0)["t"] == pytest.approx(stats.ttest_1samp(x, 0).statistic * np.sqrt(400 / 399), rel=1e-6)


def test_multiple_testing_known_values():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(qm.bonferroni(p), [0.04, 0.16, 0.12, 0.02])
    assert np.allclose(qm.holm(p), [0.03, 0.06, 0.06, 0.02])
    assert np.allclose(qm.benjamini_hochberg(p), [0.02, 0.04, 0.04, 0.02])


def test_psr_dsr_bounds():
    r = RNG.normal(0.002, 0.01, 750)
    assert 0.9 < qm.probabilistic_sharpe(r, 0.0) <= 1.0
    assert qm.deflated_sharpe(r, 50, 0.0004) < qm.probabilistic_sharpe(r, 0.0)


def test_stationary_bootstrap_detects_difference():
    a = RNG.normal(0.002, 0.01, 800)
    b = RNG.normal(0.0, 0.01, 800)
    diff, p = qm.stationary_bootstrap_sharpe_diff(a, b, 252, n_boot=1000)
    assert diff > 0 and p < 0.05
    assert qm.stationary_bootstrap_sharpe_diff(a, b, 252, n_boot=500, seed=3) == qm.stationary_bootstrap_sharpe_diff(a, b, 252, n_boot=500, seed=3)


def test_circular_shift_null_is_not_rejected():
    x = RNG.normal(size=600)
    y = RNG.normal(size=600)
    _, p = qm.circular_shift_pvalue(y, x, 5, n_draws=200)
    assert p > 0.01


def test_walk_forward_has_no_overlap_and_respects_embargo():
    for tr, te in qm.walk_forward_splits(1000, 5, 400, embargo=10):
        assert tr.max() + 10 < te.min()
        assert len(set(tr) & set(te)) == 0


def test_event_car_uses_only_prior_estimation_window():
    n = 400
    mkt = RNG.normal(0, 0.01, n)
    ret = 1.0 * mkt + RNG.normal(0, 0.002, n)
    ret[300:302] += 0.05
    out = qm.market_model_car(ret, mkt, 300, window=(0, 1))
    assert out["car"] == pytest.approx(0.10, abs=0.02) and out["t"] > 5
    assert np.isnan(qm.market_model_car(ret, mkt, 10)["car"])


def test_auc_and_brier():
    y = np.array([0, 0, 1, 1])
    assert qm.auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert qm.auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert qm.brier(y, y.astype(float)) == 0.0


def test_volatility_estimators_nonnegative():
    o, c = 100.0 * np.ones(5), 101.0 * np.ones(5)
    h, l = 102.0 * np.ones(5), 99.0 * np.ones(5)
    assert (qm.parkinson_var(h, l) > 0).all() and (qm.garman_klass_var(o, h, l, c) > 0).all()
    assert qm.ewma_var(np.array([0.01, 0.02, 0.0]))[-1] > 0

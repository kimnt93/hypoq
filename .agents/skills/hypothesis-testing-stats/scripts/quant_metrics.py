"""
Reference library of evaluation metrics and inference procedures for quantitative research.

NOT part of the (frozen) hypoq package: copy the functions you need into your own researches/<id>/code/run_pipeline.py so that the
study stays self-contained and reproducible. Everything works on 1-d NumPy arrays (Polars Series -> `.to_numpy()`), returns plain Python
floats / dicts that can go straight into the results JSON, and is deterministic given `seed`.

Conventions
- `r` is a vector of simple periodic returns (net of costs when you evaluate a strategy); `periods` is the annualisation factor
  (252 daily equities, 365 daily crypto, 52 weekly, 12 monthly).
- Forecast errors are `e = y - yhat`; losses are "lower is better". p-values are one-sided unless the name says `two_sided`.
- NaN are never filled: drop them explicitly (`x[~np.isnan(x)]`) and report how many were dropped.
Sections: 1 performance, 2 risk, 3 forecasting, 4 cross-sectional signals, 5 inference, 6 multiple testing, 7 event studies,
8 time series, 9 volatility estimators, 10 validation splits, 11 classification.
"""

from math import e as _E
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import statsmodels.api as sm
from scipy import stats

PERIODS = {"daily_equity": 252, "daily_crypto": 365, "weekly": 52, "monthly": 12, "quarterly": 4, "annual": 1}


# ---------------------------------------------------------------- 1. performance
def ann_return(r: np.ndarray, periods: int) -> float:
    """Geometric (CAGR-style) annualised return."""
    return float(np.prod(1.0 + r) ** (periods / len(r)) - 1.0)


def ann_vol(r: np.ndarray, periods: int) -> float:
    return float(r.std(ddof=1) * np.sqrt(periods))


def sharpe(r: np.ndarray, periods: int, rf: float = 0.0) -> float:
    """Annualised Sharpe ratio; `rf` is the risk-free rate per period (0 is fine for long/short or excess returns)."""
    x = r - rf
    return float(x.mean() / x.std(ddof=1) * np.sqrt(periods))


def sortino(r: np.ndarray, periods: int, target: float = 0.0) -> float:
    downside = np.sqrt(np.mean(np.minimum(r - target, 0.0) ** 2))
    return float((r.mean() - target) / downside * np.sqrt(periods))


def equity_curve(r: np.ndarray) -> np.ndarray:
    return np.cumprod(1.0 + r)


def drawdown_series(r: np.ndarray) -> np.ndarray:
    eq = equity_curve(r)
    return eq / np.maximum.accumulate(eq) - 1.0


def max_drawdown(r: np.ndarray) -> float:
    return float(drawdown_series(r).min())


def max_drawdown_duration(r: np.ndarray) -> int:
    """Longest number of consecutive periods spent below a previous equity peak."""
    dd = drawdown_series(r) < 0
    best = run = 0
    for flag in dd:
        run = run + 1 if flag else 0
        best = max(best, run)
    return int(best)


def calmar(r: np.ndarray, periods: int) -> float:
    mdd = abs(max_drawdown(r))
    return float(ann_return(r, periods) / mdd) if mdd > 0 else float("nan")


def hit_rate(r: np.ndarray) -> float:
    return float(np.mean(r > 0))


def profit_factor(r: np.ndarray) -> float:
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    return float(gains / losses) if losses > 0 else float("nan")


def alpha_beta(r: np.ndarray, bench: np.ndarray, periods: int, lags: int = 5) -> Dict[str, float]:
    """CAPM-style regression r = a + b * bench with Newey-West errors; alpha is annualised (arithmetic)."""
    fit = sm.OLS(r, sm.add_constant(bench)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {"alpha_ann": float(fit.params[0] * periods), "alpha_t": float(fit.tvalues[0]), "beta": float(fit.params[1]), "beta_t": float(fit.tvalues[1]),
            "r2": float(fit.rsquared)}


def information_ratio(r: np.ndarray, bench: np.ndarray, periods: int) -> float:
    active = r - bench
    return float(active.mean() / active.std(ddof=1) * np.sqrt(periods))


def tracking_error(r: np.ndarray, bench: np.ndarray, periods: int) -> float:
    return float((r - bench).std(ddof=1) * np.sqrt(periods))


def capture_ratios(r: np.ndarray, bench: np.ndarray) -> Dict[str, float]:
    up, dn = bench > 0, bench < 0
    return {"up_capture": float(r[up].mean() / bench[up].mean()), "down_capture": float(r[dn].mean() / bench[dn].mean())}


def turnover(weights: np.ndarray) -> float:
    """Mean one-way turnover per period from a (T, N) weight matrix (sum of absolute weight changes)."""
    return float(np.abs(np.diff(weights, axis=0)).sum(axis=1).mean())


def breakeven_cost(gross: np.ndarray, weights: np.ndarray) -> float:
    """One-way cost (decimal per unit turnover) at which the mean net return is zero."""
    t = turnover(weights)
    return float(gross.mean() / t) if t > 0 else float("nan")


def effective_n(w: np.ndarray) -> float:
    """Inverse Herfindahl: the number of equally weighted positions that has the same concentration."""
    w = np.abs(w) / np.abs(w).sum()
    return float(1.0 / np.sum(w ** 2))


def summary(r: np.ndarray, periods: int) -> Dict[str, float]:
    return {"ann_return": ann_return(r, periods), "ann_vol": ann_vol(r, periods), "sharpe": sharpe(r, periods), "sortino": sortino(r, periods),
            "max_drawdown": max_drawdown(r), "calmar": calmar(r, periods), "hit_rate": hit_rate(r), "skew": float(stats.skew(r)),
            "kurtosis": float(stats.kurtosis(r, fisher=False)), "n": int(len(r))}


# ---------------------------------------------------------------- 2. risk
def var_hist(r: np.ndarray, level: float = 0.95) -> float:
    """Historical Value-at-Risk as a positive loss number."""
    return float(-np.quantile(r, 1.0 - level))


def cvar_hist(r: np.ndarray, level: float = 0.95) -> float:
    """Historical expected shortfall (mean loss beyond VaR) as a positive number."""
    cut = np.quantile(r, 1.0 - level)
    return float(-r[r <= cut].mean())


def var_cornish_fisher(r: np.ndarray, level: float = 0.95) -> float:
    z = stats.norm.ppf(1.0 - level)
    s, k = stats.skew(r), stats.kurtosis(r)  # excess kurtosis
    zcf = z + (z ** 2 - 1) * s / 6 + (z ** 3 - 3 * z) * k / 24 - (2 * z ** 3 - 5 * z) * s ** 2 / 36
    return float(-(r.mean() + zcf * r.std(ddof=1)))


def kupiec_pof(violations: np.ndarray, level: float = 0.95) -> Dict[str, float]:
    """Kupiec unconditional-coverage test of VaR exceedances (boolean array). H0: exceedance rate = 1 - level."""
    n, x, p = len(violations), int(violations.sum()), 1.0 - level
    if x in (0, n):
        lr = -2 * (n * np.log(1 - p) if x == 0 else n * np.log(p))
    else:
        lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p) - (n - x) * np.log(1 - x / n) - x * np.log(x / n))
    return {"lr": float(lr), "p_value": float(stats.chi2.sf(lr, 1)), "exceed_rate": float(x / n)}


# ---------------------------------------------------------------- 3. forecasting
def mse(e: np.ndarray) -> float:
    return float(np.mean(e ** 2))


def rmse(e: np.ndarray) -> float:
    return float(np.sqrt(np.mean(e ** 2)))


def mae(e: np.ndarray) -> float:
    return float(np.mean(np.abs(e)))


def r2_oos(y: np.ndarray, yhat: np.ndarray, benchmark: np.ndarray) -> float:
    """Campbell-Thompson out-of-sample R^2 against a benchmark forecast (e.g. the expanding historical mean). > 0 beats the benchmark."""
    return float(1.0 - np.sum((y - yhat) ** 2) / np.sum((y - benchmark) ** 2))


def qlike(realised_var: np.ndarray, forecast_var: np.ndarray) -> float:
    """QLIKE loss for variance forecasts (robust to noisy proxies, Patton 2011). Both arguments must be strictly positive variances."""
    ratio = realised_var / forecast_var
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def directional_accuracy(y: np.ndarray, yhat: np.ndarray) -> Dict[str, float]:
    hit = np.sign(y) == np.sign(yhat)
    p = float(stats.binomtest(int(hit.sum()), len(hit), 0.5, alternative="greater").pvalue)
    return {"accuracy": float(hit.mean()), "p_value_one_sided": p}


def hac_variance_of_mean(d: np.ndarray, lags: int) -> float:
    """Newey-West (Bartlett) long-run variance of the sample mean of d."""
    n = len(d)
    x = d - d.mean()
    v = np.dot(x, x) / n
    for k in range(1, lags + 1):
        v += 2.0 * (1.0 - k / (lags + 1.0)) * np.dot(x[k:], x[:-k]) / n
    return float(v / n)


def hac_mean_t(x: np.ndarray, lags: int = 5) -> Dict[str, float]:
    """t-statistic of the mean of x with Newey-West errors; one-sided p-value for H1: mean > 0 (normal approximation)."""
    t = float(x.mean() / np.sqrt(hac_variance_of_mean(x, lags)))
    return {"mean": float(x.mean()), "t": t, "p_one_sided": float(stats.norm.sf(t))}


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1) -> Dict[str, float]:
    """
    Diebold-Mariano with the Harvey-Leybourne-Newbold small-sample correction.
    Pass per-observation losses (e.g. squared errors or QLIKE terms). H1: model A has LOWER expected loss than B (one-sided).
    """
    d = loss_a - loss_b
    n = len(d)
    stat = d.mean() / np.sqrt(hac_variance_of_mean(d, max(h - 1, 0)))
    hln = stat * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return {"dm": float(stat), "dm_hln": float(hln), "p_one_sided": float(stats.t.cdf(hln, n - 1)), "mean_loss_diff": float(d.mean())}


def clark_west(y: np.ndarray, yhat_small: np.ndarray, yhat_large: np.ndarray, h: int = 1) -> Dict[str, float]:
    """Clark-West test for NESTED models. H1: the larger model has lower MSPE (one-sided)."""
    e1, e2 = y - yhat_small, y - yhat_large
    f = e1 ** 2 - (e2 ** 2 - (yhat_small - yhat_large) ** 2)
    res = hac_mean_t(f, max(h - 1, 0))
    return {"cw_t": res["t"], "p_one_sided": res["p_one_sided"], "mean_adjusted_diff": res["mean"]}


# ---------------------------------------------------------------- 4. cross-sectional signals
# Polars patterns (per date): rank IC and quantile buckets
#   ic  = df.group_by("date").agg(pl.corr(pl.col("pred").rank(), pl.col("ret_next").rank()).alias("ic"))
#   bkt = df.with_columns(((pl.col("pred").rank("ordinal").over("date") - 1) * 5 // pl.len().over("date")).alias("q"))
def icir(ic: np.ndarray, periods: int) -> float:
    """Information-coefficient information ratio: mean(IC) / std(IC), annualised."""
    return float(ic.mean() / ic.std(ddof=1) * np.sqrt(periods))


def quantile_spread(signal: np.ndarray, ret_next: np.ndarray, q: int = 5) -> Dict[str, float]:
    """One cross-section: mean next-period return of the top and bottom signal quantile and their spread."""
    order = np.argsort(signal, kind="stable")
    k = len(signal) // q
    low, high = ret_next[order[:k]].mean(), ret_next[order[-k:]].mean()
    return {"top": float(high), "bottom": float(low), "spread": float(high - low)}


def fama_macbeth(y_by_period: List[np.ndarray], x_by_period: List[np.ndarray], lags: int = 5) -> Dict[str, np.ndarray]:
    """
    Fama-MacBeth: one cross-sectional OLS per period (constant + regressors), then the time-series mean of each slope with Newey-West t.
    x_by_period[t] has shape (n_t, k). Returns arrays over [intercept, slope_1..slope_k].
    """
    betas = []
    for y, x in zip(y_by_period, x_by_period):
        X = np.column_stack([np.ones(len(y)), x])
        betas.append(np.linalg.lstsq(X, y, rcond=None)[0])
    B = np.array(betas)
    tvals = np.array([B[:, j].mean() / np.sqrt(hac_variance_of_mean(B[:, j], lags)) for j in range(B.shape[1])])
    return {"mean": B.mean(axis=0), "t": tvals, "p_two_sided": 2 * stats.norm.sf(np.abs(tvals))}


# ---------------------------------------------------------------- 5. inference
def hac_ols(y: np.ndarray, x: np.ndarray, lags: int):
    """OLS of y on a constant and x (1-d or 2-d) with Newey-West HAC errors (statsmodels results object)."""
    return sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})


def _stationary_indices(rng: np.random.Generator, n: int, block: float, n_boot: int) -> np.ndarray:
    """(n_boot, n) index matrix of Politis-Romano stationary-bootstrap resamples with mean block length `block`."""
    restart = rng.random((n_boot, n)) < 1.0 / block
    restart[:, 0] = True
    starts = rng.integers(0, n, size=(n_boot, n))
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    for k in range(1, n):
        idx[:, k] = np.where(restart[:, k], starts[:, k], (idx[:, k - 1] + 1) % n)
    return idx


def stationary_bootstrap_sharpe_diff(a: np.ndarray, b: np.ndarray, periods: int = 252, n_boot: int = 5000,
                                     block: float = 10.0, seed: int = 11) -> Tuple[float, float]:
    """One-sided p-value for H1: Sharpe(a) > Sharpe(b), recentred under H0. Returns (annualised difference, p)."""
    rng = np.random.default_rng(seed)
    idx = _stationary_indices(rng, len(a), block, n_boot)
    sh = lambda m: m.mean(axis=1) / m.std(axis=1, ddof=1)
    obs = a.mean() / a.std(ddof=1) - b.mean() / b.std(ddof=1)
    diffs = sh(a[idx]) - sh(b[idx])
    return float(obs * np.sqrt(periods)), float((np.sum(diffs - diffs.mean() >= obs) + 1) / (n_boot + 1))


def stationary_bootstrap_mean_pvalue(x: np.ndarray, n_boot: int = 5000, block: float = 10.0, seed: int = 11) -> float:
    """One-sided p-value for H1: mean(x) > 0, recentred stationary bootstrap."""
    idx = _stationary_indices(np.random.default_rng(seed), len(x), block, n_boot)
    means = x[idx].mean(axis=1)
    return float((np.sum(means - means.mean() >= x.mean()) + 1) / (n_boot + 1))


def bootstrap_ci(x: np.ndarray, stat: Callable[[np.ndarray], float], n_boot: int = 5000, block: float = 10.0, level: float = 0.95,
                 seed: int = 11) -> Tuple[float, float]:
    """Percentile confidence interval of any statistic of a time series via the stationary bootstrap (write `stat` yourself)."""
    idx = _stationary_indices(np.random.default_rng(seed), len(x), block, n_boot)
    vals = np.array([stat(x[i]) for i in idx])
    lo, hi = np.quantile(vals, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def circular_shift_pvalue(y: np.ndarray, x: np.ndarray, lags: int, n_draws: int = 2000, seed: int = 7) -> Tuple[float, float]:
    """
    Placebo for y_t = a + b x_t + e_t: rotate x by a random offset (keeps its autocorrelation, breaks only the alignment with y) and
    recompute the HAC t-statistic of b. Returns (observed t, one-sided p for H1: b > 0).
    """
    rng = np.random.default_rng(seed)
    n, lo = len(y), lags + 10
    t_obs = float(hac_ols(y, x, lags).tvalues[1])
    sims = np.array([hac_ols(y, np.roll(x, int(rng.integers(lo, n - lo))), lags).tvalues[1] for _ in range(n_draws)])
    return t_obs, float((np.sum(sims >= t_obs) + 1) / (n_draws + 1))


def perm_placebo_mean(event_values: np.ndarray, pool: np.ndarray, n_draws: int = 10000, seed: int = 5) -> Tuple[float, float]:
    """Mean of `event_values` vs. means of equally sized random draws from `pool` of non-event values (one-sided p: larger)."""
    sims = np.random.default_rng(seed).choice(pool, size=(n_draws, len(event_values))).mean(axis=1)
    return float(sims.mean()), float((np.sum(sims >= event_values.mean()) + 1) / (n_draws + 1))


def probabilistic_sharpe(r: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """Bailey-Lopez de Prado PSR: probability that the true per-period Sharpe exceeds `sr_benchmark` (per period), given skew and kurtosis."""
    n, sr = len(r), r.mean() / r.std(ddof=1)
    s, k = stats.skew(r), stats.kurtosis(r, fisher=False)
    return float(stats.norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / np.sqrt(1 - s * sr + (k - 1) / 4 * sr ** 2)))


def deflated_sharpe(r: np.ndarray, n_trials: int, trial_sharpe_var: float) -> float:
    """
    Deflated Sharpe ratio: PSR against the Sharpe expected from the best of `n_trials` unskilled strategies. Use it whenever
    you picked this strategy out of several variants; `trial_sharpe_var` is the variance of the per-period Sharpe across the trials.
    """
    g = 0.5772156649015329
    sr0 = np.sqrt(trial_sharpe_var) * ((1 - g) * stats.norm.ppf(1 - 1.0 / n_trials) + g * stats.norm.ppf(1 - 1.0 / (n_trials * _E)))
    return probabilistic_sharpe(r, sr0)


# ---------------------------------------------------------------- 6. multiple testing
def bonferroni(p: np.ndarray) -> np.ndarray:
    return np.minimum(np.asarray(p) * len(p), 1.0)


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    out = np.empty_like(p)
    out[order] = np.minimum(adj, 1.0)
    return out


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """FDR-adjusted p-values (q-values)."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    n = len(p)
    adj = np.minimum.accumulate((p[order] * n / (np.arange(n) + 1))[::-1])[::-1]
    out = np.empty_like(p)
    out[order] = np.minimum(adj, 1.0)
    return out


# ---------------------------------------------------------------- 7. event studies
def market_model_car(ret: np.ndarray, mkt: np.ndarray, event_idx: int, est: Tuple[int, int] = (-250, -31),
                     window: Tuple[int, int] = (0, 1)) -> Dict[str, float]:
    """
    Market-model cumulative abnormal return for one event. `ret` and `mkt` are aligned return arrays and `event_idx` is the position of day 0.
    Estimation uses days event_idx+est[0] .. event_idx+est[1] only (strictly before the window). Returns NaN fields when history is missing.
    """
    a, b = event_idx + est[0], event_idx + est[1] + 1
    lo, hi = event_idx + window[0], event_idx + window[1] + 1
    if a < 0 or hi > len(ret):
        return {"car": float("nan"), "t": float("nan")}
    fit = sm.OLS(ret[a:b], sm.add_constant(mkt[a:b])).fit()
    ar = ret[lo:hi] - fit.params[0] - fit.params[1] * mkt[lo:hi]
    sigma = np.sqrt(fit.scale)
    return {"car": float(ar.sum()), "t": float(ar.sum() / (sigma * np.sqrt(len(ar))))}


def event_cross_section(cars: np.ndarray) -> Dict[str, float]:
    """Cross-sectional t-test and sign test of event CARs (H0: mean CAR = 0; two-sided)."""
    cars = cars[~np.isnan(cars)]
    t = stats.ttest_1samp(cars, 0.0)
    sign = stats.binomtest(int((cars > 0).sum()), len(cars), 0.5)
    return {"n": int(len(cars)), "mean_car": float(cars.mean()), "t": float(t.statistic), "p_t": float(t.pvalue), "p_sign": float(sign.pvalue)}


# ---------------------------------------------------------------- 8. time series
def adf_test(x: np.ndarray) -> Dict[str, float]:
    from statsmodels.tsa.stattools import adfuller
    stat, p, lag = adfuller(x[~np.isnan(x)], autolag="AIC")[:3]
    return {"stat": float(stat), "p_value": float(p), "lags": int(lag), "stationary_at_5pct": bool(p < 0.05)}


def ljung_box(x: np.ndarray, lags: int = 10) -> float:
    """p-value of the Ljung-Box test of no autocorrelation up to `lags`."""
    from statsmodels.stats.diagnostic import acorr_ljungbox
    return float(acorr_ljungbox(x, lags=[lags]).iloc[0, 1])


def half_life(x: np.ndarray) -> float:
    """Mean-reversion half-life (in periods) from the AR(1) fit x_t = c + phi x_{t-1}; NaN if phi is not in (0, 1)."""
    phi = sm.OLS(x[1:], sm.add_constant(x[:-1])).fit().params[1]
    return float(-np.log(2) / np.log(phi)) if 0 < phi < 1 else float("nan")


def granger_p(y: np.ndarray, x: np.ndarray, lags: int = 5) -> float:
    """p-value (F test) that x Granger-causes y with `lags` lags."""
    from statsmodels.tsa.stattools import grangercausalitytests
    res = grangercausalitytests(np.column_stack([y, x]), maxlag=lags, verbose=False)
    return float(res[lags][0]["ssr_ftest"][1])


# ---------------------------------------------------------------- 9. volatility estimators (per-bar arrays of open, high, low, close)
def parkinson_var(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    return np.log(high / low) ** 2 / (4 * np.log(2))


def garman_klass_var(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    return 0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2


def rogers_satchell_var(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)


def ewma_var(r: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """RiskMetrics EWMA variance; element t uses returns up to and including t."""
    v = np.empty_like(r)
    v[0] = r[0] ** 2
    for t in range(1, len(r)):
        v[t] = lam * v[t - 1] + (1 - lam) * r[t] ** 2
    return v


# ---------------------------------------------------------------- 10. validation splits
def walk_forward_splits(n: int, n_splits: int, min_train: int, embargo: int = 0, expanding: bool = True,
                        window: Optional[int] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Chronological (train_idx, test_idx) pairs. `embargo` observations between train end and test start remove label overlap
    (set it to at least the forecast horizon). `expanding=False` with `window` gives a rolling window.
    """
    edges = np.linspace(min_train + embargo, n, n_splits + 1, dtype=int)
    for a, b in zip(edges[:-1], edges[1:]):
        train_end = a - embargo
        train_start = 0 if expanding else max(0, train_end - (window or min_train))
        yield np.arange(train_start, train_end), np.arange(a, b)


# ---------------------------------------------------------------- 11. classification / probabilistic forecasts
def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(y: np.ndarray, score: np.ndarray) -> float:
    """Rank-based ROC AUC (Mann-Whitney)."""
    r = stats.rankdata(score)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

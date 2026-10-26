---
name: hypothesis-testing-stats
description: Evaluation metrics and statistical inference for quantitative research - performance, risk, forecast accuracy, signal quality, HAC / bootstrap / placebo tests, multiple-testing control, event studies, validation splits - plus rules for writing a correct custom metric with Polars and NumPy when the catalogue does not cover it.
version: 2.0.0
author: hypoq
tags: [statistics, econometrics, metrics, hypothesis-testing, newey-west, bootstrap, backtest-evaluation]
---

# Hypothesis Testing, Metrics and Evaluation

The `hypoq` package is frozen and contains no statistics. Every study keeps its own code in `researches/<id>/code/run_pipeline.py`.
This skill supplies (a) a decision framework, (b) a reference library you **copy from**, and (c) rules for writing a metric that is not in the catalogue.

Reference library: [`scripts/quant_metrics.py`](scripts/quant_metrics.py) (NumPy / SciPy / statsmodels, known-answer tests in `tests/test_skill_metrics.py`).
Copy the functions you need into your pipeline; do not import from the skill folder, so the study stays self-contained and reproducible.
`res_007` to `res_012` show the copy-in pattern.

## 1. Decision framework (fix it before you look at results)
1. State $H_0$, $H_1$, the test statistic, the sidedness, the significance level (default one-sided $\alpha=0.05$) and the multiple-testing correction.
2. Compute the statistic once, on data the model never saw. Do not tune on the evaluation window.
3. Decision is derived by code: `ACCEPTED` = $H_1$ supported by the pre-specified test(s) (when two tests are specified, both must pass); `REJECTED` = not supported; `INCONCLUSIVE` = the pre-specified tests disagree.
4. Report the economic size next to the p-value (basis points, Sharpe difference, drawdown), net of a stated cost, and say how many observations are effectively independent.

## 2. Catalogue (all in `quant_metrics.py`)

| Need | Functions | Notes |
| :--- | :--- | :--- |
| Return and risk-adjusted performance | `ann_return`, `ann_vol`, `sharpe`, `sortino`, `calmar`, `hit_rate`, `profit_factor`, `summary` | `periods`: 252 daily equities, 365 crypto, 52 weekly, 12 monthly |
| Drawdowns | `drawdown_series`, `max_drawdown`, `max_drawdown_duration` | from simple net returns |
| Benchmark-relative | `alpha_beta` (HAC), `information_ratio`, `tracking_error`, `capture_ratios` | benchmark must be investable and aligned by date |
| Trading frictions | `turnover`, `breakeven_cost`, `effective_n` | `weights` is a (T, N) matrix |
| Tail risk | `var_hist`, `cvar_hist`, `var_cornish_fisher`, `kupiec_pof` | losses are positive numbers |
| Forecast accuracy | `mse`, `rmse`, `mae`, `r2_oos` (Campbell-Thompson), `qlike` (variance), `directional_accuracy` | benchmark forecast must itself be out of sample |
| Forecast comparison | `diebold_mariano` (HLN), `clark_west` (nested models) | pass per-observation losses |
| Signal quality | `icir`, `quantile_spread`, `fama_macbeth` | per-date rank IC: Polars pattern in the file |
| Regression inference | `hac_ols`, `hac_mean_t`, `hac_variance_of_mean` | lags $\ge$ horizon - 1 for overlapping returns |
| Resampling | `stationary_bootstrap_sharpe_diff`, `stationary_bootstrap_mean_pvalue`, `bootstrap_ci`, `circular_shift_pvalue`, `perm_placebo_mean` | seeded and vectorised |
| Selection bias | `probabilistic_sharpe`, `deflated_sharpe` | use when the strategy was picked from several variants |
| Multiple testing | `bonferroni`, `holm`, `benjamini_hochberg` | apply across every test you ran, not only the best |
| Event studies | `market_model_car`, `event_cross_section` | estimation window ends before the event window |
| Time series | `adf_test`, `ljung_box`, `half_life`, `granger_p` | check stationarity of regressors in levels |
| Volatility proxies | `parkinson_var`, `garman_klass_var`, `rogers_satchell_var`, `ewma_var` | per-bar arrays |
| Validation | `walk_forward_splits` (expanding / rolling, embargo) | embargo $\ge$ label horizon |
| Probabilistic forecasts | `brier`, `log_loss`, `auc` | |

Choosing a test: compare two strategies -> `stationary_bootstrap_sharpe_diff`; one strategy's mean excess return -> `hac_mean_t` or `stationary_bootstrap_mean_pvalue`;
predictive regression with overlapping or persistent data -> `hac_ols` **and** `circular_shift_pvalue`; two forecasts -> `diebold_mariano` (non-nested) or `clark_west` (nested);
an event window -> `market_model_car` + `perm_placebo_mean` on non-event days of the same asset; many variants tried -> `deflated_sharpe` and `holm`.

## 3. When the metric you need is not in the catalogue
Write it yourself in the pipeline. This is expected (new method, user request, domain-specific measure). Follow these rules:

1. **Write the definition first** as a comment or docstring: formula, inputs and units, which direction is better, what is annualised.
2. **Vectorise.** Use Polars expressions for panels and group computations, NumPy for arrays; no Python loops over rows or dates unless the recursion is genuinely sequential (EWMA, path-dependent logic).
   ```python
   # per-date rank IC and top-minus-bottom quintile spread in Polars
   ic = df.group_by("date").agg(pl.corr(pl.col("pred").rank(), pl.col("ret_next").rank()).alias("ic"))
   q = ((pl.col("pred").rank("ordinal").over("date") - 1) * 5 // pl.len().over("date")).alias("q")
   spread = (df.with_columns(q).group_by(["date", "q"]).agg(pl.col("ret_next").mean())
               .pivot(on="q", index="date", values="ret_next").with_columns((pl.col("4") - pl.col("0")).alias("spread")))
   ```
3. **No look-ahead.** Anything computed on a rolling or expanding window must use only data up to the decision time: `shift(1)` signals before multiplying by next-period returns, `rolling_*` windows end at $t$, estimation windows end before the event, and macro data are lagged by their publication delay (`join_asof(..., strategy="backward")`).
4. **Handle missing data explicitly.** Never `fill_null(0)` a return, volume or price. Drop the rows (count them and report the count) or keep `null`/`NaN` and use NaN-aware reductions.
5. **Mind the units.** Simple vs log returns, per-period vs annualised (`sqrt(periods)` for mean/vol ratios, geometric compounding for return), percent vs decimal, variance vs volatility.
6. **Use the right dependence structure.** Overlapping horizons, persistent regressors and serially correlated losses need HAC errors or a block / stationary bootstrap, not an i.i.d. t-test. Use the same block length when comparing two series.
7. **Seed everything** (`np.random.default_rng(seed)`) so a re-run produces identical numbers; `make seed` must reproduce the stored JSON.
8. **Check it against a known answer before using it**: a case where you can derive the result by hand (constant series, perfectly correlated signal, a drawdown you can count). Put the check in a throw-away script or in `tests/`; synthetic data is allowed for testing a function, never for producing a reported result.
9. **Prefer a one-sided, pre-specified test** whose rejection region you stated before running; report two-sided results only as descriptive.
10. **Return plain floats / lists in the results dict** (`float(x)`), then format every number in the paper from that dict.

## 4. Pitfalls checklist
- Survivorship: the equity universes are today's large caps. Say so in the Limitations section and avoid claiming a tradable edge.
- Duplicate listings: some tickers appear in more than one regional file (e.g. `BABA`, `7203.T`); deduplicate on `(symbol, date)` before pooling.
- Overlapping returns: effective sample size is about $N/h$.
- Cherry-picking: if you tried $k$ variants, correct for $k$ (Holm / Bonferroni) or report the Deflated Sharpe ratio.
- Costs: report results net of an explicit cost per unit of turnover and the break-even cost.
- Stationarity: regress returns or changes, not trending levels, unless you test for cointegration.
- Small $N$ of independent events (two inversion episodes, one tightening cycle): keep the conclusion modest even when $p<0.05$.

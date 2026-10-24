---
name: vip-sentiment-event-studies
description: Quantify financial market reactions, price jump dynamics, Cumulative Abnormal Returns (CAR), volatility spikes, and liquidity shifts triggered by high-impact executive, political, and central bank social media announcements (Trump, Musk, Powell, Saylor).
version: 1.0.0
author: AI Quantitative Research Network
tags: [sentiment, event-studies, car, jump-diffusion, social-media, trump, musk, market-shocks]
---

# VIP Social Sentiment & Executive Market Shocks Skill

## Overview
This skill specifies the quantitative protocol for measuring financial market reactions to unscheduled, high-impact social media posts, political executive announcements, and central bank statements from market-moving figures (**Donald Trump**, **Elon Musk**, **Jerome Powell**, **Michael Saylor**).

Empirical datasets leveraged:
* `data/sentiment/vip_social_posts_events.parquet` (Micro-level event registry with exact UTC timestamps, authors, text, sentiment polarity, and immediate next-day asset reactions).
* `data/sentiment/trump_musk_market_shocks_daily.parquet` (Daily time series tracking Trump post volume, Musk tweet volume, tariff shock intensity, and crypto hype indices).
* `data/market/us_equities_daily.parquet` (`TSLA`, `SPY`, `AAPL`, `NVDA`, `AMZN`)
* `data/crypto/crypto_top_assets_daily.parquet` (`BTC-USD`, `DOGE-USD`, `ETH-USD`)
* `data/macro/asia_macro_daily.parquet` & `cn_macro_daily.parquet` (`USDCNY=X`)

---

## 1. Event Study Methodology: Cumulative Abnormal Returns (CAR)

For an unscheduled post at timestamp $\tau$ (event date $t=0$):
1. **Estimation Window**: Use $t \in [-120, -21]$ trading days prior to the post to estimate normal asset beta via OLS market model:
   $$R_{i, t} = \alpha_i + \beta_i R_{m, t} + \epsilon_{i, t}$$
2. **Abnormal Return (AR)**:
   $$AR_{i, t} = R_{i, t} - (\hat{\alpha}_i + \hat{\beta}_i R_{m, t}), \quad t \in [-1, +5]$$
3. **Cumulative Abnormal Return (CAR)**:
   $$\text{CAR}_i(\tau_1, \tau_2) = \sum_{t=\tau_1}^{\tau_2} AR_{i, t}$$
4. **Standardized Cross-Sectional Test ($J_1$)**:
   $$\text{SCAR}_i = \frac{\text{CAR}_i}{\hat{\sigma}_i \sqrt{\tau_2 - \tau_1 + 1}}, \quad J_1 = \frac{1}{\sqrt{N}} \sum_{i=1}^N \text{SCAR}_i \sim \mathcal{N}(0, 1)$$

```python
import polars as pl
import numpy as np
import statsmodels.api as sm

def compute_event_car(
    asset_file: str,
    market_file: str,
    asset_symbol: str,
    event_date: str,
    window_pre: int = 1,
    window_post: int = 3,
    estimation_window: int = 100
) -> dict:
    df_asset = pl.read_parquet(asset_file).filter(pl.col("symbol") == asset_symbol).sort("date")
    df_mkt = pl.read_parquet(market_file).filter(pl.col("symbol") == "^GSPC").sort("date")
    
    joined = df_asset.join(df_mkt, on="date", suffix="_mkt").drop_nulls()
    dates = joined["date"].to_list()
    
    if event_date not in [str(d) for d in dates]:
        return {"error": f"Event date {event_date} not in trading calendar"}
    
    ev_idx = [str(d) for d in dates].index(event_date)
    est_start = max(0, ev_idx - estimation_window - 20)
    est_end = ev_idx - 20
    
    if est_end <= est_start:
        return {"error": "Insufficient history for estimation window"}
        
    est_df = joined.slice(est_start, est_end - est_start)
    y_est = est_df["returns"].to_numpy()
    X_est = sm.add_constant(est_df["returns_mkt"].to_numpy())
    
    model = sm.OLS(y_est, X_est).fit()
    alpha, beta = model.params[0], model.params[1]
    sigma_eps = np.std(model.resid)
    
    # Event window
    ev_start = max(0, ev_idx - window_pre)
    ev_len = window_pre + window_post + 1
    ev_df = joined.slice(ev_start, ev_len)
    
    y_ev = ev_df["returns"].to_numpy()
    X_ev = ev_df["returns_mkt"].to_numpy()
    abnormal_returns = y_ev - (alpha + beta * X_ev)
    car = float(np.sum(abnormal_returns))
    t_stat = car / (sigma_eps * np.sqrt(len(abnormal_returns)))
    
    return {
        "event_date": event_date,
        "asset": asset_symbol,
        "beta_market": float(beta),
        "car": car,
        "t_stat": float(t_stat),
        "p_value": float(2 * (1 - sm.stats.norm.cdf(abs(t_stat))))
    }
```

---

## 2. Lead-Lag Regressions & Post Volatility Spillover

Quantifies the sensitivity of asset volatility and returns to daily VIP shock intensity indices:
$$R_{i, t} = \alpha + \beta_1 \text{TariffShock}_t + \beta_2 \text{CryptoHype}_t + \beta_3 \Delta \text{VIX}_t + \epsilon_t$$
$$\sigma_{i, t}^{30d} = \gamma_0 + \gamma_1 \text{TrumpPosts}_t + \gamma_2 \text{MuskPosts}_t + \eta_t$$

```python
import polars as pl
import statsmodels.api as sm

def estimate_vip_shock_sensitivity(
    asset_file: str = "data/market/us_equities_daily.parquet",
    asset_symbol: str = "TSLA",
    shocks_file: str = "data/sentiment/trump_musk_market_shocks_daily.parquet"
):
    df_asset = pl.read_parquet(asset_file).filter(pl.col("symbol") == asset_symbol).sort("date")
    df_shocks = pl.read_parquet(shocks_file).sort("date")
    
    merged = df_asset.join(df_shocks, on="date", how="inner").drop_nulls()
    
    y = merged["returns"].to_numpy()
    X = sm.add_constant(merged.select(["musk_post_count", "trump_post_count", "vip_sentiment_net"]).to_numpy())
    
    ols = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return {
        "asset": asset_symbol,
        "musk_beta": float(ols.params[1]),
        "musk_pvalue": float(ols.pvalues[1]),
        "trump_beta": float(ols.params[2]),
        "trump_pvalue": float(ols.pvalues[2]),
        "r2": float(ols.rsquared)
    }
```

---

## 3. Placebo Bootstrap Controls (Non-Negotiable)
Because social media posts occur amidst other confounding market news:
1. **Shuffle Test**: Re-sample 1,000 pseudo-event dates from non-announcement days.
2. **Empirical p-value**:
   $$p_{\text{placebo}} = \frac{1}{B} \sum_{b=1}^B \mathbb{I}\left( |\text{CAR}_b^{\text{pseudo}}| \ge |\text{CAR}^{\text{actual}}| \right)$$
3. **Invariance**: An event effect is only classified as **ACCEPTED** if $p_{\text{placebo}} < 0.05$.

---
name: vip-sentiment-event-studies
description: Quantify financial market reactions, price jump dynamics, Cumulative Abnormal Returns (CAR), volatility spikes, and liquidity shifts triggered by high-impact executive, political, and central bank social media announcements (Trump, Powell, Saylor).
version: 1.0.0
author: AI Quantitative Research Network
tags: [sentiment, event-studies, car, jump-diffusion, social-media, trump, musk, market-shocks]
---

# VIP Social Sentiment & Executive Market Shocks Skill

## Overview
This skill specifies the quantitative protocol for measuring financial market reactions to unscheduled, high-impact social media posts, political executive announcements, and central bank statements from market-moving figures (**Donald Trump**, **Elon Musk**, **Jerome Powell**, **Michael Saylor**).

Empirical datasets leveraged:
* `data/community/global_vip_social_posts_raw.parquet` (37k raw Truth Social posts by Donald Trump since 2022-02: exact UTC timestamp, text, replies, reposts, likes). It contains NO sentiment or reaction columns: derive keyword flags, counts or your own text features, and join returns yourself.
* `data/community/global_hackernews_finance_posts.parquet` and `global_stackexchange_finance_questions.parquet` (raw discussion text for topic-attention proxies).
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

## 2. Lead-Lag Regressions on Daily Post Counts

Aggregate the raw posts into daily features strictly dated by the post timestamp, then regress returns of the **next** session on them (the post can only affect sessions after it is published):

```python
import polars as pl
import statsmodels.api as sm

def estimate_post_sensitivity(
    asset_file: str = "data/market/us_indices_daily.parquet",
    asset_symbol: str = "^GSPC",
    posts_file: str = "data/community/global_vip_social_posts_raw.parquet",
    keyword: str = r"\btariffs?\b",
):
    asset = pl.read_parquet(asset_file).filter(pl.col("symbol") == asset_symbol).sort("date")
    posts = pl.read_parquet(posts_file).with_columns(
        pl.col("text").str.to_lowercase().str.contains(keyword).cast(pl.Int32).alias("kw"),
        (pl.col("date") + pl.duration(days=1)).alias("next_day"),
    )
    # map every post to the first trading session strictly after its publication date
    mapped = posts.sort("next_day").join_asof(
        asset.select(pl.col("date").alias("session")), left_on="next_day", right_on="session", strategy="forward"
    ).drop_nulls("session")
    daily = mapped.group_by("session").agg(pl.len().alias("post_count"), pl.col("kw").sum().alias("keyword_posts"))
    merged = asset.join(daily, left_on="date", right_on="session", how="left").fill_null(0).drop_nulls()
    X = sm.add_constant(merged.select(["post_count", "keyword_posts"]).to_numpy())
    ols = sm.OLS(merged["returns"].to_numpy(), X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return {"beta_posts": float(ols.params[1]), "p_posts": float(ols.pvalues[1]),
            "beta_keyword": float(ols.params[2]), "p_keyword": float(ols.pvalues[2]), "r2": float(ols.rsquared)}
```

A complete worked example (placebo-tested) is `researches/res_002_vip_tariff_posts_event_study/code/run_pipeline.py`.

---

## 3. Placebo Bootstrap Controls (Non-Negotiable)
Because social media posts occur amidst other confounding market news:
1. **Shuffle Test**: Re-sample 1,000 pseudo-event dates from non-announcement days.
2. **Empirical p-value**:
   $$p_{\text{placebo}} = \frac{1}{B} \sum_{b=1}^B \mathbb{I}\left( |\text{CAR}_b^{\text{pseudo}}| \ge |\text{CAR}^{\text{actual}}| \right)$$
3. **Invariance**: An event effect is only classified as **ACCEPTED** if $p_{\text{placebo}} < 0.05$.

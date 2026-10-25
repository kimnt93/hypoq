---
name: world-data-taxonomy-modeling
description: Multi-domain quantitative financial and macroeconomic modeling across institutional data taxonomies: integrating real economic activity (FRED), GICS sector rotations, Fama-French style factors, corporate bond credit spreads, commodity term volatility surfaces, and cross-sectional valuation multiples.
version: 1.0.0
author: hypoq
tags: [macro, gics-sectors, factors, credit-spreads, fundamentals, valuation, cross-asset, fred, volatility-surface]
---

# World Financial Data Taxonomy & Multi-Domain Modeling Skill

## Overview
This skill operationalizes an institutional-grade, multi-domain quantitative modeling framework mirroring and extending professional data platforms (such as WiFeed, Bloomberg, and FactSet). It bridges **Macroeconomic Activity**, **11 GICS Sectors**, **Smart Beta Factors**, **Credit Spreads & Maritime Transport**, **Implied Volatility Surfaces**, and **Corporate Accounting Fundamentals**.

Empirical datasets leveraged:
* `data/macro/global_macro_indicators_monthly.parquet` (FRED PCE Inflation, Unemployment, NFCI, Breakeven Inflation, 10Y-2Y Spread, M2, Debt/GDP, High Yield OAS)
* `data/market/global_sectors_daily.parquet` (11 GICS Sectors: XLK, XLF, XLV, XLE, XLY, XLP, XLI, XLB, XLU, XLRE, XLC)
* `data/market/global_factors_daily.parquet` (MTUM, QUAL, USMV, VLUE, VUG, IJR, IJH, NOBL)
* `data/macro/global_credit_spreads_daily.parquet` (LQD, HYG, EMB, TIP, SHY, TLT, MUB, BDRY)
* `data/macro/global_cross_asset_volatility_daily.parquet` (VIX, VVIX, OVX Oil Vol, GVZ Gold Vol)
* `data/financials/global_corporate_fundamentals.parquet` (50+ global mega-caps: P/E, P/B, EV/EBITDA, ROE, margins, debt-to-equity)

---

## 1. Macroeconomic Regime Quadrants & GICS Sector Rotation

Constructs a 4-regime macro framework based on growth ($\Delta \text{Unemployment}_t$) and inflation ($\Delta \text{PCE}_t$):
$$\text{Regime}_t = \begin{cases}
\text{Reflation (Growth} \uparrow, \text{Inflation} \uparrow\text{)}: & \text{Overweight Energy (XLE), Materials (XLB), Financials (XLF)} \\
\text{Expansion (Growth} \uparrow, \text{Inflation} \downarrow\text{)}: & \text{Overweight Tech (XLK), Consumer Discretionary (XLY)} \\
\text{Stagflation (Growth} \downarrow, \text{Inflation} \uparrow\text{)}: & \text{Overweight Commodities, Energy (XLE), Cash / TIPS (TIP)} \\
\text{Contraction (Growth} \downarrow, \text{Inflation} \downarrow\text{)}: & \text{Overweight Utilities (XLU), Consumer Staples (XLP), Treasuries (TLT)}
\end{cases}$$

```python
import polars as pl
import numpy as np

def classify_macro_regimes(parquet_path: str = "data/macro/global_macro_indicators_monthly.parquet") -> pl.DataFrame:
    df = pl.read_parquet(parquet_path).sort("date")
    
    return (
        df.with_columns([
            pl.col("pce_headline_inflation_index").pct_change(12).alias("inflation_yoy"),
            pl.col("unemployment_rate_pct").diff(12).alias("d_unrate_12m"),
            pl.col("term_spread_10y_2y").alias("yield_curve_slope")
        ])
        .with_columns([
            pl.when((pl.col("d_unrate_12m") < 0) & (pl.col("inflation_yoy") > 0.025))
            .then(pl.lit("REFLATION"))
            .when((pl.col("d_unrate_12m") < 0) & (pl.col("inflation_yoy") <= 0.025))
            .then(pl.lit("GOLDILOCKS_EXPANSION"))
            .when((pl.col("d_unrate_12m") >= 0) & (pl.col("inflation_yoy") > 0.025))
            .then(pl.lit("STAGFLATION"))
            .otherwise(pl.lit("DEFLATIONARY_CONTRACTION"))
            .alias("macro_quadrant")
        ])
    )
```

---

## 2. Credit Spread & Financial Conditions Transmission

High Yield Option-Adjusted Spread ($\text{HY OAS}$) and Chicago Fed Financial Conditions Index ($\text{NFCI}$) serve as forward indicators of market liquidity and default risk.

```python
import polars as pl
import statsmodels.api as sm

def test_credit_conditions_transmission(
    macro_file: str = "data/macro/global_macro_indicators_monthly.parquet",
    sectors_file: str = "data/market/global_sectors_daily.parquet"
):
    df_macro = pl.read_parquet(macro_file).select(["date", "high_yield_option_adjusted_spread_bps", "chicago_fed_financial_conditions_index"]).drop_nulls()
    df_sec = pl.read_parquet(sectors_file).filter(pl.col("symbol") == "XLF").select(["date", "returns"]).rename({"returns": "financials_ret"})
    
    aligned = df_sec.join_asof(df_macro, on="date", strategy="backward").drop_nulls()
    
    y = aligned["financials_ret"].to_numpy()
    X = sm.add_constant(aligned.select(["high_yield_option_adjusted_spread_bps", "chicago_fed_financial_conditions_index"]).to_numpy())
    
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return {
        "beta_hy_spread": float(model.params[1]),
        "pvalue_hy_spread": float(model.pvalues[1]),
        "beta_nfci": float(model.params[2]),
        "pvalue_nfci": float(model.pvalues[2]),
        "r2": float(model.rsquared)
    }
```

---

## 3. Cross-Asset Volatility Spillovers & Supply Chain Stress

Models the interaction between equity risk ($\text{VIX}$), commodity shocks ($\text{OVX}, \text{GVZ}$), and global shipping freight rates ($\text{BDRY}$):

```python
import polars as pl
import numpy as np

def analyze_cross_asset_stress(
    vol_file: str = "data/macro/global_cross_asset_volatility_daily.parquet",
    freight_file: str = "data/macro/global_credit_spreads_daily.parquet"
):
    df_vol = pl.read_parquet(vol_file)
    df_bdry = pl.read_parquet(freight_file).filter(pl.col("symbol") == "BDRY").select(["date", "returns", "close"]).rename({"returns": "freight_returns", "close": "freight_price"})
    
    merged = df_vol.join(df_bdry, on="date", how="inner").drop_nulls()
    
    # Correlation matrix of cross-asset stress
    stress_cols = ["vix_equity", "vvix_vol_of_vol", "ovx_crude_oil", "gvz_gold", "freight_returns"]
    corr_df = merged.select(stress_cols).to_pandas().corr()
    
    return {
        "observations": len(merged),
        "correlation_matrix": corr_df.to_dict()
    }
```

---

## 4. Fundamental Quality & Valuation Alpha Screen

Combines institutional valuation multiples with profitability hurdles:
$$\text{Composite Quality Score}_i = \mathcal{Z}(\text{ROE}_i) + \mathcal{Z}(\text{Operating Margin}_i) - \mathcal{Z}(\text{Debt-to-Equity}_i) - \mathcal{Z}(\text{P/E}_i)$$

```python
import polars as pl

def rank_global_fundamentals(parquet_path: str = "data/financials/global_corporate_fundamentals.parquet") -> pl.DataFrame:
    df = pl.read_parquet(parquet_path).filter(pl.col("pe_trailing").is_not_null() & (pl.col("pe_trailing") > 0))
    
    return (
        df.with_columns([
            ((pl.col("roe") - pl.col("roe").mean()) / pl.col("roe").std()).alias("z_roe"),
            ((pl.col("operating_margin") - pl.col("operating_margin").mean()) / pl.col("operating_margin").std()).alias("z_op_margin"),
            ((pl.col("pe_trailing") - pl.col("pe_trailing").mean()) / pl.col("pe_trailing").std()).alias("z_pe"),
            ((pl.col("debt_to_equity").fill_null(0.0) - pl.col("debt_to_equity").fill_null(0.0).mean()) / pl.col("debt_to_equity").fill_null(0.0).std()).alias("z_debt"),
        ])
        .with_columns(
            (pl.col("z_roe") + pl.col("z_op_margin") - 0.5 * pl.col("z_pe") - 0.5 * pl.col("z_debt")).alias("composite_quality_score")
        )
        .sort("composite_quality_score", descending=True)
    )
```

---

## 5. Non-Negotiables
1. **Low-Frequency Alignment**: When merging monthly FRED indicators with daily market series, always use zero-copy backward as-of joins (`join_asof(..., strategy="backward")`). Never look forward.
2. **Missing Multiples Handling**: Avoid arbitrary zero-imputation on P/E or debt multiples. For firms with negative earnings, assign negative trailing P/E to lowest quintile or filter by positive EBITDA.
3. **Transaction Costs**: Sector rotation strategies must deduct minimum 10 bps two-way friction on liquid SPDR ETFs.

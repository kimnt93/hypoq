# Research 006: Monetary Transmission, Cross-Border Spillovers, and NAV Parity in Vietnamese Exchange-Traded Funds

## Overview
This study provides the first comprehensive econometric investigation into the tripartite relationship connecting US Federal Reserve FOMC announcements, State Bank of Vietnam (SBV) monetary interventions, and Vietnamese Exchange-Traded Funds (ETFs: E1VFVN30, FUEVFVND, and offshore VNM).

## Key Empirical Findings
- **Offshore Lead-Lag Price Discovery**: VanEck Vietnam ETF ($VNM_t$ on NYSE Arca) significantly leads domestic ETF returns ($E1VFVN30_{t+1}$ on HOSE) with $\beta = 0.2258$ ($t = 6.92$, $p < 10^{-11}$).
- **FOMC Announcement Decoupling**: On FOMC decision days, this lead-lag transmission experiences statistically significant decoupling ($\beta_{\text{interaction}} = -0.2390$, $t = -2.58$, $p = 0.0099$) as domestic market participants await the SBV central exchange rate fixing and liquidity operations before repricing equities.
- **Foreign Ownership Limit (FOL) Premium Expansion**: USD/VND currency depreciation shocks significantly widen the Diamond ETF ($FUEVFVND$) premium over NAV ($\beta = +0.4629$, $t = 2.67$, $p = 0.0076$), confirming foreign demand for FOL-exempt equities as currency hedging vehicles.
- **NAV Arbitrage Efficiency**: Domestic ETF NAV parity errors mean-revert rapidly with an implied half-life of $0.21$ trading sessions ($t = -49.62$).
- **Final Research Decision**: **ACCEPTED**.

## Directory Structure
```
researches/res_006_etf_monetary_transmission_sbv_fomc/
├── code/
│   └── run_pipeline.py          # Master econometric pipeline (Polars, Statsmodels, HAC)
├── figures/
│   ├── fig1_cross_border_transmission.png
│   ├── fig2_fomc_event_study.png
│   └── fig3_etf_premium_macro_dynamics.png
├── results/
│   ├── empirical_results.json   # Machine-readable test metrics
│   ├── table1_lead_lag_fomc.tex
│   ├── table2_fomc_event_car.tex
│   └── table3_etf_premium_parity.tex
├── references.bib               # BibTeX citations
├── main.tex                     # Formal research manuscript
├── main.pdf                     # Compiled publication PDF (Tectonic)
└── README.md
```

## Replication
```bash
# Run econometric pipeline
.venv/bin/python3 researches/res_006_etf_monetary_transmission_sbv_fomc/code/run_pipeline.py

# Compile manuscript with Tectonic
cd researches/res_006_etf_monetary_transmission_sbv_fomc && /home/kim/.local/bin/tectonic main.tex
```

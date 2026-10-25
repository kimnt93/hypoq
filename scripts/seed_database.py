"""
Seed hypoq SQLite Database with verified empirical hypotheses from paper-hub papers.
"""

import os
import sys

# Ensure root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hypoq.db import init_db, record_hypothesis, create_session, DEFAULT_DB_PATH


def seed():
    db_path = DEFAULT_DB_PATH
    init_db(db_path)

    sess_id = create_session(
        session_id="sess_baseline_paperhub_archive",
        agents=["codex", "claude", "hermes"],
        topic="Baseline Empirical Findings from Paper-Hub Studies (001 - 015)",
        notes="Historical seed records ingested from empirical_results.json and published manuscripts.",
        db_path=db_path,
    )
    print(f"Created seed session: {sess_id}")

    hypotheses_data = [
        # Paper 015 - H1
        {
            "title": "Capitalization Wedge and Futures Mispricing Levels",
            "statement": "Tests whether the market capitalization wedge between full-capitalization and tradable VN30 weights exerts structural downward pressure on carry-adjusted VN30F1M futures mispricing.",
            "null_hypothesis": "The capitalization wedge coefficient in carry-adjusted futures mispricing regressions is zero (beta_wedge = 0).",
            "alt_hypothesis": "The capitalization wedge drives systematic negative futures mispricing (beta_wedge < 0).",
            "domain": "derivatives_basis",
            "agent_name": "claude",
            "dataset_dependencies": [
                "data/market_structure/free_float.parquet",
                "data/market/vn30f1m_derivatives_daily.parquet",
                "data/macro/vietnam_monetary_daily.parquet"
            ],
            "methodology": "Point-in-time cost-of-carry decomposition, Newey-West HAC robust errors, levels and first differences over 2,282 sessions.",
            "metric_name": "HAC t-stat (levels)",
            "metric_value": -1.45,
            "p_value": 0.147,
            "confidence_level": 0.95,
            "status": "REJECTED",
            "summary_findings": "Wedge coefficient is statistically insignificant with HAC errors in levels (t = -1.45) and first differences (t = -1.02). Arbitrageurs absorb wedge-induced inventory without persistent pricing dislocation.",
            "report_dir": "015_tightly_held_shares_basis_dynamics",
            "pdf_path": "015_tightly_held_shares_basis_dynamics/main.pdf"
        },
        # Paper 015 - H2
        {
            "title": "Index Review Abnormal Mispricing Discount",
            "statement": "Tests whether scheduled semi-annual and quarterly index review sessions create temporary price pressure and abnormal futures discount.",
            "null_hypothesis": "Cumulative abnormal mispricing during scheduled index review windows is zero (CAR = 0).",
            "alt_hypothesis": "Index review windows exhibit abnormal negative futures mispricing due to liquidity dislocations (CAR < 0).",
            "domain": "free_float_wedges",
            "agent_name": "codex",
            "dataset_dependencies": [
                "data/market_structure/basket_history.parquet",
                "data/market/vn30f1m_derivatives_daily.parquet"
            ],
            "methodology": "Event study on 20 review dates with 63 calendar placebo windows and 10,000 bootstrap draws; Brown-Forsythe variance test.",
            "metric_name": "Placebo p-value",
            "metric_value": 0.284,
            "p_value": 0.284,
            "confidence_level": 0.95,
            "status": "REJECTED",
            "summary_findings": "Event study displays no significant abnormal mispricing discount compared to placebo months. However, review sessions exhibit significantly elevated mispricing volatility (0.54% vs 0.45%, p = 0.03).",
            "report_dir": "015_tightly_held_shares_basis_dynamics",
            "pdf_path": "015_tightly_held_shares_basis_dynamics/main.pdf"
        },
        # Paper 015 - H3
        {
            "title": "Capitalization Wedge and Benchmark Tracking Divergence",
            "statement": "Tests whether the capitalization wedge between tightly held and float-adjusted equities explains tracking divergence between headline VNINDEX and tradable VN30.",
            "null_hypothesis": "The wedge-implied active return has zero explanatory power for daily VNINDEX - VN30 return spread (beta = 0, R2 = 0).",
            "alt_hypothesis": "The wedge-implied active return positively explains tracking divergence (beta > 0, R2 > 0.40).",
            "domain": "free_float_wedges",
            "agent_name": "claude",
            "dataset_dependencies": [
                "data/market/vn_indices_daily.parquet",
                "data/market_structure/free_float.parquet"
            ],
            "methodology": "Beginning-of-day replicated weights, active return decomposition, HAC robust regressions over 2,282 trading sessions.",
            "metric_name": "HAC t-stat",
            "metric_value": 35.5,
            "p_value": 0.00001,
            "confidence_level": 0.99,
            "status": "ACCEPTED",
            "summary_findings": "The wedge-implied active return explains 51.6% of daily VNINDEX-VN30 tracking divergence (beta = 0.714, HAC t = 35.5). The tight-minus-wide spread matters through its interaction with the wedge (t = 3.98).",
            "report_dir": "015_tightly_held_shares_basis_dynamics",
            "pdf_path": "015_tightly_held_shares_basis_dynamics/main.pdf"
        },
        # Paper 015 - H4
        {
            "title": "Cross-Index Hedging Residual Volatility Wedge",
            "statement": "Tests whether hedging headline VNINDEX with VN30F1M futures leaves significantly higher residual risk than hedging the underlying VN30 basket.",
            "null_hypothesis": "Hedging VNINDEX with VN30F1M achieves identical variance reduction to hedging VN30 (Residual Vol Ratio = 1.0).",
            "alt_hypothesis": "Hedging VNINDEX leaves higher residual unhedged volatility due to constituent tracking dispersion (Residual Vol Ratio > 1.15).",
            "domain": "derivatives_basis",
            "agent_name": "hermes",
            "dataset_dependencies": [
                "data/market/vn_indices_daily.parquet",
                "data/market/vn30f1m_derivatives_daily.parquet"
            ],
            "methodology": "Minimum-variance hedge ratio estimation, rolling out-of-sample 250-session windows, daily and 20-session horizons.",
            "metric_name": "Residual Vol Ratio (20-day)",
            "metric_value": 1.40,
            "p_value": 0.001,
            "confidence_level": 0.95,
            "status": "ACCEPTED",
            "summary_findings": "Out-of-sample hedging effectiveness is 0.883 for VN30 vs 0.818 for VNINDEX. Hedged VNINDEX leaves 18% (daily) to 40% (20-session horizon) more residual volatility than hedged VN30.",
            "report_dir": "015_tightly_held_shares_basis_dynamics",
            "pdf_path": "015_tightly_held_shares_basis_dynamics/main.pdf"
        },
        # Paper 011
        {
            "title": "Order Flow Imbalance and Next-Session Return Predictability",
            "statement": "Tests whether high-frequency institutional order book supply/demand imbalance predicts next-session return drift in frontier index constituents.",
            "null_hypothesis": "Order flow imbalance exhibits zero forward return predictability across constituent equities (beta = 0).",
            "alt_hypothesis": "Order flow imbalance generates positive 1-session return continuation (beta > 0, t > 2.5).",
            "domain": "microstructure",
            "agent_name": "codex",
            "dataset_dependencies": [
                "data/market/vn_order_supply_demand_daily.parquet",
                "data/market/vn_top_equities_daily.parquet"
            ],
            "methodology": "Kyle's lambda estimation, panel fixed effects with clustered standard errors across 55 constituents.",
            "metric_name": "Panel t-stat",
            "metric_value": 4.12,
            "p_value": 0.0001,
            "confidence_level": 0.99,
            "status": "ACCEPTED",
            "summary_findings": "Institutional buy order flow imbalance produces statistically significant 1-session return drift of +18.4 bps (t = 4.12), decaying by session 3.",
            "report_dir": "011_order_flow_imbalance_price_impact",
            "pdf_path": "011_order_flow_imbalance_price_impact/main.pdf"
        },
        # Paper 014
        {
            "title": "Cross-Sectional Momentum Crash Prevention via Volatility Scaling",
            "statement": "Tests whether dynamic inverse volatility scaling eliminates catastrophic momentum crashes in emerging equity markets.",
            "null_hypothesis": "Volatility-scaled momentum generates identical drawdown and tail risk to unhedged momentum.",
            "alt_hypothesis": "Volatility-scaled momentum significantly dampens maximum drawdown while enhancing the Sharpe ratio.",
            "domain": "fundamental_factors",
            "agent_name": "pi",
            "dataset_dependencies": [
                "data/market/vn_top_equities_daily.parquet"
            ],
            "methodology": "Daniel-Moskowitz volatility-scaled cross-sectional momentum, Barroso-Santa-Clara risk management over 2015-2026.",
            "metric_name": "Sharpe Ratio Improvement",
            "metric_value": 0.65,
            "p_value": 0.002,
            "confidence_level": 0.95,
            "status": "ACCEPTED",
            "summary_findings": "Inverse volatility scaling reduces maximum momentum drawdown from -48.2% to -18.7%, boosting annualized Sharpe ratio from 0.72 to 1.37.",
            "report_dir": "014_cross_sectional_momentum_alpha_execution",
            "pdf_path": "014_cross_sectional_momentum_alpha_execution/main.pdf"
        }
    ]

    for item in hypotheses_data:
        hid = record_hypothesis(
            title=item["title"],
            statement=item["statement"],
            null_hypothesis=item["null_hypothesis"],
            alt_hypothesis=item["alt_hypothesis"],
            domain=item["domain"],
            session_id=sess_id,
            agent_name=item["agent_name"],
            dataset_dependencies=item["dataset_dependencies"],
            methodology=item["methodology"],
            metric_name=item["metric_name"],
            metric_value=item["metric_value"],
            p_value=item["p_value"],
            confidence_level=item["confidence_level"],
            status=item["status"],
            summary_findings=item["summary_findings"],
            report_dir=item["report_dir"],
            pdf_path=item["pdf_path"],
            db_path=db_path,
        )
        print(f"Recorded hypothesis #{hid}: [{item['status']}] {item['title']}")

    print(f"\nSuccessfully seeded {len(hypotheses_data)} empirical hypotheses into {db_path}!")


if __name__ == "__main__":
    seed()

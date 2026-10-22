#!/usr/bin/env python3
"""
Publication-Quality Dataset Distribution & Empirical Overview Renderer.
Visualizes:
1. Major Shareholder Ownership Distribution (% State vs Institutional vs Individuals)
2. Corporate Structure: Subsidiaries & Affiliates Count Across Sectors
3. Foreign Investor Trading Dynamics (Cumulative Net Foreign Buying/Selling)
4. Order Book Supply-Demand Imbalance Distributions (Bid vs Offer Pressure)
5. Historical Charter Capital Growth Paths (Vốn điều lệ via Issuance & Stock Dividends)
Outputs vector PDF and 300 DPI PNG figures to figures_overview/.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import polars as pl
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MARKET_DIR = os.path.join(DATA_DIR, "market")
STRUCT_DIR = os.path.join(DATA_DIR, "market_structure")
OUT_DIR = os.path.join(BASE_DIR, "figures_overview")
os.makedirs(OUT_DIR, exist_ok=True)

# Styling configuration
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.bbox": "tight"
})

def render_figure1_shareholders_distribution():
    print(">>> [1/5] Rendering Figure 1: Shareholder Concentration & Ownership Structure...")
    sh_path = os.path.join(STRUCT_DIR, "vn_shareholders_history.parquet")
    if not os.path.exists(sh_path):
        print(f"    ! {sh_path} not found.")
        return

    df = pl.read_parquet(sh_path)
    
    # Filter top 15 key representative liquid stocks
    focus_syms = ["HVN", "VCB", "BID", "GAS", "CTG", "VHM", "VIC", "HPG", "FPT", "MSN", "MWG", "SSI", "VND", "POW", "PLX"]
    df_focus = df.filter(pl.col("symbol").is_in(focus_syms))
    
    # Aggregate ownership by type
    agg_df = (
        df_focus.group_by(["symbol", "shareholder_type"])
        .agg(pl.col("ownership_pct").sum().alias("total_pct"))
        .sort("symbol")
    ).to_pandas()

    pivoted = agg_df.pivot(index="symbol", columns="shareholder_type", values="total_pct").fillna(0.0)
    pivoted = pivoted.reindex(focus_syms)

    fig, ax = plt.subplots(figsize=(12, 6))
    symbols = pivoted.index.tolist()
    y_pos = np.arange(len(symbols))
    
    company_pct = pivoted.get("COMPANY", [0]*len(symbols))
    person_pct = pivoted.get("PERSON", [0]*len(symbols))
    
    ax.barh(y_pos, company_pct, color="#1f77b4", edgecolor="black", alpha=0.85, label="Institutional / State Entities (Tổ chức / Nhà nước)")
    ax.barh(y_pos, person_pct, left=company_pct, color="#ff7f0e", edgecolor="black", alpha=0.85, label="Individual Insiders / Executives (Cá nhân / Lãnh đạo)")
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(symbols, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Cumulative Reported Major Ownership (%)", fontweight="bold")
    ax.set_title("Cross-Sectional Shareholder Concentration & Ownership Breakdown Across Key Equities", fontweight="bold", loc="left")
    ax.set_xlim(0, 105)
    ax.axvline(50, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Majority Control (50%)")
    ax.legend(loc="lower right", framealpha=0.95)
    
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig1_shareholders_concentration.png"), dpi=300)
    plt.close(fig)
    print("    - Saved fig1_shareholders_concentration.png (PNG only)")


def render_figure2_subsidiaries_distribution():
    print(">>> [2/5] Rendering Figure 2: Subsidiaries & Affiliates Corporate Network...")
    rel_path = os.path.join(STRUCT_DIR, "vn_subsidiaries_affiliates.parquet")
    if not os.path.exists(rel_path):
        print(f"    ! {rel_path} not found.")
        return

    df = pl.read_parquet(rel_path)
    
    # Filter for Subsidiaries (Công ty con) and Associates (Công ty liên kết)
    df_sub = df.filter(pl.col("relation_type").is_in(["SUBSIDIARY", "ASSOCIATE"]))
    
    # Count by symbol
    counts = (
        df_sub.group_by(["symbol", "relation_type"])
        .len()
        .sort("len", descending=True)
    ).to_pandas()

    # Get top 15 conglomerates / companies by total network size
    top_syms = (
        counts.groupby("symbol")["len"].sum().sort_values(ascending=False).head(15).index.tolist()
    )
    piv = counts[counts["symbol"].isin(top_syms)].pivot(index="symbol", columns="relation_type", values="len").fillna(0)
    piv = piv.reindex(top_syms)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(top_syms))
    width = 0.4
    
    sub_vals = piv.get("SUBSIDIARY", [0]*len(top_syms))
    assoc_vals = piv.get("ASSOCIATE", [0]*len(top_syms))
    
    ax.bar(x - width/2, sub_vals, width, label="Subsidiaries (Công ty con, >50% ownership)", color="#2ca02c", edgecolor="black", alpha=0.85)
    ax.bar(x + width/2, assoc_vals, width, label="Associates & Affiliates (Công ty liên kết, 20-50%)", color="#d62728", edgecolor="black", alpha=0.85)
    
    ax.set_xticks(x)
    ax.set_xticklabels(top_syms, fontweight="bold", rotation=30)
    ax.set_ylabel("Number of Corporate Entities", fontweight="bold")
    ax.set_title("Corporate Structure Network: Subsidiaries vs. Affiliates Across Leading Listed Groups", fontweight="bold", loc="left")
    ax.legend(loc="upper right", framealpha=0.95)
    
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig2_corporate_structure_network.png"), dpi=300)
    plt.close(fig)
    print("    - Saved fig2_corporate_structure_network.png (PNG only)")


def render_figure3_foreign_trading():
    print(">>> [3/5] Rendering Figure 3: Cumulative Foreign Investor Flows (GD NĐTNN)...")
    fo_path = os.path.join(MARKET_DIR, "vn_foreign_trading_daily.parquet")
    if not os.path.exists(fo_path):
        print(f"    ! {fo_path} not found.")
        return

    df = pl.read_parquet(fo_path)
    
    # Focus on representative blue-chips across sectors
    syms = ["HPG", "FPT", "VCB", "VHM", "MSN", "HVN", "SSI", "MWG"]
    df_filtered = df.filter(pl.col("symbol").is_in(syms)).sort(["symbol", "date"]).to_pandas()
    
    fig, ax = plt.subplots(figsize=(13, 6))
    colors = sns.color_palette("tab10", len(syms))
    
    for i, s in enumerate(syms):
        sub = df_filtered[df_filtered["symbol"] == s].copy()
        if len(sub) > 0:
            sub["date"] = pd.to_datetime(sub["date"])
            sub["cum_net_val_bn"] = sub["net_val"].cumsum() / 1e9  # Billion VND
            ax.plot(sub["date"], sub["cum_net_val_bn"], label=f"{s}", color=colors[i], linewidth=1.8)

    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_ylabel("Cumulative Net Foreign Value (Billion VND)", fontweight="bold")
    ax.set_xlabel("Trading Date", fontweight="bold")
    ax.set_title("Cumulative Foreign Net Investment Flows (GD NĐTNN Mua/Bán Ròng Lũy Kế 2018–2026)", fontweight="bold", loc="left")
    ax.legend(loc="upper left", ncol=2, framealpha=0.95)
    
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig3_cumulative_foreign_flows.png"), dpi=300)
    plt.close(fig)
    print("    - Saved fig3_cumulative_foreign_flows.png (PNG only)")


def render_figure4_supply_demand_imbalance():
    print(">>> [4/5] Rendering Figure 4: Order Supply & Demand Imbalance Distributions (Cung cầu)...")
    sd_path = os.path.join(MARKET_DIR, "vn_order_supply_demand_daily.parquet")
    if not os.path.exists(sd_path):
        print(f"    ! {sd_path} not found.")
        return

    df = pl.read_parquet(sd_path)
    
    syms = ["VCB", "HPG", "FPT", "VHM", "HVN", "SSI"]
    df_sub = df.filter(pl.col("symbol").is_in(syms)).to_pandas()

    fig, ax = plt.subplots(figsize=(12, 6))
    
    for s in syms:
        subset = df_sub[df_sub["symbol"] == s]
        if len(subset) > 0:
            sns.kdeplot(subset["order_imbalance"], label=f"{s} (Mean: {subset['order_imbalance'].mean():.2f})", ax=ax, linewidth=2.0)

    ax.axvline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.8, label="Neutral Imbalance (0.0)")
    ax.set_xlabel("Daily Order Imbalance: (Bid Vol - Offer Vol) / (Bid Vol + Offer Vol)", fontweight="bold")
    ax.set_ylabel("Kernel Density", fontweight="bold")
    ax.set_title("Distribution of Daily Order Book Imbalance (Cung cầu áp lực Mua/Bán) 2021–2026", fontweight="bold", loc="left")
    ax.set_xlim(-1.0, 1.0)
    ax.legend(loc="upper left", framealpha=0.95)
    
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig4_order_imbalance_distribution.png"), dpi=300)
    plt.close(fig)
    print("    - Saved fig4_order_imbalance_distribution.png (PNG only)")


def render_figure5_charter_capital_growth():
    print(">>> [5/5] Rendering Figure 5: Charter Capital Growth & Share Issuance Events...")
    cap_path = os.path.join(STRUCT_DIR, "vn_charter_capital_history.parquet")
    if not os.path.exists(cap_path):
        print(f"    ! {cap_path} not found.")
        return

    df = pl.read_parquet(cap_path)
    
    # Event breakdown
    ev_counts = (
        df.filter(pl.col("event_type").is_not_null())
        .group_by("event_type")
        .len()
        .sort("len", descending=True)
    ).to_pandas()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Left: Event Types Distribution
    colors = sns.color_palette("muted", len(ev_counts))
    ax1.pie(ev_counts["len"], labels=ev_counts["event_type"], autopct="%1.1f%%", startangle=140, colors=colors,
            wedgeprops={"edgecolor": "black", "linewidth": 0.8})
    ax1.set_title("(a) Distribution of Capital Adjustment Events", fontweight="bold")

    # Right: Focus on HVN and Banking Giants Share Expansion
    symbols = ["HVN", "VCB", "BID", "CTG", "MBB", "TCB"]
    shares_issued = (
        df.filter(pl.col("symbol").is_in(symbols) & (pl.col("number_of_shares") > 0))
        .group_by("symbol")
        .agg(pl.col("number_of_shares").sum().alias("total_new_shares"))
        .sort("total_new_shares", descending=True)
    ).to_pandas()

    ax2.bar(shares_issued["symbol"], shares_issued["total_new_shares"] / 1e6, color="#4c72b0", edgecolor="black", alpha=0.85)
    ax2.set_ylabel("Cumulative New Shares Issued (Million Shares)", fontweight="bold")
    ax2.set_xlabel("Stock Symbol", fontweight="bold")
    ax2.set_title("(b) Cumulative Capital Expansion Shares Across Selected Giants", fontweight="bold")
    
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig5_charter_capital_events.png"), dpi=300)
    plt.close(fig)
    print("    - Saved fig5_charter_capital_events.png (PNG only)")


if __name__ == "__main__":
    import pandas as pd
    render_figure1_shareholders_distribution()
    render_figure2_subsidiaries_distribution()
    render_figure3_foreign_trading()
    render_figure4_supply_demand_imbalance()
    render_figure5_charter_capital_growth()
    try:
        try:
            from build_corporate_network import build_network
        except ImportError:
            from scripts.build_corporate_network import build_network
        build_network()
    except Exception as e:
        print(f"    ! Error building corporate network: {e}")
    print(">>> All 6 Overview Figures rendered into figures_overview/ (PNG ONLY, no PDF)!")

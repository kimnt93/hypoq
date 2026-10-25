#!/usr/bin/env python3
"""
Split and normalize widata_macro_all_series into clean, domain-grouped wide Parquet tables.
Replaces the unpivoted long format with explicit English-named column datasets:
1. vn_trade_exports_monthly.parquet
2. vn_fdi_sectors_monthly.parquet
3. vn_sbv_policy_rates_daily.parquet

"""

import os
import polars as pl

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MACRO_DIR = os.path.join(BASE_DIR, "data/macro")

def split_macro():
    raw_path = os.path.join(MACRO_DIR, "widata_macro_all_series.parquet")
    if not os.path.exists(raw_path):
        print(f"Error: {raw_path} not found.")
        return

    print(">>> Loading widata_macro_all_series.parquet...")
    df = pl.read_parquet(raw_path)

    # 1. Trade & Commodity Exports (Table 278)
    print(">>> [1/4] Pivoting Trade & Commodity Exports...")
    sub_trade = df.filter(pl.col("table_name") == "278").sort(["date", "column_name"])
    piv_trade = sub_trade.pivot(on="column_name", index="date", values="value").sort("date")
    
    trade_rename = {
        "tong": "total_export_value_usd",
        "tong_yoy": "total_export_growth_yoy",
        "gt_sat_thep_mtq": "steel_export_value_usd",
        "gt_sat_thep_yym": "steel_export_growth_yoy",
        "gt_sp_tu_thep_mtq": "steel_products_export_value_usd",
        "gt_sp_tu_thep_yym": "steel_products_export_growth_yoy",
        "gt_ca_phe": "coffee_export_value_usd"
    }
    for old_col, new_col in trade_rename.items():
        if old_col in piv_trade.columns:
            piv_trade = piv_trade.rename({old_col: new_col})

    trade_path = os.path.join(MACRO_DIR, "vn_trade_exports_monthly.parquet")
    piv_trade.write_parquet(trade_path)
    print(f"    - Saved vn_trade_exports_monthly.parquet ({len(piv_trade):,} rows x {len(piv_trade.columns)} cols)")

    # 2. Foreign Direct Investment (FDI) Sectors (Table 236)
    print(">>> [2/4] Pivoting FDI Sector Disbursed Capital...")
    sub_fdi = df.filter(pl.col("table_name") == "236").sort(["date", "column_name"])
    piv_fdi = sub_fdi.pivot(on="column_name", index="date", values="value").sort("date")
    
    fdi_rename = {
        "che_bien_che_tao_ttm": "fdi_manufacturing_ttm_usd",
        "bat_dong_san_ttm": "fdi_real_estate_ttm_usd",
        "dien_khi_nuoc_ttm": "fdi_utilities_energy_ttm_usd",
        "khoa_hoc_cong_nghe_ttm": "fdi_science_technology_ttm_usd"
    }
    for old_col, new_col in fdi_rename.items():
        if old_col in piv_fdi.columns:
            piv_fdi = piv_fdi.rename({old_col: new_col})

    fdi_path = os.path.join(MACRO_DIR, "vn_fdi_sectors_monthly.parquet")
    piv_fdi.write_parquet(fdi_path)
    print(f"    - Saved vn_fdi_sectors_monthly.parquet ({len(piv_fdi):,} rows x {len(piv_fdi.columns)} cols)")

    # 3. State Bank of Vietnam Policy Rates & Reserve Requirements (Table 250)
    print(">>> [3/4] Pivoting SBV Policy Rates & Required Reserves...")
    sub_sbv = df.filter(pl.col("table_name") == "250").sort(["date", "column_name"])
    piv_sbv = sub_sbv.pivot(on="column_name", index="date", values="value").sort("date")
    
    sbv_rename = {
        "lai_suat_dieu_hanh_tai_cap_von": "refinancing_rate",
        "lai_suat_dieu_hanh_chiet_khau": "discount_rate",
        "ls_cho_vay_bu_dap_thieu_hut_von_nhnn": "capital_shortfall_lending_rate",
        "ls_du_tru_bat_buoc_vnd": "required_reserve_rate_vnd",
        "ls_du_tru_bat_buoc_ngoai_te": "required_reserve_rate_fx"
    }
    for old_col, new_col in sbv_rename.items():
        if old_col in piv_sbv.columns:
            piv_sbv = piv_sbv.rename({old_col: new_col})

    sbv_path = os.path.join(MACRO_DIR, "vn_sbv_policy_rates_daily.parquet")
    piv_sbv.write_parquet(sbv_path)
    print(f"    - Saved vn_sbv_policy_rates_daily.parquet ({len(piv_sbv):,} rows x {len(piv_sbv.columns)} cols)")

    # (US Federal Reserve balance sheet comes straight from FRED: scripts/ingest_fred_us_macro.py)

    # 5. Remove raw unpivoted file
    print(">>> Removing raw monolithic widata_macro_all_series.parquet...")
    os.remove(raw_path)
    print("    - Cleaned up raw long table successfully!")

if __name__ == "__main__":
    split_macro()

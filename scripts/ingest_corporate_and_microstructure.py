#!/usr/bin/env python3
"""
Corporate Structure, Shareholders, Foreign Trading & Order Supply/Demand Ingestion.
Extracts Point-in-Time datasets from official market registry endpoints:
1. Major Shareholders & Ownership History (Cổ đông lớn & Sở hữu theo thời gian)
2. Subsidiaries & Affiliates (Công ty con & Công ty liên kết)
3. Charter Capital History (Lịch sử tăng vốn điều lệ & phát hành)
4. Foreign Trading Daily (Giao dịch Nhà đầu tư nước ngoài - GD NĐTNN)
5. Order Supply & Demand Daily (Cung cầu đặt lệnh mua/bán)
"""

import os
import time
import concurrent.futures
from datetime import datetime
import requests
import polars as pl

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MARKET_DIR = os.path.join(DATA_DIR, "market")
STRUCT_DIR = os.path.join(DATA_DIR, "market_structure")

for d in [MARKET_DIR, STRUCT_DIR]:
    os.makedirs(d, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# 65 Core liquid stocks covering VN30 + liquid equities across all 10 ICB sectors
DEFAULT_SYMBOLS = [
    # Banking (16)
    "VCB", "BID", "CTG", "TCB", "MBB", "VPB", "ACB", "HDB", "STB", "SHB", "LPB", "VIB", "TPB", "MSB", "SSB", "EIB",
    # Real Estate & Construction (15)
    "VHM", "VIC", "VRE", "NVL", "KDH", "DIG", "DXG", "PDR", "CEO", "KBC", "CII", "CTD", "HBC", "BCM", "TCH",
    # Financial Services / Securities (8)
    "SSI", "VND", "VCI", "HCM", "MBS", "SHS", "FTS", "BSI",
    # Steel & Basic Materials (6)
    "HPG", "HSG", "NKG", "DGC", "DPM", "BMP",
    # Technology & Telecommunications (1)
    "FPT",
    # Retail & Consumer Goods (8)
    "MWG", "MSN", "VNM", "SAB", "PNJ", "KDC", "SBT", "MCH",
    # Energy, Oil & Gas, Infrastructure & Utilities (8)
    "GAS", "PLX", "PVD", "PVS", "BSR", "POW", "REE", "NT2",
    # Aviation & Logistics (3)
    "HVN", "VJC", "GMD"
]

def fetch_shareholders(symbols=None):
    if symbols is None:
        symbols = DEFAULT_SYMBOLS
    print(f"\n>>> [1/5] Fetching Shareholders History for {len(symbols)} symbols...")
    rows = []

    def _fetch_one(sym):
        url = f"https://api-finfo.vndirect.com.vn/v4/shareholders?q=code:{sym}&size=1000"
        try:
            r = requests.get(url, headers=HEADERS, timeout=12)
            if r.status_code == 200:
                data = r.json().get("data", [])
                out = []
                for item in data:
                    out.append({
                        "symbol": sym,
                        "shareholder_type": item.get("shareholderType"),
                        "shareholder_name": item.get("shareholderName"),
                        "role_type": item.get("roleType"),
                        "en_role_type": item.get("enRoleType"),
                        "number_of_shares": float(item.get("numberOfShares", 0.0)) if item.get("numberOfShares") is not None else 0.0,
                        "ownership_pct": float(item.get("ownershipPct", 0.0)) if item.get("ownershipPct") is not None else 0.0,
                        "group_pct": float(item.get("groupPct", 0.0)) if item.get("groupPct") is not None else None,
                        "effective_date": item.get("effectiveDate")
                    })
                return out
        except Exception as e:
            print(f"    ! Error fetching shareholders for {sym}: {e}")
        return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_fetch_one, s): s for s in symbols}
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res:
                rows.extend(res)

    if rows:
        schema = {
            "symbol": pl.Utf8,
            "shareholder_type": pl.Utf8,
            "shareholder_name": pl.Utf8,
            "role_type": pl.Utf8,
            "en_role_type": pl.Utf8,
            "number_of_shares": pl.Float64,
            "ownership_pct": pl.Float64,
            "group_pct": pl.Float64,
            "effective_date": pl.Utf8
        }
        df = pl.DataFrame(rows, schema=schema).sort(["symbol", "effective_date", "ownership_pct"], descending=[False, True, True])
        out_path = os.path.join(STRUCT_DIR, "vn_shareholders_history.parquet")
        df.write_parquet(out_path)
        print(f"    - Saved vn_shareholders_history.parquet ({len(df):,} rows across {df['symbol'].n_unique()} symbols)")
        return df
    return pl.DataFrame()


def fetch_subsidiaries_and_affiliates(symbols=None):
    if symbols is None:
        symbols = DEFAULT_SYMBOLS
    print(f"\n>>> [2/5] Fetching Subsidiaries & Affiliates (Công ty con & liên kết) for {len(symbols)} symbols...")
    rows = []

    def _fetch_one(sym):
        url = f"https://api-finfo.vndirect.com.vn/v4/company_relations?q=code:{sym}&size=1000"
        try:
            r = requests.get(url, headers=HEADERS, timeout=12)
            if r.status_code == 200:
                data = r.json().get("data", [])
                out = []
                for item in data:
                    out.append({
                        "symbol": sym,
                        "relation_type": item.get("relationType"),
                        "relation_name_vn": item.get("relationNameVn"),
                        "relation_name_en": item.get("relationNameEn"),
                        "relation_id": int(item.get("relationId")) if item.get("relationId") is not None else None,
                        "effective_date": item.get("effectiveDate"),
                        "year": str(item.get("year", "")) if item.get("year") else None,
                        "ownership_pct": float(item.get("ownershipPct")) if item.get("ownershipPct") is not None else None,
                        "investment_capital": float(item.get("investmentCapital")) if item.get("investmentCapital") is not None else None,
                        "unit": item.get("unit")
                    })
                return out
        except Exception as e:
            print(f"    ! Error fetching relations for {sym}: {e}")
        return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_fetch_one, s): s for s in symbols}
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res:
                rows.extend(res)

    if rows:
        schema = {
            "symbol": pl.Utf8,
            "relation_type": pl.Utf8,
            "relation_name_vn": pl.Utf8,
            "relation_name_en": pl.Utf8,
            "relation_id": pl.Int64,
            "effective_date": pl.Utf8,
            "year": pl.Utf8,
            "ownership_pct": pl.Float64,
            "investment_capital": pl.Float64,
            "unit": pl.Utf8
        }
        df = pl.DataFrame(rows, schema=schema).sort(["symbol", "relation_type", "effective_date"], descending=[False, False, True])
        out_path = os.path.join(STRUCT_DIR, "vn_subsidiaries_affiliates.parquet")
        df.write_parquet(out_path)
        print(f"    - Saved vn_subsidiaries_affiliates.parquet ({len(df):,} rows across {df['symbol'].n_unique()} symbols)")
        return df
    return pl.DataFrame()


def fetch_charter_capital_history(symbols=None):
    if symbols is None:
        symbols = DEFAULT_SYMBOLS
    print(f"\n>>> [3/5] Fetching Charter Capital & Issuance History (Vốn điều lệ) for {len(symbols)} symbols...")
    rows = []

    def _fetch_one(sym):
        url = f"https://api-finfo.vndirect.com.vn/v4/events?q=code:{sym}~group:investorRight&size=1000"
        try:
            r = requests.get(url, headers=HEADERS, timeout=12)
            if r.status_code == 200:
                data = r.json().get("data", [])
                out = []
                for item in data:
                    out.append({
                        "symbol": sym,
                        "event_id": str(item.get("id", "")),
                        "event_group": item.get("group"),
                        "event_type": item.get("type"),
                        "type_desc": item.get("typeDesc"),
                        "note": item.get("note"),
                        "price": float(item.get("price")) if item.get("price") is not None else None,
                        "ratio": float(item.get("ratio")) if item.get("ratio") is not None else None,
                        "number_of_shares": float(item.get("numberOfShares")) if item.get("numberOfShares") is not None else None,
                        "disclosure_date": item.get("disclosureDate"),
                        "effective_date": item.get("effectiveDate"),
                        "trading_start_date": item.get("tradingStartDate"),
                        "actual_date": item.get("actualDate")
                    })
                return out
        except Exception as e:
            print(f"    ! Error fetching events for {sym}: {e}")
        return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_fetch_one, s): s for s in symbols}
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res:
                rows.extend(res)

    if rows:
        schema = {
            "symbol": pl.Utf8,
            "event_id": pl.Utf8,
            "event_group": pl.Utf8,
            "event_type": pl.Utf8,
            "type_desc": pl.Utf8,
            "note": pl.Utf8,
            "price": pl.Float64,
            "ratio": pl.Float64,
            "number_of_shares": pl.Float64,
            "disclosure_date": pl.Utf8,
            "effective_date": pl.Utf8,
            "trading_start_date": pl.Utf8,
            "actual_date": pl.Utf8
        }
        df = pl.DataFrame(rows, schema=schema).sort(["symbol", "effective_date"], descending=[False, True])
        out_path = os.path.join(STRUCT_DIR, "vn_charter_capital_history.parquet")
        df.write_parquet(out_path)
        print(f"    - Saved vn_charter_capital_history.parquet ({len(df):,} rows across {df['symbol'].n_unique()} symbols)")
        return df
    return pl.DataFrame()


def fetch_foreign_trading_daily(symbols=None):
    if symbols is None:
        symbols = DEFAULT_SYMBOLS
    print(f"\n>>> [4/5] Fetching Daily Foreign Trading (GD NĐTNN) for {len(symbols)} symbols...")
    rows = []

    def _fetch_one(sym):
        url = f"https://api-finfo.vndirect.com.vn/v4/foreigns?q=code:{sym}&size=3000"
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                data = r.json().get("data", [])
                out = []
                for item in data:
                    tdate = item.get("tradingDate")
                    if not tdate:
                        continue
                    tot_room = float(item.get("totalRoom", 0.0)) if item.get("totalRoom") is not None else 0.0
                    cur_room = float(item.get("currentRoom", 0.0)) if item.get("currentRoom") is not None else 0.0
                    room_pct = (cur_room / tot_room * 100.0) if tot_room > 0 else 0.0
                    out.append({
                        "symbol": sym,
                        "date": tdate,
                        "floor": item.get("floor"),
                        "buy_val": float(item.get("buyVal", 0.0)) if item.get("buyVal") is not None else 0.0,
                        "sell_val": float(item.get("sellVal", 0.0)) if item.get("sellVal") is not None else 0.0,
                        "net_val": float(item.get("netVal", 0.0)) if item.get("netVal") is not None else 0.0,
                        "buy_vol": float(item.get("buyVol", 0.0)) if item.get("buyVol") is not None else 0.0,
                        "sell_vol": float(item.get("sellVol", 0.0)) if item.get("sellVol") is not None else 0.0,
                        "net_vol": float(item.get("netVol", 0.0)) if item.get("netVol") is not None else 0.0,
                        "total_room": tot_room,
                        "current_room": cur_room,
                        "room_pct": room_pct
                    })
                return out
        except Exception as e:
            print(f"    ! Error fetching foreign trading for {sym}: {e}")
        return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_fetch_one, s): s for s in symbols}
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res:
                rows.extend(res)

    if rows:
        schema = {
            "symbol": pl.Utf8,
            "date": pl.Utf8,
            "floor": pl.Utf8,
            "buy_val": pl.Float64,
            "sell_val": pl.Float64,
            "net_val": pl.Float64,
            "buy_vol": pl.Float64,
            "sell_vol": pl.Float64,
            "net_vol": pl.Float64,
            "total_room": pl.Float64,
            "current_room": pl.Float64,
            "room_pct": pl.Float64
        }
        df = pl.DataFrame(rows, schema=schema).sort(["symbol", "date"])
        out_path = os.path.join(MARKET_DIR, "vn_foreign_trading_daily.parquet")
        df.write_parquet(out_path)
        print(f"    - Saved vn_foreign_trading_daily.parquet ({len(df):,} rows across {df['symbol'].n_unique()} symbols)")
        return df
    return pl.DataFrame()


def fetch_order_supply_demand_daily(symbols=None):
    if symbols is None:
        symbols = DEFAULT_SYMBOLS
    print(f"\n>>> [5/5] Fetching Daily Order Supply & Demand (Cung cầu) for {len(symbols)} symbols...")
    rows = []

    def _fetch_one(sym):
        url = f"https://api-finfo.vndirect.com.vn/v4/trading_statistics?q=code:{sym}&size=3000"
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                data = r.json().get("data", [])
                out = []
                for item in data:
                    tdate = item.get("tradingDate")
                    if not tdate:
                        continue
                    bvol = float(item.get("bidVol", 0.0)) if item.get("bidVol") is not None else 0.0
                    ovol = float(item.get("offerVol", 0.0)) if item.get("offerVol") is not None else 0.0
                    imbalance = ((bvol - ovol) / (bvol + ovol)) if (bvol + ovol) > 0 else 0.0
                    out.append({
                        "symbol": sym,
                        "date": tdate,
                        "floor": item.get("floor"),
                        "bid_order": float(item.get("bidOrder", 0.0)) if item.get("bidOrder") is not None else 0.0,
                        "bid_vol": bvol,
                        "offer_order": float(item.get("offerOrder", 0.0)) if item.get("offerOrder") is not None else 0.0,
                        "offer_vol": ovol,
                        "avg_bid_vol": float(item.get("avgBidVol", 0.0)) if item.get("avgBidVol") is not None else 0.0,
                        "avg_offer_vol": float(item.get("avgOfferVol", 0.0)) if item.get("avgOfferVol") is not None else 0.0,
                        "net_vol": float(item.get("netVol", 0.0)) if item.get("netVol") is not None else (bvol - ovol),
                        "order_imbalance": imbalance
                    })
                return out
        except Exception as e:
            print(f"    ! Error fetching supply/demand for {sym}: {e}")
        return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_fetch_one, s): s for s in symbols}
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res:
                rows.extend(res)

    if rows:
        schema = {
            "symbol": pl.Utf8,
            "date": pl.Utf8,
            "floor": pl.Utf8,
            "bid_order": pl.Float64,
            "bid_vol": pl.Float64,
            "offer_order": pl.Float64,
            "offer_vol": pl.Float64,
            "avg_bid_vol": pl.Float64,
            "avg_offer_vol": pl.Float64,
            "net_vol": pl.Float64,
            "order_imbalance": pl.Float64
        }
        df = pl.DataFrame(rows, schema=schema).sort(["symbol", "date"])
        out_path = os.path.join(MARKET_DIR, "vn_order_supply_demand_daily.parquet")
        df.write_parquet(out_path)
        print(f"    - Saved vn_order_supply_demand_daily.parquet ({len(df):,} rows across {df['symbol'].n_unique()} symbols)")
        return df
    return pl.DataFrame()


def run_all(symbols=None):
    t0 = time.time()
    print("=" * 80)
    print("STARTING CORPORATE & MICROSTRUCTURE INGESTION PIPELINE")
    print("=" * 80)
    df_sh = fetch_shareholders(symbols)
    df_rel = fetch_subsidiaries_and_affiliates(symbols)
    df_cap = fetch_charter_capital_history(symbols)
    df_fo = fetch_foreign_trading_daily(symbols)
    df_sd = fetch_order_supply_demand_daily(symbols)
    print("=" * 80)
    print(f"COMPLETED ALL 5 INGESTION STEPS IN {time.time() - t0:.2f}s!")
    print("=" * 80)


if __name__ == "__main__":
    run_all()

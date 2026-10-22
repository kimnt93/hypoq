#!/usr/bin/env python3
"""
Corporate & Financial Relationship Network Graph Builder.
Constructs directed network graph of Vietnamese listed companies, conglomerates,
subsidiaries, associates, major shareholders, state institutions, and auditors.

Generates:
1. vnresearch/data/market_structure/vn_corporate_network_edges.parquet
2. vnresearch/data/market_structure/vn_corporate_network_nodes.parquet
3. vnresearch/figures_overview/fig6_corporate_network_graph.png (PNG only, no PDF)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
import networkx as nx
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
STRUCT_DIR = os.path.join(DATA_DIR, "market_structure")
MARKET_DIR = os.path.join(DATA_DIR, "market")
OUT_DIR = os.path.join(BASE_DIR, "figures_overview")

os.makedirs(STRUCT_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

STATE_KEYWORDS = [
    "bộ ", "ủy ban", "scic", "ngân hàng nhà nước", "tổng công ty đầu tư và kinh doanh vốn",
    "ubnd", "tập đoàn dầu khí", "tập đoàn điện lực", "tập đoàn bưu chính", "bộ giao thông",
    "bộ công thương", "tổng công ty hàng không", "viettel", "bộ quốc phòng"
]

AUDIT_KEYWORDS = ["kpmg", "ernst & young", "deloitte", "pwc", "price waterhouse", "a&c", "vaaco", "aasc"]

def build_network():
    print("=" * 80)
    print("STARTING CORPORATE NETWORK GRAPH GENERATION")
    print("=" * 80)

    # 1. Load input datasets
    sub_path = os.path.join(STRUCT_DIR, "vn_subsidiaries_affiliates.parquet")
    sh_path = os.path.join(STRUCT_DIR, "vn_shareholders_history.parquet")
    stocks_path = os.path.join(MARKET_DIR, "vn_stock_industry_classification.parquet")

    if not os.path.exists(sub_path) or not os.path.exists(sh_path):
        print("    ! Error: Required input parquet files not found.")
        return

    df_sub = pl.read_parquet(sub_path)
    df_sh = pl.read_parquet(sh_path)
    df_stocks = pl.read_parquet(stocks_path) if os.path.exists(stocks_path) else pl.DataFrame()

    stock_info = {}
    if len(df_stocks) > 0:
        for r in df_stocks.iter_rows(named=True):
            stock_info[r["symbol"]] = {
                "name": r.get("company_name", r["symbol"]),
                "floor": r.get("floor", "HOSE"),
                "sector": r.get("icb_level2_en", "General")
            }

    # 2. Build Directed Multigraph
    G = nx.DiGraph()
    edges = []

    # 2.1 Add Subsidiary, Associate, Auditor, Parent links
    for r in df_sub.iter_rows(named=True):
        src = r["symbol"]
        tgt_raw = r["relation_name_vn"] or r["relation_name_en"] or f"REL_{r['relation_id']}"
        tgt = tgt_raw.strip()
        rtype = r["relation_type"] or "RELATED"
        w = float(r["ownership_pct"] or 50.0)
        cap = float(r["investment_capital"] or 0.0)
        dt = r["effective_date"] or r["year"] or "2024"

        # Determine target type
        tgt_lower = tgt.lower()
        if any(kw in tgt_lower for kw in AUDIT_KEYWORDS):
            target_type = "AUDITOR"
            edge_type = "AUDITOR_SERVICE"
        elif rtype == "SUBSIDIARY":
            target_type = "SUBSIDIARY"
            edge_type = "SUBSIDIARY_CONTROL"
        elif rtype == "ASSOCIATE":
            target_type = "ASSOCIATE"
            edge_type = "ASSOCIATE_AFFILIATE"
        elif rtype == "PARENT":
            target_type = "PARENT_GROUP"
            edge_type = "PARENT_GROUP"
        else:
            target_type = "AFFILIATED_ENTITY"
            edge_type = rtype

        edges.append({
            "source_id": src,
            "source_name": stock_info.get(src, {}).get("name", src),
            "source_type": "LISTED_EQUITY",
            "target_id": tgt,
            "target_name": tgt,
            "target_type": target_type,
            "relationship_type": edge_type,
            "ownership_pct": w if w > 0 else None,
            "capital_value_vnd": cap if cap > 0 else None,
            "effective_date": str(dt)[:10],
            "weight": min(1.0, max(0.05, w / 100.0 if w > 0 else 0.5)),
            "is_directed": True
        })
        G.add_edge(src, tgt, weight=min(1.0, max(0.05, w / 100.0 if w > 0 else 0.5)), rel=edge_type)

    # 2.2 Add Major Shareholders & State Control links
    for r in df_sh.iter_rows(named=True):
        holder = str(r["shareholder_name"] or "").strip()
        tgt = r["symbol"]
        if not holder:
            continue
        stype = r["shareholder_type"]
        w = float(r["ownership_pct"] or 1.0)
        dt = r["effective_date"] or "2024"
        role = r["role_type"] or ""

        holder_lower = holder.lower()
        if any(kw in holder_lower for kw in STATE_KEYWORDS):
            src_type = "STATE_OWNED_ENTITY"
            edge_type = "STATE_EQUITY_CONTROL"
        elif stype == "COMPANY":
            src_type = "INSTITUTIONAL_INVESTOR"
            edge_type = "INSTITUTIONAL_BLOCK_HOLDING"
        else:
            src_type = "INDIVIDUAL_INSIDER"
            edge_type = "INSIDER_MANAGEMENT_HOLDING"

        edges.append({
            "source_id": holder,
            "source_name": holder,
            "source_type": src_type,
            "target_id": tgt,
            "target_name": stock_info.get(tgt, {}).get("name", tgt),
            "target_type": "LISTED_EQUITY",
            "relationship_type": edge_type,
            "ownership_pct": w if w > 0 else None,
            "capital_value_vnd": None,
            "effective_date": str(dt)[:10],
            "weight": min(1.0, max(0.05, w / 100.0 if w > 0 else 0.1)),
            "is_directed": True
        })
        G.add_edge(holder, tgt, weight=min(1.0, max(0.05, w / 100.0 if w > 0 else 0.1)), rel=edge_type)

    print(f">>> Constructed Network Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} directed edges")

    # 3. Compute Centrality Metrics
    print(">>> Computing PageRank and Degree Centrality Metrics...")
    pagerank_scores = nx.pagerank(G, alpha=0.85, max_iter=200)
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())

    # Build Node Records
    nodes = []
    for node in G.nodes():
        node_str = str(node)
        node_lower = node_str.lower()
        
        # Determine classification
        if node_str in stock_info:
            ntype = "LISTED_EQUITY"
            nname = stock_info[node_str]["name"]
            sector = stock_info[node_str]["sector"]
        elif any(kw in node_lower for kw in STATE_KEYWORDS):
            ntype = "STATE_OWNED_ENTITY"
            nname = node_str
            sector = "Government & State Institutions"
        elif any(kw in node_lower for kw in AUDIT_KEYWORDS):
            ntype = "AUDIT_FIRM"
            nname = node_str
            sector = "Auditing & Assurance"
        elif "công ty" in node_lower or "tổng công ty" in node_lower or "jsc" in node_lower or "corp" in node_lower or "ltd" in node_lower:
            ntype = "CORPORATE_AFFILIATE"
            nname = node_str
            sector = "Corporate Enterprise"
        else:
            ntype = "INDIVIDUAL_OR_OTHER"
            nname = node_str
            sector = "Private Individual / Family"

        nodes.append({
            "node_id": node_str,
            "node_name": nname,
            "node_type": ntype,
            "sector": sector,
            "in_degree": int(in_degrees.get(node, 0)),
            "out_degree": int(out_degrees.get(node, 0)),
            "total_degree": int(in_degrees.get(node, 0) + out_degrees.get(node, 0)),
            "pagerank": float(pagerank_scores.get(node, 0.0))
        })

    # 4. Serialize to Parquet
    df_edges = pl.DataFrame(edges).sort(["source_id", "target_id"])
    df_nodes = pl.DataFrame(nodes).sort("pagerank", descending=True)

    edges_path = os.path.join(STRUCT_DIR, "vn_corporate_network_edges.parquet")
    nodes_path = os.path.join(STRUCT_DIR, "vn_corporate_network_nodes.parquet")

    df_edges.write_parquet(edges_path)
    df_nodes.write_parquet(nodes_path)

    print(f"    - Saved vn_corporate_network_edges.parquet ({len(df_edges):,} directed edges)")
    print(f"    - Saved vn_corporate_network_nodes.parquet ({len(df_nodes):,} entity nodes)")



if __name__ == "__main__":
    build_network()

#!/usr/bin/env python3
"""
hypoq CLI Runner.
Supports multi-agent orchestration, hypothesis search, research execution, and web dashboard.

Usage:
    python run.py --agents=codex,claude,hermes
    python run.py --agent=claude --topic="Order flow imbalance price impact"
    python run.py --search="tightly held shares"
    python run.py --list [--status=ACCEPTED]
    python run.py --agents=claude,codex --topic="..." --execute   # actually launch the agent CLIs
    python run.py --audit                                          # integrity-check every research folder
    python run.py --web [--port=8080]
"""

import sys
import os
import argparse
from typing import List

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from hypoq.db import (
    init_db,
    search_hypotheses,
    list_hypotheses,
    get_hypothesis,
    DEFAULT_DB_PATH,
)
from hypoq.harness import (
    ResearchSession,
    SUPPORTED_AGENTS,
    audit_research,
    list_research_dirs,
    print_audit,
)


def format_table(rows: List[dict], headers: List[str], keys: List[str]):
    """Pretty prints a terminal table."""
    if not rows:
        print("No records found.")
        return

    widths = [len(h) for h in headers]
    for r in rows:
        for i, k in enumerate(keys):
            val = str(r.get(k, ""))
            if len(val) > widths[i]:
                widths[i] = min(len(val), 50)

    # Print header
    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * widths[i] for i in range(len(headers)))
    print(header_line)
    print(sep_line)

    for r in rows:
        vals = []
        for i, k in enumerate(keys):
            val = str(r.get(k, ""))
            if len(val) > widths[i]:
                val = val[: widths[i] - 3] + "..."
            vals.append(val.ljust(widths[i]))
        print(" | ".join(vals))


def main():
    parser = argparse.ArgumentParser(
        description="hypoq: Hypothesis × Quant Research Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--agents",
        type=str,
        help="Comma-separated agent names (e.g., --agents=codex,claude,hermes)",
    )
    parser.add_argument(
        "--agent",
        type=str,
        help="Single agent name (e.g., --agent=claude)",
    )
    parser.add_argument(
        "--topic",
        type=str,
        help="Target research topic or hypothesis idea",
    )
    parser.add_argument(
        "--domain",
        type=str,
        help="Research domain (microstructure, derivatives_basis, corporate_networks, macro_transmission, fundamental_factors, free_float_wedges)",
    )
    parser.add_argument(
        "--search",
        type=str,
        help="Search existing hypotheses in database using dense semantic vector similarity",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all hypotheses recorded in the database",
    )
    parser.add_argument(
        "--status",
        type=str,
        choices=["PENDING", "RUNNING", "ACCEPTED", "REJECTED", "INCONCLUSIVE"],
        help="Filter hypotheses by status",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Launch the agent CLIs (default only writes their prompts under runs/<session>/)",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Verify every researches/* folder is reproducible and consistent with the database",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Start the interactive research web dashboard",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the web dashboard (default: 8080)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=DEFAULT_DB_PATH,
        help="Path to SQLite research database",
    )

    args = parser.parse_args()

    # Ensure database is initialized
    init_db(args.db)
    if not list_hypotheses(limit=1, db_path=args.db):
        import glob
        n_records = len(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "researches", "*", "results", "hypothesis_record.json")))
        if n_records:
            print(f"ℹ️  The database is empty but {n_records} study records exist: run `make db-import` to load them (and create the embeddings).")

    # Mode 1: Search Hypotheses
    if args.search:
        print(f"\n🔍 [Dense Semantic Search] Query: \"{args.search}\"")
        matches = search_hypotheses(args.search, top_k=10, db_path=args.db)
        if not matches:
            print("No matching hypotheses found in database.")
        else:
            print(f"Found {len(matches)} match(es) ranked by semantic similarity:\n")
            format_table(
                matches,
                headers=["ID", "Status", "Similarity", "Domain", "Metric", "p-val", "Title"],
                keys=["id", "status", "similarity_score", "domain", "metric_value", "p_value", "title"],
            )
            print("\nTop Semantic Match Details:")
            top = matches[0]
            print(f"Title      : {top['title']}")
            print(f"Similarity : {top.get('similarity_score', 0.0):.4f}")
            print(f"H0         : {top['null_hypothesis']}")
            print(f"H1         : {top['alt_hypothesis']}")
            print(f"Findings   : {top.get('summary_findings', 'N/A')}")
            if top.get("pdf_path"):
                print(f"PDF        : {top['pdf_path']}")
        return

    # Mode 2: List Hypotheses
    if args.list:
        print("\n📋 [hypoq Hypotheses Inventory]")
        records = list_hypotheses(
            status=args.status, domain=args.domain, limit=100, db_path=args.db
        )
        format_table(
            records,
            headers=["ID", "Status", "Agent", "Domain", "Metric", "p-val", "Title"],
            keys=["id", "status", "agent_name", "domain", "metric_value", "p_value", "title"],
        )
        return

    # Mode 2b: Research integrity audit
    if args.audit:
        reports = [audit_research(d, args.db) for d in list_research_dirs()]
        print_audit(reports)
        sys.exit(0 if all(r["ok"] for r in reports) else 1)

    # Mode 3: Start Web Server
    if args.web:
        from api import run_server
        run_server(port=args.port, db_path=args.db)
        return

    # Mode 4: Multi-Agent Session Launch
    agents_list = []
    if args.agents:
        agents_list = [a.strip() for a in args.agents.split(",") if a.strip()]
    elif args.agent:
        agents_list = [args.agent.strip()]

    if agents_list:
        session = ResearchSession(
            agents=agents_list,
            topic=args.topic,
            domain=args.domain,
            db_path=args.db,
        )
        session.run(verbose=True, execute=args.execute)
        return

    # If no flags provided, show help
    parser.print_help()


if __name__ == "__main__":
    main()

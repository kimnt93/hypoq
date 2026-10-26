#!/usr/bin/env python3
"""
Regenerate the "Studies produced with the harness" table in README.md from the hypothesis database,
so every status and number in the README is read back from the computed rows, never typed.
Run after `make seed`:  python .agents/skills/hypoq-loop/scripts/update_study_table.py
"""
import os
import re
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, BASE_DIR)
from hypoq.db import list_hypotheses  # noqa: E402

README = os.path.join(BASE_DIR, "README.md")
START, END = "<!-- STUDIES:START -->", "<!-- STUDIES:END -->"
PARADIGM = {"res_001": "Econometrics", "res_002": "Event study", "res_003": "Machine learning", "res_004": "Deep learning",
            "res_005": "Reinforcement learning", "res_006": "Event study", "res_007": "Event study", "res_008": "Predictive regression",
            "res_009": "Rule-based strategy", "res_010": "Predictive regression", "res_011": "Cross-sectional sort", "res_012": "Predictive regression",
            "res_013": "Predictive regression", "res_014": "Safe-haven test", "res_015": "Predictive regression", "res_016": "Calendar effect",
            "res_017": "Return decomposition", "res_018": "Cross-sectional sort", "res_019": "Predictive regression", "res_020": "Calendar effect",
            "res_021": "Predictive regression", "res_022": "Machine learning"}
DOMAIN = {"macro_transmission": "Macro transmission", "sentiment_events": "Sentiment / events", "fundamental_factors": "Factors / allocation",
          "crypto": "Crypto", "market_microstructure": "Market microstructure"}


def fmt(v):
    return f"{v:.3g}" if abs(v) >= 0.001 or v == 0 else f"{v:.1e}"


def table() -> str:
    rows = sorted(list_hypotheses(), key=lambda h: h["report_dir"] or "")
    out = ["| ID | Domain | Paradigm | Status | Test statistic (p-value) | Title |", "| :--- | :--- | :--- | :---: | :--- | :--- |"]
    for h in rows:
        rd = h["report_dir"] or ""
        key = os.path.basename(rd)[:7]
        if not rd.startswith("researches/"):
            continue
        p = f" ({'p' if (h['p_value'] or 0) >= 0.001 else 'p'} = {fmt(h['p_value'])})" if h["p_value"] is not None else ""
        metric = f"{h['metric_name']} = {fmt(h['metric_value'])}{p}".replace("|", "\\|")
        out.append(f"| [`{key[4:]}`]({rd}) | {DOMAIN.get(h['domain'], h['domain'])} | {PARADIGM.get(key, '')} | **{h['status']}** | {metric} | {h['title'].replace('|', '/')} |")
    return "\n".join(out)


def main():
    text = open(README, encoding="utf-8").read()
    block = f"{START}\n{table()}\n{END}"
    if START in text:
        text = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda m: block, text, flags=re.S)
    else:
        text = re.sub(r"\| ID \| Domain \| Paradigm.*?(?=\n\n)", lambda m: block, text, count=1, flags=re.S)
    open(README, "w", encoding="utf-8").write(text)
    print("README study table updated")


if __name__ == "__main__":
    main()

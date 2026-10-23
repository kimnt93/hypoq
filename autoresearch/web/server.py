"""
AutoResearch Web Dashboard & Explorer Server.
Serves interactive HTML UI for:
1. Hypotheses tracker (Accepted / Rejected / Inconclusive with metrics & test stats)
2. Dataset Explorer (All 36 Parquet files with live Polars preview)
3. Research Paper & PDF Reports Viewer
4. Real-time FTS5 Search & Deduplication Engine
"""

import os
import sys
import json
import glob
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, List

import polars as pl
from autoresearch.db import (
    get_connection,
    list_hypotheses,
    get_hypothesis,
    search_hypotheses,
    DEFAULT_DB_PATH,
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def get_dataset_inventory() -> List[Dict[str, Any]]:
    """Inspects all parquet files in data/ and gathers metadata."""
    results = []
    parquet_files = sorted(glob.glob(os.path.join(BASE_DIR, "data", "**", "*.parquet"), recursive=True))
    if not parquet_files:
        # Fallback to vnresearch/data
        parquet_files = sorted(glob.glob("data/**/*.parquet", recursive=True))

    for p in parquet_files:
        rel_path = os.path.relpath(p, BASE_DIR)
        category = os.path.basename(os.path.dirname(p))
        size_kb = os.path.getsize(p) / 1024.0
        filename = os.path.basename(p)

        # Quick schema scan via Polars
        try:
            schema = lf.collect_schema()
            col_count = len(schema)
            # Fetch row count
            row_count = lf.select(pl.len()).collect().item()
        except Exception:
            col_count = 0
            row_count = 0

        results.append({
            "path": rel_path,
            "filename": filename,
            "category": category,
            "size_kb": round(size_kb, 1),
            "col_count": col_count,
            "row_count": row_count,
        })
    return results


def get_reports_inventory() -> List[Dict[str, Any]]:
    """Gathers research reports with compiled PDFs."""
    reports = []
    # Search in researches/ and ../paper-hub/0*
    search_patterns = [
        os.path.join(BASE_DIR, "researches", "*", "main.pdf"),
        os.path.join(BASE_DIR, "templates", "*", "main.pdf"),
        os.path.join(BASE_DIR, "..", "015*", "main.pdf"),
        os.path.join(BASE_DIR, "..", "0*", "main.pdf"),
    ]
    seen = set()
    for pattern in search_patterns:
        for pdf in sorted(glob.glob(pattern)):
            abs_pdf = os.path.abspath(pdf)
            if abs_pdf in seen:
                continue
            seen.add(abs_pdf)
            report_dir = os.path.dirname(abs_pdf)
            name = os.path.basename(report_dir)
            size_kb = os.path.getsize(abs_pdf) / 1024.0

            # Check if README or paper.md exists for title
            title = name.replace("_", " ").title()
            for text_f in ["README.md", "paper.md"]:
                tf = os.path.join(report_dir, text_f)
                if os.path.exists(tf):
                    try:
                        with open(tf, "r", encoding="utf-8") as f:
                            first_line = f.readline().strip()
                            if first_line.startswith("#"):
                                title = first_line.lstrip("#").strip()
                                break
                    except Exception:
                        pass

            reports.append({
                "id": name,
                "title": title,
                "dir": os.path.relpath(report_dir, BASE_DIR),
                "pdf_path": os.path.relpath(abs_pdf, BASE_DIR),
                "size_kb": round(size_kb, 1),
            })
    return reports


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AutoResearch Hub — Autonomous Quantitative Research</title>
    <style>
        :root {
            --bg: #0d1117;
            --surface: #161b22;
            --surface-hover: #21262d;
            --border: #30363d;
            --text: #c9d1d9;
            --text-heading: #f0f6fc;
            --muted: #8b949e;
            --accent: #58a6ff;
            --accent-bg: rgba(56, 139, 253, 0.15);
            --accepted: #2ea043;
            --accepted-bg: rgba(46, 160, 67, 0.18);
            --rejected: #f85149;
            --rejected-bg: rgba(248, 81, 73, 0.18);
            --inconclusive: #d29922;
            --inconclusive-bg: rgba(210, 153, 34, 0.18);
            --running: #1f6feb;
            --running-bg: rgba(31, 111, 235, 0.18);
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            line-height: 1.5;
            padding: 24px;
        }
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            padding-bottom: 20px;
            margin-bottom: 24px;
        }
        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .brand h1 {
            font-size: 24px;
            color: var(--text-heading);
            font-weight: 700;
        }
        .brand span {
            background: var(--accent-bg);
            color: var(--accent);
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }
        nav {
            display: flex;
            gap: 8px;
        }
        nav button {
            background: transparent;
            border: 1px solid var(--border);
            color: var(--text);
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.2s;
        }
        nav button.active, nav button:hover {
            background: var(--surface-hover);
            color: var(--text-heading);
            border-color: var(--accent);
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }
        .stat-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
        }
        .stat-card .label { font-size: 13px; color: var(--muted); }
        .stat-card .value { font-size: 28px; font-weight: 700; color: var(--text-heading); margin-top: 4px; }
        
        .search-bar {
            display: flex;
            gap: 12px;
            margin-bottom: 24px;
        }
        .search-bar input {
            flex: 1;
            background: var(--surface);
            border: 1px solid var(--border);
            color: var(--text-heading);
            padding: 10px 14px;
            border-radius: 6px;
            font-size: 14px;
        }
        .search-bar input:focus {
            outline: none;
            border-color: var(--accent);
        }
        .search-bar button {
            background: var(--accent);
            color: #ffffff;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
        }

        .filter-tags {
            display: flex;
            gap: 8px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .tag-btn {
            background: var(--surface);
            border: 1px solid var(--border);
            color: var(--muted);
            padding: 4px 12px;
            border-radius: 16px;
            font-size: 13px;
            cursor: pointer;
        }
        .tag-btn.active {
            background: var(--accent-bg);
            border-color: var(--accent);
            color: var(--accent);
        }

        .tab-content { display: none; }
        .tab-content.active { display: block; }

        .hypo-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 16px;
        }
        .hypo-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
            transition: border-color 0.2s;
        }
        .hypo-card:hover { border-color: var(--muted); }
        .hypo-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 12px;
        }
        .hypo-title {
            font-size: 18px;
            font-weight: 600;
            color: var(--text-heading);
        }
        .badge {
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
        }
        .badge.ACCEPTED { background: var(--accepted-bg); color: var(--accepted); border: 1px solid var(--accepted); }
        .badge.REJECTED { background: var(--rejected-bg); color: var(--rejected); border: 1px solid var(--rejected); }
        .badge.INCONCLUSIVE { background: var(--inconclusive-bg); color: var(--inconclusive); border: 1px solid var(--inconclusive); }
        .badge.RUNNING { background: var(--running-bg); color: var(--running); border: 1px solid var(--running); }
        .badge.PENDING { background: var(--surface-hover); color: var(--muted); border: 1px solid var(--border); }
        
        .hypo-body { margin-bottom: 16px; }
        .hypo-statement { color: var(--text); margin-bottom: 8px; font-size: 14px; }
        .hypo-h0-h1 {
            background: var(--bg);
            border-left: 3px solid var(--accent);
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 13px;
            margin: 8px 0;
        }
        .hypo-meta {
            display: flex;
            gap: 16px;
            font-size: 13px;
            color: var(--muted);
            margin-top: 12px;
            flex-wrap: wrap;
        }
        .hypo-meta strong { color: var(--text-heading); }
        .hypo-actions {
            display: flex;
            gap: 10px;
            margin-top: 14px;
        }
        .btn-action {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: var(--surface-hover);
            border: 1px solid var(--border);
            color: var(--accent);
            text-decoration: none;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 500;
            transition: all 0.2s;
        }
        .btn-action:hover {
            background: var(--accent-bg);
            border-color: var(--accent);
        }

        table.data-table {
            width: 100%;
            border-collapse: collapse;
            background: var(--surface);
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--border);
        }
        table.data-table th, table.data-table td {
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid var(--border);
            font-size: 14px;
        }
        table.data-table th {
            background: var(--surface-hover);
            color: var(--text-heading);
            font-weight: 600;
        }
        table.data-table tr:hover { background: var(--surface-hover); }

        .modal {
            display: none;
            position: fixed;
            top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0, 0, 0, 0.75);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }
        .modal.open { display: flex; }
        .modal-content {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 10px;
            width: 90%;
            max-width: 950px;
            max-height: 85vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 16px 20px;
            border-bottom: 1px solid var(--border);
        }
        .modal-header h3 { color: var(--text-heading); }
        .modal-close {
            background: none;
            border: none;
            color: var(--muted);
            font-size: 20px;
            cursor: pointer;
        }
        .modal-body {
            padding: 20px;
            overflow-y: auto;
            flex: 1;
        }
    </style>
</head>
<body>

<header>
    <div class="brand">
        <h1>AutoResearch Hub</h1>
        <span>v1.0 • Autonomous Quantitative Lab</span>
    </div>
    <nav>
        <button class="active" onclick="showTab('hypotheses')">Hypotheses & Results</button>
        <button onclick="showTab('datasets')">Data Lake (36 Parquets)</button>
        <button onclick="showTab('reports')">Research Manuscripts & PDFs</button>
    </nav>
</header>

<div class="stats-grid">
    <div class="stat-card">
        <div class="label">Total Hypotheses</div>
        <div class="value" id="stat-total">--</div>
    </div>
    <div class="stat-card">
        <div class="label">Accepted (Supported)</div>
        <div class="value" style="color: var(--accepted);" id="stat-accepted">--</div>
    </div>
    <div class="stat-card">
        <div class="label">Rejected (Null Maintained)</div>
        <div class="value" style="color: var(--rejected);" id="stat-rejected">--</div>
    </div>
    <div class="stat-card">
        <div class="label">Data Lake Parquets</div>
        <div class="value" style="color: var(--accent);" id="stat-datasets">36</div>
    </div>
</div>

<!-- TAB 1: HYPOTHESES -->
<div id="tab-hypotheses" class="tab-content active">
    <div class="search-bar">
        <input type="text" id="searchInput" placeholder="Search hypotheses using FTS5 BM25 text match (e.g. tightly held, basis, order flow)..." onkeydown="if(event.key==='Enter') executeSearch()">
        <button onclick="executeSearch()">Search</button>
        <button style="background: var(--surface-hover); border: 1px solid var(--border); color: var(--text);" onclick="resetFilter()">Reset</button>
    </div>

    <div class="filter-tags">
        <button class="tag-btn active" onclick="filterStatus('')">All Statuses</button>
        <button class="tag-btn" onclick="filterStatus('ACCEPTED')">🟢 Accepted</button>
        <button class="tag-btn" onclick="filterStatus('REJECTED')">🔴 Rejected</button>
        <button class="tag-btn" onclick="filterStatus('INCONCLUSIVE')">🟡 Inconclusive</button>
        <button class="tag-btn" onclick="filterStatus('RUNNING')">🔵 Running</button>
    </div>

    <div class="hypo-grid" id="hypoContainer">
        <!-- Rendered via JS -->
    </div>
</div>

<!-- TAB 2: DATASETS -->
<div id="tab-datasets" class="tab-content">
    <h2 style="color: var(--text-heading); margin-bottom: 16px;">Quantitative Data Lake Architecture (36 Parquet Datasets)</h2>
    <table class="data-table">
        <thead>
            <tr>
                <th>Category</th>
                <th>Filename</th>
                <th>Rows</th>
                <th>Columns</th>
                <th>Size</th>
                <th>Actions</th>
            </tr>
        </thead>
        <tbody id="datasetsTableBody">
            <!-- Rendered via JS -->
        </tbody>
    </table>
</div>

<!-- TAB 3: REPORTS & PDFS -->
<div id="tab-reports" class="tab-content">
    <h2 style="color: var(--text-heading); margin-bottom: 16px;">Compiled Research Papers & Empirical Manuscripts</h2>
    <table class="data-table">
        <thead>
            <tr>
                <th>Study ID</th>
                <th>Title & Topic</th>
                <th>PDF Size</th>
                <th>PDF Document</th>
            </tr>
        </thead>
        <tbody id="reportsTableBody">
            <!-- Rendered via JS -->
        </tbody>
    </table>
</div>

<!-- PREVIEW MODAL -->
<div class="modal" id="previewModal">
    <div class="modal-content">
        <div class="modal-header">
            <h3 id="modalTitle">Dataset Preview</h3>
            <button class="modal-close" onclick="closeModal()">&times;</button>
        </div>
        <div class="modal-body" id="modalBody">
            <!-- Rendered preview table -->
        </div>
    </div>
</div>

<script>
let currentStatusFilter = '';

function showTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('nav button').forEach(el => el.classList.remove('active'));
    document.getElementById('tab-' + tabName).classList.add('active');
    event.target.classList.add('active');

    if (tabName === 'datasets') loadDatasets();
    if (tabName === 'reports') loadReports();
}

async function loadHypotheses(query = '', status = '') {
    let url = '/api/hypotheses?';
    if (query) url += 'q=' + encodeURIComponent(query) + '&';
    if (status) url += 'status=' + encodeURIComponent(status);

    const resp = await fetch(url);
    const data = await resp.json();

    const container = document.getElementById('hypoContainer');
    container.innerHTML = '';

    let total = data.length;
    let accepted = 0;
    let rejected = 0;

    data.forEach(h => {
        if (h.status === 'ACCEPTED') accepted++;
        if (h.status === 'REJECTED') rejected++;

        const card = document.createElement('div');
        card.className = 'hypo-card';
        card.innerHTML = `
            <div class="hypo-header">
                <div>
                    <div class="hypo-title">#${h.id} — ${h.title}</div>
                    <div style="font-size: 12px; color: var(--muted); margin-top: 4px;">Domain: <strong>${h.domain || 'General'}</strong> | Agent: <strong>${h.agent_name || 'Harness'}</strong></div>
                </div>
                <span class="badge ${h.status}">${h.status}</span>
            </div>
            <div class="hypo-body">
                <div class="hypo-statement">${h.statement}</div>
                <div class="hypo-h0-h1">
                    <div><strong>H0 (Null):</strong> ${h.null_hypothesis}</div>
                    <div><strong>H1 (Alt):</strong> ${h.alt_hypothesis}</div>
                </div>
                <div class="hypo-meta">
                    <div>Primary Metric: <strong>${h.metric_name || 'N/A'}: ${h.metric_value !== null ? h.metric_value : '--'}</strong></div>
                    <div>p-Value: <strong>${h.p_value !== null ? h.p_value : '--'}</strong></div>
                    <div>Updated: <strong>${(h.updated_at || '').substring(0, 10)}</strong></div>
                </div>
                ${h.summary_findings ? `<div style="margin-top: 10px; font-size: 13px; color: var(--text);"><strong>Findings:</strong> ${h.summary_findings}</div>` : ''}
            </div>
            <div class="hypo-actions">
                ${h.pdf_path ? `<a href="/pdf?path=${encodeURIComponent(h.pdf_path)}" target="_blank" class="btn-action">📄 View PDF Report</a>` : ''}
                ${h.report_dir ? `<span class="btn-action" style="cursor: default; color: var(--muted);">📁 ${h.report_dir}</span>` : ''}
            </div>
        `;
        container.appendChild(card);
    });

    document.getElementById('stat-total').innerText = total;
    document.getElementById('stat-accepted').innerText = accepted;
    document.getElementById('stat-rejected').innerText = rejected;
}

function filterStatus(status) {
    currentStatusFilter = status;
    document.querySelectorAll('.filter-tags button').forEach(b => b.classList.remove('active'));
    event.target.classList.add('active');
    loadHypotheses(document.getElementById('searchInput').value, status);
}

function executeSearch() {
    const q = document.getElementById('searchInput').value;
    loadHypotheses(q, currentStatusFilter);
}

function resetFilter() {
    document.getElementById('searchInput').value = '';
    currentStatusFilter = '';
    document.querySelectorAll('.filter-tags button').forEach((b, i) => {
        b.classList.toggle('active', i === 0);
    });
    loadHypotheses();
}

async function loadDatasets() {
    const resp = await fetch('/api/datasets');
    const datasets = await resp.json();
    const tbody = document.getElementById('datasetsTableBody');
    tbody.innerHTML = '';
    document.getElementById('stat-datasets').innerText = datasets.length;

    datasets.forEach(d => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><span style="background: var(--surface-hover); padding: 3px 8px; border-radius: 4px; font-size: 12px;">${d.category}</span></td>
            <td><strong>${d.filename}</strong></td>
            <td>${d.row_count.toLocaleString()}</td>
            <td>${d.col_count} cols</td>
            <td>${d.size_kb} KB</td>
            <td><button class="btn-action" onclick="previewDataset('${encodeURIComponent(d.path)}')">🔍 Preview Data</button></td>
        `;
        tbody.appendChild(tr);
    });
}

async function previewDataset(path) {
    document.getElementById('modalTitle').innerText = 'Loading dataset preview: ' + decodeURIComponent(path);
    document.getElementById('modalBody').innerHTML = '<div>Loading parquet samples using Polars...</div>';
    document.getElementById('previewModal').classList.add('open');

    const resp = await fetch('/api/dataset_preview?path=' + path);
    const data = await resp.json();

    if (data.error) {
        document.getElementById('modalBody').innerHTML = `<div style="color: var(--rejected);">Error: ${data.error}</div>`;
        return;
    }

    document.getElementById('modalTitle').innerText = `Dataset: ${data.filename} (${data.row_count.toLocaleString()} rows, ${data.cols.length} cols)`;

    let tableHtml = '<table class="data-table" style="font-size: 12px;"><thead><tr>';
    data.cols.forEach(c => { tableHtml += `<th>${c}</th>`; });
    tableHtml += '</tr></thead><tbody>';

    data.rows.forEach(r => {
        tableHtml += '<tr>';
        r.forEach(val => {
            tableHtml += `<td>${val !== null ? val : '<span style="color:var(--muted)">null</span>'}</td>`;
        });
        tableHtml += '</tr>';
    });
    tableHtml += '</tbody></table>';

    document.getElementById('modalBody').innerHTML = tableHtml;
}

function closeModal() {
    document.getElementById('previewModal').classList.remove('open');
}

async function loadReports() {
    const resp = await fetch('/api/reports');
    const reports = await resp.json();
    const tbody = document.getElementById('reportsTableBody');
    tbody.innerHTML = '';

    reports.forEach(r => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><code>${r.id}</code></td>
            <td><strong>${r.title}</strong></td>
            <td>${r.size_kb} KB</td>
            <td><a href="/pdf?path=${encodeURIComponent(r.pdf_path)}" target="_blank" class="btn-action">📄 Open Manuscript PDF</a></td>
        `;
        tbody.appendChild(tr);
    });
}

// Initial load
loadHypotheses();
</script>
</body>
</html>
"""


class AutoResearchHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler for AutoResearch web dashboard."""

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        # 1. Main Dashboard HTML
        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
            return

        # 2. PDF Serving
        if path == "/pdf":
            rel_pdf = query.get("path", [""])[0]
            # Security: normalize path and prevent directory traversal outside BASE_DIR / parent
            target_path = os.path.abspath(os.path.join(BASE_DIR, rel_pdf))
            allowed_root = os.path.abspath(os.path.join(BASE_DIR, ".."))
            if not os.path.exists(target_path):
                alt_path = os.path.abspath(os.path.join(BASE_DIR, "..", rel_pdf))
                if os.path.exists(alt_path):
                    target_path = alt_path
            if not target_path.startswith(allowed_root) or not os.path.exists(target_path):
                self.send_error(404, "PDF file not found")
                return

            try:
                with open(target_path, "rb") as f:
                    pdf_bytes = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/pdf")
                self.send_header("Content-Disposition", f"inline; filename={os.path.basename(target_path)}")
                self.send_header("Content-Length", str(len(pdf_bytes)))
                self.end_headers()
                self.wfile.write(pdf_bytes)
            except Exception as e:
                self.send_error(500, f"Error reading PDF: {e}")
            return

        # 3. API: Hypotheses
        if path == "/api/hypotheses":
            q_str = query.get("q", [""])[0]
            status = query.get("status", [""])[0]
            db_path = os.environ.get("AUTORESEARCH_DB", DEFAULT_DB_PATH)

            if q_str:
                items = search_hypotheses(q_str, top_k=25, db_path=db_path)
                if status:
                    items = [i for i in items if i.get("status") == status.upper()]
            else:
                items = list_hypotheses(status=status, limit=100, db_path=db_path)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(items).encode("utf-8"))
            return

        # 4. API: Datasets
        if path == "/api/datasets":
            datasets = get_dataset_inventory()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(datasets).encode("utf-8"))
            return

        # 5. API: Dataset Preview (Polars)
        if path == "/api/dataset_preview":
            rel_path = query.get("path", [""])[0]
            abs_path = os.path.abspath(os.path.join(BASE_DIR, rel_path))
            if not os.path.exists(abs_path):
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Parquet file not found"}).encode("utf-8"))
                return

            try:
                lf = pl.scan_parquet(abs_path)
                cols = lf.collect_schema().names()
                row_count = lf.select(pl.len()).collect().item()
                preview_df = lf.limit(15).collect()
                # Convert rows to serializable types
                rows = []
                for row in preview_df.iter_rows():
                    rows.append([str(c) if c is not None else None for c in row])

                resp_data = {
                    "filename": os.path.basename(abs_path),
                    "cols": cols,
                    "row_count": row_count,
                    "rows": rows,
                }
            except Exception as e:
                resp_data = {"error": str(e)}

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp_data).encode("utf-8"))
            return

        # 6. API: Reports
        if path == "/api/reports":
            reports = get_reports_inventory()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(reports).encode("utf-8"))
            return

        self.send_error(404, "Endpoint not found")

    def log_message(self, format, *args):
        # Concise logging
        sys.stderr.write(f"[AutoResearch Web] {self.address_string()} - {args[0]} {args[1]}\n")


def run_server(port: int = 8080, db_path: str = DEFAULT_DB_PATH):
    """Starts the AutoResearch web server."""
    os.environ["AUTORESEARCH_DB"] = db_path
    server_address = ("0.0.0.0", port)
    httpd = HTTPServer(server_address, AutoResearchHandler)
    print(f"\n🌐 [AutoResearch Dashboard] Server live at http://127.0.0.1:{port}")
    print(f"   Database : {db_path}")
    print(f"   Exploring: Hypotheses, 36 Parquet Datasets, PDF Reports, FTS5 Dedup Search")
    print(f"   Press Ctrl+C to terminate.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping AutoResearch web server...")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    run_server(port=port)

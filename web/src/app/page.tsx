'use client';

import { useState, useEffect } from 'react';

interface HypothesisItem {
  id: number;
  title: string;
  statement: string;
  null_hypothesis: string;
  alt_hypothesis: string;
  domain: string;
  agent_name: string | null;
  metric_name: string | null;
  metric_value: number | null;
  p_value: number | null;
  status: string;
  summary_findings: string | null;
  report_dir: string | null;
  pdf_path: string | null;
}

interface DatasetItem {
  filename: string;
  category: string;
  relativePath: string;
  sizeKb: number;
  updatedAt: string;
}

interface ReportItem {
  id: string;
  title: string;
  pdfPath: string;
  sizeKb: number;
  updatedAt: string;
}

interface PreviewData {
  filename: string;
  totalRows?: number;
  columns?: string[];
  sampleRows?: Record<string, unknown>[];
  error?: string;
}

export default function HomePage() {
  const [activeTab, setActiveTab] = useState<'hypotheses' | 'datasets' | 'reports'>('hypotheses');
  const [hypotheses, setHypotheses] = useState<HypothesisItem[]>([]);
  const [statusFilter, setStatusFilter] = useState<string>('');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [datasets, setDatasets] = useState<DatasetItem[]>([]);
  const [reports, setReports] = useState<ReportItem[]>([]);
  const [previewData, setPreviewData] = useState<PreviewData | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  const fetchHypotheses = async (query = '', status = '') => {
    try {
      let url = '/api/hypotheses?';
      if (query) url += `q=${encodeURIComponent(query)}&`;
      if (status) url += `status=${encodeURIComponent(status)}`;
      const res = await fetch(url);
      const data = await res.json();
      if (Array.isArray(data)) setHypotheses(data);
    } catch (e) {
      console.error(e);
    }
  };

  const fetchDatasets = async () => {
    try {
      const res = await fetch('/api/datasets');
      const data = await res.json();
      if (Array.isArray(data)) setDatasets(data);
    } catch (e) {
      console.error(e);
    }
  };

  const fetchReports = async () => {
    try {
      const res = await fetch('/api/reports');
      const data = await res.json();
      if (Array.isArray(data)) setReports(data);
    } catch (e) {
      console.error(e);
    }
  };

  const loadPreview = async (path: string) => {
    setLoading(true);
    setPreviewData({ filename: 'Loading via JS Arquero...', columns: [], sampleRows: [] });
    try {
      const res = await fetch(`/api/dataset-preview?path=${encodeURIComponent(path)}`);
      const data = await res.json();
      setPreviewData(data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHypotheses();
    fetchDatasets();
    fetchReports();
  }, []);

  const totalHypo = hypotheses.length;
  const acceptedHypo = hypotheses.filter((h) => h.status === 'ACCEPTED').length;
  const rejectedHypo = hypotheses.filter((h) => h.status === 'REJECTED').length;

  return (
    <div style={{ maxWidth: 1280, margin: '0 auto' }}>
      {/* Header */}
      <header
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          borderBottom: '1px solid var(--border)',
          paddingBottom: 20,
          marginBottom: 24,
          flexWrap: 'wrap',
          gap: 16,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ fontSize: 28 }}>🔬</span>
          <div>
            <h1 style={{ fontSize: 22, color: 'var(--text-heading)', fontWeight: 700 }}>
              AI AutoResearch Lab
            </h1>
            <p style={{ fontSize: 13, color: 'var(--muted)' }}>
              Autonomous Quantitative Research Network • Empirical Financial Economics
            </p>
          </div>
        </div>

        <nav style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <button
            onClick={() => setActiveTab('hypotheses')}
            style={{
              background: activeTab === 'hypotheses' ? 'var(--surface-hover)' : 'transparent',
              color: activeTab === 'hypotheses' ? 'var(--text-heading)' : 'var(--text)',
              border: '1px solid',
              borderColor: activeTab === 'hypotheses' ? 'var(--accent)' : 'var(--border)',
              padding: '8px 14px',
              borderRadius: 6,
              cursor: 'pointer',
              fontWeight: 600,
            }}
          >
            Hypotheses ({totalHypo})
          </button>
          <button
            onClick={() => setActiveTab('datasets')}
            style={{
              background: activeTab === 'datasets' ? 'var(--surface-hover)' : 'transparent',
              color: activeTab === 'datasets' ? 'var(--text-heading)' : 'var(--text)',
              border: '1px solid',
              borderColor: activeTab === 'datasets' ? 'var(--accent)' : 'var(--border)',
              padding: '8px 14px',
              borderRadius: 6,
              cursor: 'pointer',
              fontWeight: 600,
            }}
          >
            Data Lake (36 Parquets)
          </button>
          <button
            onClick={() => setActiveTab('reports')}
            style={{
              background: activeTab === 'reports' ? 'var(--surface-hover)' : 'transparent',
              color: activeTab === 'reports' ? 'var(--text-heading)' : 'var(--text)',
              border: '1px solid',
              borderColor: activeTab === 'reports' ? 'var(--accent)' : 'var(--border)',
              padding: '8px 14px',
              borderRadius: 6,
              cursor: 'pointer',
              fontWeight: 600,
            }}
          >
            Manuscripts & PDFs ({reports.length})
          </button>
        </nav>
      </header>

      {/* Stats Counter */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
          gap: 16,
          marginBottom: 24,
        }}
      >
        <div
          style={{
            background: 'var(--surface)',
            border: '1px solid var(--border)',
            borderRadius: 8,
            padding: 16,
          }}
        >
          <div style={{ fontSize: 13, color: 'var(--muted)' }}>Total Hypotheses</div>
          <div style={{ fontSize: 26, fontWeight: 700, color: 'var(--text-heading)' }}>
            {totalHypo}
          </div>
        </div>
        <div
          style={{
            background: 'var(--surface)',
            border: '1px solid var(--border)',
            borderRadius: 8,
            padding: 16,
          }}
        >
          <div style={{ fontSize: 13, color: 'var(--muted)' }}>Accepted (Supported)</div>
          <div style={{ fontSize: 26, fontWeight: 700, color: 'var(--accepted)' }}>
            {acceptedHypo}
          </div>
        </div>
        <div
          style={{
            background: 'var(--surface)',
            border: '1px solid var(--border)',
            borderRadius: 8,
            padding: 16,
          }}
        >
          <div style={{ fontSize: 13, color: 'var(--muted)' }}>Rejected (Null Maintained)</div>
          <div style={{ fontSize: 26, fontWeight: 700, color: 'var(--rejected)' }}>
            {rejectedHypo}
          </div>
        </div>
        <div
          style={{
            background: 'var(--surface)',
            border: '1px solid var(--border)',
            borderRadius: 8,
            padding: 16,
          }}
        >
          <div style={{ fontSize: 13, color: 'var(--muted)' }}>Data Lake Parquets</div>
          <div style={{ fontSize: 26, fontWeight: 700, color: 'var(--accent)' }}>36</div>
        </div>
      </div>

      {/* TAB 1: HYPOTHESES */}
      {activeTab === 'hypotheses' && (
        <div>
          {/* Search bar */}
          <div style={{ display: 'flex', gap: 10, marginBottom: 16 }}>
            <input
              type="text"
              placeholder="Search hypotheses with SQLite FTS5 (e.g. tightly held, basis, order flow, monetary)..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') fetchHypotheses(searchQuery, statusFilter);
              }}
              style={{
                flex: 1,
                background: 'var(--surface)',
                border: '1px solid var(--border)',
                borderRadius: 6,
                padding: '10px 14px',
                color: 'var(--text-heading)',
                fontSize: 14,
                outline: 'none',
              }}
            />
            <button
              onClick={() => fetchHypotheses(searchQuery, statusFilter)}
              style={{
                background: 'var(--accent)',
                color: '#fff',
                border: 'none',
                padding: '10px 18px',
                borderRadius: 6,
                fontWeight: 600,
                cursor: 'pointer',
              }}
            >
              Search
            </button>
            <button
              onClick={() => {
                setSearchQuery('');
                setStatusFilter('');
                fetchHypotheses('', '');
              }}
              style={{
                background: 'var(--surface)',
                color: 'var(--text)',
                border: '1px solid var(--border)',
                padding: '10px 14px',
                borderRadius: 6,
                cursor: 'pointer',
              }}
            >
              Reset
            </button>
          </div>

          {/* Filter Chips */}
          <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
            {['', 'ACCEPTED', 'REJECTED', 'INCONCLUSIVE'].map((s) => (
              <button
                key={s}
                onClick={() => {
                  setStatusFilter(s);
                  fetchHypotheses(searchQuery, s);
                }}
                style={{
                  background: statusFilter === s ? 'var(--accent-bg)' : 'var(--surface)',
                  color: statusFilter === s ? 'var(--accent)' : 'var(--muted)',
                  border: '1px solid',
                  borderColor: statusFilter === s ? 'var(--accent)' : 'var(--border)',
                  padding: '4px 12px',
                  borderRadius: 16,
                  fontSize: 13,
                  cursor: 'pointer',
                  fontWeight: 600,
                }}
              >
                {s === '' ? 'All Statuses' : s}
              </button>
            ))}
          </div>

          {/* Hypotheses Cards List */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
            {hypotheses.map((h) => (
              <div
                key={h.id}
                style={{
                  background: 'var(--surface)',
                  border: '1px solid var(--border)',
                  borderRadius: 8,
                  padding: 20,
                }}
              >
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'flex-start',
                    marginBottom: 10,
                  }}
                >
                  <div>
                    <h2
                      style={{ fontSize: 17, color: 'var(--text-heading)', fontWeight: 600 }}
                    >
                      #{h.id} — {h.title}
                    </h2>
                    <p style={{ fontSize: 12, color: 'var(--muted)', marginTop: 2 }}>
                      Domain: <strong>{h.domain}</strong> | Agent: <strong>{h.agent_name}</strong>
                    </p>
                  </div>
                  <span
                    style={{
                      background:
                        h.status === 'ACCEPTED'
                          ? 'var(--accepted-bg)'
                          : h.status === 'REJECTED'
                          ? 'var(--rejected-bg)'
                          : 'var(--inconclusive-bg)',
                      color:
                        h.status === 'ACCEPTED'
                          ? 'var(--accepted)'
                          : h.status === 'REJECTED'
                          ? 'var(--rejected)'
                          : 'var(--inconclusive)',
                      border: `1px solid ${
                        h.status === 'ACCEPTED'
                          ? 'var(--accepted)'
                          : h.status === 'REJECTED'
                          ? 'var(--rejected)'
                          : 'var(--inconclusive)'
                      }`,
                      padding: '4px 10px',
                      borderRadius: 12,
                      fontSize: 12,
                      fontWeight: 700,
                    }}
                  >
                    {h.status}
                  </span>
                </div>

                <p style={{ fontSize: 14, color: 'var(--text)', marginBottom: 8 }}>
                  {h.statement}
                </p>

                <div
                  style={{
                    background: 'var(--bg)',
                    borderLeft: '3px solid var(--accent)',
                    padding: '8px 12px',
                    borderRadius: 4,
                    fontSize: 13,
                    marginBottom: 12,
                  }}
                >
                  <div>
                    <strong>H0 (Null):</strong> {h.null_hypothesis}
                  </div>
                  <div>
                    <strong>H1 (Alt):</strong> {h.alt_hypothesis}
                  </div>
                </div>

                <div
                  style={{
                    display: 'flex',
                    gap: 16,
                    fontSize: 13,
                    color: 'var(--muted)',
                    flexWrap: 'wrap',
                  }}
                >
                  <div>
                    Metric: <strong>{h.metric_name || 'N/A'}: {h.metric_value ?? '--'}</strong>
                  </div>
                  <div>
                    p-Value: <strong>{h.p_value ?? '--'}</strong>
                  </div>
                </div>

                {h.summary_findings && (
                  <p style={{ fontSize: 13, color: 'var(--text)', marginTop: 10 }}>
                    <strong>Findings:</strong> {h.summary_findings}
                  </p>
                )}

                <div style={{ display: 'flex', gap: 10, marginTop: 14 }}>
                  {h.pdf_path && (
                    <a
                      href={`/api/pdf?path=${encodeURIComponent(h.pdf_path)}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      style={{
                        background: 'var(--surface-hover)',
                        border: '1px solid var(--border)',
                        color: 'var(--accent)',
                        padding: '6px 12px',
                        borderRadius: 6,
                        textDecoration: 'none',
                        fontSize: 13,
                        fontWeight: 500,
                      }}
                    >
                      📄 Open Manuscript PDF
                    </a>
                  )}
                  {h.report_dir && (
                    <span
                      style={{
                        background: 'var(--surface-hover)',
                        border: '1px solid var(--border)',
                        color: 'var(--muted)',
                        padding: '6px 12px',
                        borderRadius: 6,
                        fontSize: 13,
                      }}
                    >
                      📁 {h.report_dir}
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* TAB 2: DATASETS */}
      {activeTab === 'datasets' && (
        <div>
          <h2 style={{ fontSize: 18, color: 'var(--text-heading)', marginBottom: 16 }}>
            Vietnamese Quantitative Data Lake (36 Parquet Datasets)
          </h2>
          <div
            style={{
              background: 'var(--surface)',
              borderRadius: 8,
              border: '1px solid var(--border)',
              overflow: 'hidden',
            }}
          >
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 14 }}>
              <thead>
                <tr style={{ background: 'var(--surface-hover)', borderBottom: '1px solid var(--border)' }}>
                  <th style={{ textAlign: 'left', padding: '12px 16px' }}>Category</th>
                  <th style={{ textAlign: 'left', padding: '12px 16px' }}>Filename</th>
                  <th style={{ textAlign: 'left', padding: '12px 16px' }}>Size</th>
                  <th style={{ textAlign: 'left', padding: '12px 16px' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {datasets.map((d, i) => (
                  <tr
                    key={i}
                    style={{ borderBottom: '1px solid var(--border)' }}
                  >
                    <td style={{ padding: '12px 16px' }}>
                      <span
                        style={{
                          background: 'var(--surface-hover)',
                          padding: '2px 8px',
                          borderRadius: 4,
                          fontSize: 12,
                        }}
                      >
                        {d.category}
                      </span>
                    </td>
                    <td style={{ padding: '12px 16px', fontWeight: 600 }}>{d.filename}</td>
                    <td style={{ padding: '12px 16px', color: 'var(--muted)' }}>
                      {d.sizeKb.toLocaleString()} KB
                    </td>
                    <td style={{ padding: '12px 16px' }}>
                      <button
                        onClick={() => loadPreview(d.relativePath)}
                        style={{
                          background: 'var(--accent-bg)',
                          border: '1px solid var(--accent)',
                          color: 'var(--accent)',
                          padding: '4px 10px',
                          borderRadius: 6,
                          fontSize: 13,
                          cursor: 'pointer',
                          fontWeight: 600,
                        }}
                      >
                        Preview via Arquero (JS)
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* TAB 3: REPORTS */}
      {activeTab === 'reports' && (
        <div>
          <h2 style={{ fontSize: 18, color: 'var(--text-heading)', marginBottom: 16 }}>
            Compiled Empirical Research Manuscripts
          </h2>
          <div
            style={{
              background: 'var(--surface)',
              borderRadius: 8,
              border: '1px solid var(--border)',
              overflow: 'hidden',
            }}
          >
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 14 }}>
              <thead>
                <tr style={{ background: 'var(--surface-hover)', borderBottom: '1px solid var(--border)' }}>
                  <th style={{ textAlign: 'left', padding: '12px 16px' }}>Study ID</th>
                  <th style={{ textAlign: 'left', padding: '12px 16px' }}>Title</th>
                  <th style={{ textAlign: 'left', padding: '12px 16px' }}>Size</th>
                  <th style={{ textAlign: 'left', padding: '12px 16px' }}>PDF Document</th>
                </tr>
              </thead>
              <tbody>
                {reports.map((r, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
                    <td style={{ padding: '12px 16px' }}>
                      <code style={{ color: 'var(--accent)' }}>{r.id}</code>
                    </td>
                    <td style={{ padding: '12px 16px', fontWeight: 600 }}>{r.title}</td>
                    <td style={{ padding: '12px 16px', color: 'var(--muted)' }}>
                      {r.sizeKb.toLocaleString()} KB
                    </td>
                    <td style={{ padding: '12px 16px' }}>
                      <a
                        href={`/api/pdf?path=${encodeURIComponent(r.pdfPath)}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        style={{
                          background: 'var(--surface-hover)',
                          border: '1px solid var(--border)',
                          color: 'var(--accent)',
                          padding: '6px 12px',
                          borderRadius: 6,
                          textDecoration: 'none',
                          fontSize: 13,
                          fontWeight: 500,
                        }}
                      >
                        📄 View PDF Manuscript
                      </a>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* PREVIEW MODAL (Arquero JS DataFrame Preview) */}
      {previewData && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            background: 'rgba(0, 0, 0, 0.8)',
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            zIndex: 1000,
          }}
        >
          <div
            style={{
              background: 'var(--surface)',
              border: '1px solid var(--border)',
              borderRadius: 10,
              width: '92%',
              maxWidth: 1000,
              maxHeight: '85vh',
              display: 'flex',
              flexDirection: 'column',
              overflow: 'hidden',
            }}
          >
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                padding: '16px 20px',
                borderBottom: '1px solid var(--border)',
              }}
            >
              <div>
                <h3 style={{ fontSize: 16, color: 'var(--text-heading)' }}>
                  {previewData.filename}
                </h3>
                {previewData.totalRows && (
                  <p style={{ fontSize: 12, color: 'var(--muted)' }}>
                    Processed via Arquero (JS Dataframe): {previewData.totalRows.toLocaleString()}{' '}
                    total rows • Showing top 20 rows
                  </p>
                )}
              </div>
              <button
                onClick={() => setPreviewData(null)}
                style={{
                  background: 'none',
                  border: 'none',
                  color: 'var(--muted)',
                  fontSize: 22,
                  cursor: 'pointer',
                }}
              >
                &times;
              </button>
            </div>

            <div style={{ padding: 20, overflow: 'auto', flex: 1 }}>
              {loading ? (
                <div style={{ color: 'var(--muted)' }}>Parsing Parquet with Hyparquet + Arquero...</div>
              ) : previewData.error ? (
                <div style={{ color: 'var(--rejected)' }}>Error: {previewData.error}</div>
              ) : (
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ background: 'var(--surface-hover)' }}>
                      {previewData.columns?.map((c) => (
                        <th
                          key={c}
                          style={{
                            padding: '8px 12px',
                            textAlign: 'left',
                            borderBottom: '1px solid var(--border)',
                            color: 'var(--text-heading)',
                          }}
                        >
                          {c}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {previewData.sampleRows?.map((row, rIdx) => (
                      <tr key={rIdx} style={{ borderBottom: '1px solid var(--border)' }}>
                        {previewData.columns?.map((c) => (
                          <td key={c} style={{ padding: '8px 12px' }}>
                            {row[c] !== null && row[c] !== undefined ? (
                              String(row[c])
                            ) : (
                              <span style={{ color: 'var(--muted)' }}>null</span>
                            )}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

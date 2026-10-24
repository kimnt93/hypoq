export const dynamic = 'force-dynamic';

import { NextRequest, NextResponse } from 'next/server';
import { DatabaseSync } from 'node:sqlite';
import fs from 'node:fs';
import path from 'node:path';
import { getProjectRoot } from '../../lib/paths';

export interface HypothesisRecord {
  id: number;
  session_id: string | null;
  agent_name: string | null;
  title: string;
  statement: string;
  null_hypothesis: string;
  alt_hypothesis: string;
  domain: string;
  dataset_dependencies: string | null;
  methodology: string | null;
  metric_name: string | null;
  metric_value: number | null;
  p_value: number | null;
  confidence_level: number;
  status: string;
  summary_findings: string | null;
  report_dir: string | null;
  pdf_path: string | null;
  created_at: string;
  updated_at: string;
  rank_score?: number;
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const status = searchParams.get('status');
  const query = searchParams.get('q');

  const root = getProjectRoot();
  const dbPath = process.env.AUTORESEARCH_DB && fs.existsSync(process.env.AUTORESEARCH_DB)
    ? process.env.AUTORESEARCH_DB
    : path.join(root, 'autoresearch.db');

  if (!fs.existsSync(dbPath)) {
    return NextResponse.json({ error: `Database not found at ${dbPath}` }, { status: 404 });
  }

  try {
    const db = new DatabaseSync(dbPath);
    let rows: HypothesisRecord[] = [];

    if (query && query.trim().length > 2) {
      const rawTokens = query.trim().match(/[A-Za-z0-9_]{3,}/g) || [];
      if (rawTokens.length > 0) {
        const ftsQuery = rawTokens.map(t => `"${t}"`).join(' OR ');
        const sql = `
          SELECT h.*, bm25(hypotheses_fts) as rank_score
          FROM hypotheses_fts
          JOIN hypotheses h ON hypotheses_fts.rowid = h.id
          WHERE hypotheses_fts MATCH ?
          ORDER BY rank_score ASC
          LIMIT 50;
        `;
        const stmt = db.prepare(sql);
        rows = stmt.all(ftsQuery) as unknown as HypothesisRecord[];
        if (status) {
          rows = rows.filter(r => r.status.toUpperCase() === status.toUpperCase());
        }
      }
    } else {
      let sql = `
        SELECT h.*, (e.hypothesis_id IS NOT NULL) AS has_embedding, e.model_name
        FROM hypotheses h
        LEFT JOIN hypothesis_embeddings e ON h.id = e.hypothesis_id
      `;
      const params: string[] = [];
      if (status) {
        sql += ' WHERE UPPER(h.status) = ?';
        params.push(status.toUpperCase());
      }
      sql += ' ORDER BY h.id DESC LIMIT 100';
      const stmt = db.prepare(sql);
      rows = stmt.all(...params) as unknown as HypothesisRecord[];
    }

    return NextResponse.json(rows);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

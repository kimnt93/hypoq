import { NextResponse } from 'next/server';
import fs from 'node:fs';
import path from 'node:path';
import { getProjectRoot } from '../../lib/paths';

export interface DatasetItem {
  filename: string;
  category: string;
  relativePath: string;
  sizeKb: number;
  updatedAt: string;
}

export async function GET() {
  const root = getProjectRoot();
  const dataRoot = path.join(root, 'data');

  if (!fs.existsSync(dataRoot)) {
    return NextResponse.json({ error: `Data folder not found at ${dataRoot}` }, { status: 404 });
  }

  const results: DatasetItem[] = [];
  const categories = fs.readdirSync(dataRoot).filter(f => {
    return fs.statSync(path.join(dataRoot, f)).isDirectory();
  });

  for (const cat of categories) {
    const catDir = path.join(dataRoot, cat);
    const files = fs.readdirSync(catDir).filter(f => f.endsWith('.parquet'));
    for (const f of files) {
      const fullPath = path.join(catDir, f);
      const stat = fs.statSync(fullPath);
      results.push({
        filename: f,
        category: cat,
        relativePath: `data/${cat}/${f}`,
        sizeKb: Math.round(stat.size / 1024),
        updatedAt: stat.mtime.toISOString().substring(0, 10)
      });
    }
  }

  return NextResponse.json(results);
}

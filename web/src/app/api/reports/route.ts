export const dynamic = 'force-dynamic';

import { NextResponse } from 'next/server';
import fs from 'node:fs';
import path from 'node:path';
import { getProjectRoot } from '../../lib/paths';

export interface ReportItem {
  id: string;
  title: string;
  pdfPath: string;
  sizeKb: number;
  updatedAt: string;
}

export async function GET() {
  const root = getProjectRoot();
  const reports: ReportItem[] = [];
  const searchDirs = [
    path.join(root, 'researches'),
    path.join(root, 'templates'),
    path.join(root, '..', '015_tightly_held_shares_basis_dynamics')
  ];

  for (const sDir of searchDirs) {
    if (!fs.existsSync(sDir)) continue;
    if (fs.existsSync(path.join(sDir, 'main.pdf'))) {
      const stat = fs.statSync(path.join(sDir, 'main.pdf'));
      reports.push({
        id: path.basename(sDir),
        title: path.basename(sDir).replace(/_/g, ' ').toUpperCase(),
        pdfPath: path.relative(root, path.join(sDir, 'main.pdf')),
        sizeKb: Math.round(stat.size / 1024),
        updatedAt: stat.mtime.toISOString().substring(0, 10)
      });
      continue;
    }

    const subdirs = fs.readdirSync(sDir).filter(f => {
      try {
        return fs.statSync(path.join(sDir, f)).isDirectory();
      } catch {
        return false;
      }
    });

    for (const sub of subdirs) {
      const pdfFile = path.join(sDir, sub, 'main.pdf');
      if (fs.existsSync(pdfFile)) {
        const stat = fs.statSync(pdfFile);
        let title = sub.replace(/_/g, ' ').toUpperCase();
        const readmeFile = path.join(sDir, sub, 'README.md');
        if (fs.existsSync(readmeFile)) {
          const content = fs.readFileSync(readmeFile, 'utf8');
          const firstLine = content.split('\n')[0].replace(/^#\s*/, '').trim();
          if (firstLine) title = firstLine;
        }

        reports.push({
          id: sub,
          title,
          pdfPath: path.relative(root, pdfFile),
          sizeKb: Math.round(stat.size / 1024),
          updatedAt: stat.mtime.toISOString().substring(0, 10)
        });
      }
    }
  }

  return NextResponse.json(reports);
}

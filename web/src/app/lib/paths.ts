import path from 'node:path';
import fs from 'node:fs';

export function getProjectRoot(): string {
  const candidates = [
    process.cwd(),
    path.resolve(process.cwd(), '..'),
    path.resolve(process.cwd(), '..', '..'),
    path.resolve(process.cwd(), '..', '..', '..'),
  ];
  for (const c of candidates) {
    if (fs.existsSync(path.join(c, 'data')) && fs.existsSync(path.join(c, 'autoresearch.db'))) {
      return c;
    }
  }
  return path.resolve(process.cwd(), '..');
}

export function getDbPath(): string {
  if (process.env.AUTORESEARCH_DB && fs.existsSync(process.env.AUTORESEARCH_DB)) {
    return process.env.AUTORESEARCH_DB;
  }
  const root = getProjectRoot();
  return path.join(root, 'autoresearch.db');
}

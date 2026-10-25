export const dynamic = 'force-dynamic';

import { NextRequest, NextResponse } from 'next/server';
import fs from 'node:fs';
import path from 'node:path';
import { parquetRead } from 'hyparquet';
import { compressors } from 'hyparquet-compressors';
import * as aq from 'arquero';
import { getProjectRoot } from '../../lib/paths';

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const relPath = searchParams.get('path');

  if (!relPath) {
    return NextResponse.json({ error: 'Missing path parameter' }, { status: 400 });
  }

  const allowedRoot = path.resolve(getProjectRoot(), 'data');
  const absPath = path.resolve(getProjectRoot(), relPath);

  if (
    !absPath.startsWith(allowedRoot + path.sep) ||
    !absPath.toLowerCase().endsWith('.parquet') ||
    !fs.existsSync(absPath)
  ) {
    return NextResponse.json({ error: 'File not found or outside root' }, { status: 404 });
  }

  try {
    const nodeBuf = fs.readFileSync(absPath);
    const arrayBuffer = nodeBuf.buffer.slice(nodeBuf.byteOffset, nodeBuf.byteOffset + nodeBuf.byteLength);

    const rows = await new Promise<Record<string, unknown>[]>((resolve, reject) => {
      try {
        parquetRead({
          file: arrayBuffer,
          compressors,
          rowFormat: 'object',
          onComplete: (data: unknown) => resolve(data as Record<string, unknown>[])
        });
      } catch (err) {
        reject(err);
      }
    });

    // Process via Arquero (JS pandas alternative)
    const dt = aq.from(rows);
    const totalRows = dt.numRows();
    const columns = dt.columnNames();
    const rawSample = dt.slice(0, 20).objects();
    // Convert BigInt to Number or String for JSON compatibility
    const sampleRows = rawSample.map(row => {
      const cleanRow: Record<string, unknown> = {};
      for (const [key, val] of Object.entries(row)) {
        if (typeof val === 'bigint') {
          cleanRow[key] = Number(val);
        } else {
          cleanRow[key] = val;
        }
      }
      return cleanRow;
    });

    return NextResponse.json({
      filename: path.basename(absPath),
      totalRows,
      columns,
      sampleRows
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

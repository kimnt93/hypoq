import { NextRequest, NextResponse } from 'next/server';
import fs from 'node:fs';
import path from 'node:path';
import { getProjectRoot } from '../../lib/paths';

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const relPath = searchParams.get('path');

  if (!relPath) {
    return NextResponse.json({ error: 'Missing path parameter' }, { status: 400 });
  }

  const root = getProjectRoot();
  let absPath = path.resolve(root, relPath);
  if (!fs.existsSync(absPath)) {
    absPath = path.resolve(root, '..', relPath);
  }

  if (!fs.existsSync(absPath)) {
    return NextResponse.json({ error: 'PDF file not found' }, { status: 404 });
  }

  try {
    const fileBuffer = fs.readFileSync(absPath);
    return new NextResponse(fileBuffer, {
      headers: {
        'Content-Type': 'application/pdf',
        'Content-Disposition': `inline; filename="${path.basename(absPath)}"`,
        'Content-Length': fileBuffer.length.toString()
      }
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

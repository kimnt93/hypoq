import { NextResponse } from 'next/server';

export async function GET() {
  return NextResponse.json({
    status: 'ok',
    service: 'vnresearch-web',
    timestamp: new Date().toISOString()
  });
}

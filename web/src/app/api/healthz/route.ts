import { NextResponse } from 'next/server';

export async function GET() {
  return NextResponse.json({
    status: 'ok',
    service: 'ai-autoresearch-lab',
    timestamp: new Date().toISOString()
  });
}

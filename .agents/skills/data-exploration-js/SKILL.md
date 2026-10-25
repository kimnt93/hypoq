---
name: data-exploration-js
description: Explore, query, and transform Parquet datasets and SQLite databases directly in JavaScript/TypeScript using Arquero (JS pandas alternative), Hyparquet, and node:sqlite.
version: 1.0.0
author: hypoq
tags: [javascript, node, arquero, hyparquet, sqlite, dataframe, fullstack]
---

# JavaScript Data Exploration Skill (Arquero + Hyparquet + node:sqlite)

## Overview
This skill provides routines for querying Parquet files and SQLite databases in JavaScript and TypeScript environments (Next.js server routes, Node.js scripts, or Bun). It utilizes **Arquero** as the JavaScript alternative to Pandas/Polars, **Hyparquet** for pure-JS zero-native Parquet decoding, and `node:sqlite` for native database queries.

## When to Use
- Implementing Next.js API endpoints (`app/api/dataset-preview/route.ts` or `app/api/hypotheses/route.ts`).
- Transforming, slicing, or aggregating Parquet datasets on the Node/Next.js backend without invoking external Python processes.
- Querying `hypoq.db` inside fullstack JavaScript applications.

## Core Libraries
- **Arquero** (`import * as aq from 'arquero'`): High-performance table/dataframe operations (filter, derive, groupby, rollup, sample, order) modeled after dplyr and pandas.
- **Hyparquet** (`import { parquetRead } from 'hyparquet'`): Fast parser for Parquet buffers with `hyparquet-compressors` (ZSTD, Snappy).
- **Node SQLite** (`import { DatabaseSync } from 'node:sqlite'`): Native synchronous SQLite client built into Node 22+.

## Implementation Recipes

### Recipe 1: Reading ZSTD Parquet into an Arquero DataFrame
```javascript
const fs = require('fs');
const { parquetRead } = require('hyparquet');
const { compressors } = require('hyparquet-compressors');
const aq = require('arquero');

function loadParquetAsArquero(filePath) {
  return new Promise((resolve, reject) => {
    try {
      const buffer = fs.readFileSync(filePath);
      const arrayBuffer = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);

      parquetRead({
        file: arrayBuffer,
        compressors,
        rowFormat: 'object',
        onComplete: (rows) => {
          const dt = aq.from(rows);
          resolve(dt);
        }
      });
    } catch (err) {
      reject(err);
    }
  });
}

// Example usage:
// const dt = await loadParquetAsArquero("data/market/vn_indices_daily.parquet");
// dt.filter(d => d.symbol === "VNINDEX")
//   .orderby(aq.desc("date"))
//   .slice(0, 10)
//   .print();
```

### Recipe 2: Grouping and Summary Statistics in Arquero
```javascript
async function computeSectorSummary(dt) {
  const summary = dt
    .groupby('sector_name')
    .rollup({
      count: aq.op.count(),
      avg_close: aq.op.mean('close'),
      total_vol: aq.op.sum('volume')
    })
    .orderby(aq.desc('total_vol'));

  return summary.objects();
}
```

### Recipe 3: Querying SQLite Database via `node:sqlite`
```javascript
const { DatabaseSync } = require('node:sqlite');

function getHypotheses(dbPath = 'hypoq.db', status = null) {
  const db = new DatabaseSync(dbPath);
  let queryStr = 'SELECT id, title, statement, domain, metric_name, metric_value, p_value, status, pdf_path FROM hypotheses';
  const params = [];

  if (status) {
    queryStr += ' WHERE status = ?';
    params.push(status.toUpperCase());
  }

  queryStr += ' ORDER BY id DESC';
  const stmt = db.prepare(queryStr);
  return stmt.all(...params);
}
```

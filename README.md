# IR 2025 Paper Scraper

A two-stage tool to collect and download open-access research papers from top Information Retrieval venues published in 2025.

**Venues covered:** SIGIR · WSDM · CIKM · WWW · ACL · NeurIPS · ICLR

## How It Works

```
Stage 1 (collect.py)          Stage 2 (download.py)
─────────────────────         ─────────────────────
OpenAlex  → SIGIR             papers_metadata.json
           WSDM          →         ↓
           CIKM               parallel download
           WWW                (5 threads, resumable)
ACL Anthology → ACL           ↓
OpenReview → ICLR        papers_2025/{VENUE}/*.pdf
             NeurIPS
```

## Data Sources

| Venue | Source | Rate Limit |
|-------|--------|-----------|
| SIGIR, WSDM, CIKM, WWW | [OpenAlex API](https://openalex.org) | 10 req/s |
| ACL | [ACL Anthology](https://aclanthology.org) | polite 1 req/s |
| ICLR, NeurIPS | [OpenReview API](https://openreview.net) | ~1 req/s |

All sources are free, open, and do **not** require API keys.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Stage 1 — Collect Metadata

```bash
# Default: 15 papers per venue (105 total)
python3 collect.py

# Custom target
python3 collect.py --target 20

# Specific venues only
python3 collect.py --venues SIGIR ACL ICLR
```

Output:
- `papers_metadata.json` — structured paper metadata
- `papers_metadata.csv`  — human-readable spreadsheet

### Stage 2 — Download PDFs

```bash
# Default: 5 parallel threads
python3 download.py

# More threads (for fast cloud connections)
python3 download.py --workers 10

# Download specific venues only
python3 download.py --venues SIGIR ACL

# Limit papers per venue
python3 download.py --limit 5
```

Output:
```
papers_2025/
├── SIGIR/    001_Paper_Title.pdf ...
├── WSDM/     001_Paper_Title.pdf ...
├── CIKM/     ...
├── WWW/      ...
├── ACL/      ...
├── NeurIPS/  ...
└── ICLR/     ...
download_report.json
```

## Features

- **Resumable** — re-running `download.py` skips already-downloaded files
- **Retry logic** — 3× retries with exponential backoff per file
- **PDF validation** — checks magic bytes `%PDF` and minimum file size
- **IR keyword filter** — only collects retrieval-relevant papers

## IR Keywords Filter

Papers are included if their title or abstract contains any of:

> retrieval, ranking, reranking, dense retrieval, sparse retrieval, RAG,
> retrieval-augmented, passage retrieval, document retrieval, bi-encoder,
> cross-encoder, BM25, semantic search, query expansion, question answering,
> vector search, embedding retrieval, open-domain QA, ...

## Estimated Runtime

| Stage | Duration |
|-------|----------|
| Stage 1 (collect) | ~2–3 min |
| Stage 2 (download, 105 papers, 5 threads) | ~5–8 min |
| **Total** | **~10 min** |

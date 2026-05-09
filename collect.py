"""
Stage 1 — Kumpul metadata paper IR 2025 dari semua venue.

Sumber:
  - OpenAlex  → SIGIR, WSDM, CIKM, WWW
  - ACL Anthology → ACL
  - OpenReview    → ICLR, NeurIPS

Output:
  papers_metadata.json  ← input untuk download.py
  papers_metadata.csv   ← untuk inspeksi manual

Usage:
  python3 collect.py
  python3 collect.py --target 20        # 20 paper per venue
  python3 collect.py --venues SIGIR ACL # venue tertentu saja
"""

import argparse
import json
import re
import time
import sys
from pathlib import Path

import requests
import pandas as pd

# ─── CONFIG ──────────────────────────────────────────────────────────────────

DEFAULT_TARGET   = 15          # paper per venue
YEAR             = 2025
OUT_JSON         = Path("papers_metadata.json")
OUT_CSV          = Path("papers_metadata.csv")
HEADERS          = {"User-Agent": "IR-course-collector/1.0 (academic research)"}

IR_KEYWORDS = [
    "retrieval", "search engine", "ranking", "reranking", "rerank",
    "indexing", "dense retrieval", "sparse retrieval", "rag",
    "retrieval-augmented", "retrieval augmented",
    "question answering", "passage retrieval", "document retrieval",
    "neural retrieval", "bi-encoder", "cross-encoder",
    "query expansion", "bm25", "semantic search",
    "knowledge retrieval", "open-domain qa", "ad-hoc retrieval",
    "embedding retrieval", "vector search", "approximate nearest neighbor",
]

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def is_ir_relevant(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in IR_KEYWORDS)


def get(url, params=None, retries=3, wait=2.0):
    """HTTP GET dengan retry."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=25)
            r.raise_for_status()
            return r
        except requests.HTTPError as e:
            code = e.response.status_code if e.response else 0
            wait_t = wait * (2 ** attempt)
            print(f"  [HTTP {code}] attempt {attempt+1}/{retries} — tunggu {wait_t:.0f}s", flush=True)
            if code == 429:
                wait_t = max(wait_t, 60)
            if attempt == retries - 1:
                print(f"  [FAIL] {url}")
                return None
            time.sleep(wait_t)
        except Exception as e:
            if attempt == retries - 1:
                print(f"  [ERROR] {type(e).__name__}: {e}")
                return None
            time.sleep(wait * (attempt + 1))
    return None


def reconstruct_abstract(inverted_index: dict) -> str:
    """OpenAlex menyimpan abstract sebagai inverted index; rekonstruksi jadi teks."""
    if not inverted_index:
        return ""
    positions = {}
    for word, pos_list in inverted_index.items():
        for pos in pos_list:
            positions[pos] = word
    return " ".join(positions[i] for i in sorted(positions))


# ─── SOURCE 1: OPENALEX ──────────────────────────────────────────────────────

OPENALEX_VENUE_NAMES = {
    "SIGIR": "SIGIR",
    "WSDM":  "WSDM",
    "CIKM":  "CIKM",
    "WWW":   "The Web Conference",
}


def collect_openalex(venue_key: str, target: int) -> list[dict]:
    """Ambil paper dari OpenAlex untuk satu venue."""
    venue_name = OPENALEX_VENUE_NAMES[venue_key]
    print(f"  [OpenAlex] venue='{venue_name}' year={YEAR}", flush=True)

    results = []
    cursor = "*"
    page = 0

    while len(results) < target * 3:   # ambil lebih, filter belakangan
        params = {
            "filter": (
                f"primary_location.source.display_name.search:{venue_name},"
                f"publication_year:{YEAR},"
                "open_access.is_oa:true"
            ),
            "select": "title,authorships,publication_year,doi,open_access,abstract_inverted_index,primary_location",
            "per-page": 50,
            "cursor": cursor,
            "mailto": "ir-course@example.com",  # OpenAlex minta email di User-Agent/params
        }
        resp = get("https://api.openalex.org/works", params=params)
        if not resp:
            break

        data = resp.json()
        works = data.get("results", [])
        if not works:
            break

        for w in works:
            title    = w.get("title") or ""
            abstract = reconstruct_abstract(w.get("abstract_inverted_index") or {})
            if not is_ir_relevant(title + " " + abstract):
                continue
            authors = [
                a["author"]["display_name"]
                for a in (w.get("authorships") or [])
                if a.get("author")
            ]
            oa      = w.get("open_access") or {}
            pdf_url = oa.get("oa_url") or ""
            results.append({
                "venue":    venue_key,
                "title":    title.strip(),
                "authors":  authors,
                "year":     YEAR,
                "pdf_url":  pdf_url,
                "source":   "openalex",
                "doi":      w.get("doi") or "",
                "abstract": abstract,
            })

        meta   = data.get("meta", {})
        cursor = meta.get("next_cursor")
        page  += 1
        print(f"    halaman {page}: {len(works)} hasil, lolos filter: {len(results)}", flush=True)

        if not cursor or len(results) >= target * 3:
            break
        time.sleep(0.15)   # ~7 req/s — masih di bawah limit 10/s

    # Filter yang tidak punya pdf_url
    has_pdf = [r for r in results if r["pdf_url"]]
    print(f"  → {len(results)} relevan, {len(has_pdf)} punya PDF", flush=True)
    return has_pdf[:target]


# ─── SOURCE 2: ACL ANTHOLOGY ─────────────────────────────────────────────────

ACL_TRACKS = ["long", "short", "findings"]


def collect_acl(target: int) -> list[dict]:
    """Scrape ACL 2025 dari ACL Anthology."""
    print("  [ACL Anthology] scraping volumes ...", flush=True)
    results = []

    for track in ACL_TRACKS:
        if len(results) >= target * 2:
            break

        # Coba volume URL langsung
        vol_url = f"https://aclanthology.org/volumes/{YEAR}.acl-{track}/"
        resp = get(vol_url)
        if not resp:
            # Fallback: cari dari events page
            resp = get(f"https://aclanthology.org/events/acl-{YEAR}/")
            if not resp:
                continue

        html  = resp.text
        # Temukan semua paper ID dalam format 2025.acl-{track}.N
        pids  = list(dict.fromkeys(
            re.findall(rf"{YEAR}\.acl-{track}\.\d+", html)
        ))
        print(f"    track={track}: {len(pids)} paper ditemukan", flush=True)

        for pid in pids:
            if len(results) >= target * 2:
                break
            # Ambil judul dari HTML (cari anchor tag dengan href matching)
            pattern = rf'href="/{pid}"[^>]*>\s*([^<]+?)\s*</a>'
            m = re.search(pattern, html)
            title = m.group(1).strip() if m else pid

            if not is_ir_relevant(title):
                continue

            pdf_url = f"https://aclanthology.org/{pid}.pdf"
            results.append({
                "venue":    "ACL",
                "title":    title,
                "authors":  [],   # bisa di-enrich nanti
                "year":     YEAR,
                "pdf_url":  pdf_url,
                "source":   "acl_anthology",
                "doi":      "",
                "abstract": "",
            })

        time.sleep(1.0)

    print(f"  → {len(results)} paper ACL relevan IR", flush=True)
    return results[:target]


# ─── SOURCE 3: OPENREVIEW ────────────────────────────────────────────────────

OPENREVIEW_VENUES = {
    "ICLR": [
        "ICLR 2025 poster",
        "ICLR 2025 oral",
        "ICLR 2025 spotlight",
    ],
    "NeurIPS": [
        "NeurIPS 2025 poster",
        "NeurIPS 2025 oral",
        "NeurIPS 2025 spotlight",
    ],
}


def collect_openreview(venue_key: str, target: int) -> list[dict]:
    """Ambil paper dari OpenReview untuk ICLR / NeurIPS."""
    venue_labels = OPENREVIEW_VENUES[venue_key]
    results      = []
    base         = "https://api2.openreview.net/notes"

    for label in venue_labels:
        if len(results) >= target * 2:
            break
        print(f"  [OpenReview] venue='{label}'", flush=True)
        offset = 0
        while True:
            params = {"content.venue": label, "limit": 50, "offset": offset}
            resp   = get(base, params=params)
            if not resp:
                break

            notes = resp.json().get("notes", [])
            if not notes:
                break

            for note in notes:
                content  = note.get("content", {})
                def val(field):
                    v = content.get(field, {})
                    return v.get("value", "") if isinstance(v, dict) else str(v or "")

                title    = val("title")
                abstract = val("abstract")
                if not is_ir_relevant(title + " " + abstract):
                    continue

                paper_id = note.get("id", "")
                pdf_url  = f"https://openreview.net/pdf?id={paper_id}"
                authors_field = content.get("authors", {})
                authors_list  = (
                    authors_field.get("value", [])
                    if isinstance(authors_field, dict)
                    else (authors_field if isinstance(authors_field, list) else [])
                )
                results.append({
                    "venue":    venue_key,
                    "title":    title,
                    "authors":  authors_list,
                    "year":     YEAR,
                    "pdf_url":  pdf_url,
                    "source":   "openreview",
                    "doi":      "",
                    "abstract": abstract,
                })

            offset += len(notes)
            if len(notes) < 50 or len(results) >= target * 2:
                break
            time.sleep(1.0)

    print(f"  → {len(results)} paper {venue_key} relevan IR", flush=True)
    return results[:target]


# ─── DISPATCHER ──────────────────────────────────────────────────────────────

VENUE_HANDLERS = {
    "SIGIR":   lambda t: collect_openalex("SIGIR",   t),
    "WSDM":    lambda t: collect_openalex("WSDM",    t),
    "CIKM":    lambda t: collect_openalex("CIKM",    t),
    "WWW":     lambda t: collect_openalex("WWW",     t),
    "ACL":     lambda t: collect_acl(t),
    "NeurIPS": lambda t: collect_openreview("NeurIPS", t),
    "ICLR":    lambda t: collect_openreview("ICLR",    t),
}

ALL_VENUES = list(VENUE_HANDLERS.keys())


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 1: Kumpul metadata paper IR 2025")
    parser.add_argument("--target",  type=int,   default=DEFAULT_TARGET,
                        help=f"Paper per venue (default: {DEFAULT_TARGET})")
    parser.add_argument("--venues",  nargs="+",  default=ALL_VENUES,
                        choices=ALL_VENUES, metavar="VENUE",
                        help=f"Venue yang dikumpulkan (default: semua). Pilihan: {ALL_VENUES}")
    parser.add_argument("--out-json", default=str(OUT_JSON), help="Output JSON path")
    parser.add_argument("--out-csv",  default=str(OUT_CSV),  help="Output CSV path")
    args = parser.parse_args()

    target   = args.target
    venues   = args.venues
    out_json = Path(args.out_json)
    out_csv  = Path(args.out_csv)

    print(f"Target: {target} paper/venue × {len(venues)} venue = {target * len(venues)} paper")
    print(f"Venues: {venues}\n")

    all_papers = []

    for venue_key in venues:
        print(f"\n{'='*60}")
        print(f"  {venue_key}")
        print(f"{'='*60}")
        handler = VENUE_HANDLERS[venue_key]
        papers  = handler(target)
        all_papers.extend(papers)
        print(f"  Berhasil: {len(papers)} paper dari {venue_key}")
        time.sleep(1.5)

    # Simpan JSON
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_papers, f, ensure_ascii=False, indent=2)

    # Simpan CSV
    if all_papers:
        df = pd.DataFrame(all_papers)
        df["authors"] = df["authors"].apply(lambda a: "; ".join(a) if isinstance(a, list) else a)
        df.to_csv(out_csv, index=False, encoding="utf-8")
    else:
        pd.DataFrame(columns=["venue","title","authors","year","pdf_url","source","doi","abstract"]
                     ).to_csv(out_csv, index=False, encoding="utf-8")

    # Ringkasan
    print(f"\n{'='*60}")
    print("RINGKASAN")
    print(f"{'='*60}")
    by_venue = {}
    for p in all_papers:
        by_venue.setdefault(p["venue"], []).append(p)

    for vk in venues:
        papers = by_venue.get(vk, [])
        has_pdf = sum(1 for p in papers if p.get("pdf_url"))
        print(f"  {vk:8s}: {len(papers):3d} paper  ({has_pdf} punya PDF)")

    total = len(all_papers)
    total_pdf = sum(1 for p in all_papers if p.get("pdf_url"))
    print(f"\nTotal: {total} paper, {total_pdf} siap download")
    print(f"Metadata : {out_json}")
    print(f"CSV      : {out_csv}")
    print("\nLangkah berikutnya:")
    print(f"  python3 download.py")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nWaktu collect: {time.time()-t0:.1f} detik")

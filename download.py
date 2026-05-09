"""
Stage 2 — Download PDF dari metadata yang dikumpulkan collect.py.

Fitur:
  - Resumable: skip file yang sudah ada
  - Parallel: ThreadPoolExecutor (configurable --workers)
  - Retry: 3x dengan exponential backoff
  - Validasi: cek magic bytes %PDF dan ukuran minimum
  - Laporan: download_report.json

Usage:
  python3 download.py
  python3 download.py --workers 8             # lebih banyak thread
  python3 download.py --input custom.json     # JSON lain
  python3 download.py --venues SIGIR ACL      # venue tertentu saja
  python3 download.py --limit 5              # max 5 paper per venue
"""

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# ─── CONFIG ──────────────────────────────────────────────────────────────────

DEFAULT_INPUT   = Path("papers_metadata.json")
DEFAULT_OUT_DIR = Path("papers_2025")
DEFAULT_WORKERS = 5
DEFAULT_REPORT  = Path("download_report.json")
MIN_PDF_BYTES   = 10_000

HEADERS = {"User-Agent": "IR-course-downloader/1.0 (academic research)"}

print_lock = threading.Lock()


def log(msg: str):
    with print_lock:
        print(msg, flush=True)


# ─── DOWNLOAD ────────────────────────────────────────────────────────────────

def safe_filename(title: str, idx: int) -> str:
    clean = "".join(c if c.isalnum() or c in " -_" else "" for c in title)
    slug  = clean.strip().replace(" ", "_")[:60]
    return f"{idx:03d}_{slug}.pdf"


def download_pdf(url: str, dest: Path, retries: int = 3) -> tuple[bool, str]:
    """
    Download satu PDF.
    Return (sukses: bool, pesan: str)
    """
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=45, stream=True)
            resp.raise_for_status()

            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)

            size = dest.stat().st_size
            if size < MIN_PDF_BYTES:
                dest.unlink(missing_ok=True)
                return False, f"terlalu kecil ({size} bytes)"

            # Validasi magic bytes %PDF
            with open(dest, "rb") as f:
                magic = f.read(4)
            if magic != b"%PDF":
                dest.unlink(missing_ok=True)
                return False, f"bukan PDF (magic={magic!r})"

            return True, f"{size // 1024} KB"

        except requests.HTTPError as e:
            code = e.response.status_code if e.response else 0
            wait = 3 * (2 ** attempt)
            if code == 429:
                wait = max(wait, 30)
            if attempt < retries - 1:
                time.sleep(wait)
            else:
                dest.unlink(missing_ok=True)
                return False, f"HTTP {code}"

        except Exception as e:
            dest.unlink(missing_ok=True)
            if attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
            else:
                return False, f"{type(e).__name__}: {e}"

    return False, "semua retry gagal"


# ─── TASK RUNNER ─────────────────────────────────────────────────────────────

def run_downloads(tasks: list[dict], workers: int) -> list[dict]:
    """
    Jalankan download secara paralel.
    Setiap task: {idx, venue, title, pdf_url, dest}
    Return list hasil: {... + ok, msg, size_kb}
    """
    results = []
    total   = len(tasks)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        future_map = {
            ex.submit(download_pdf, t["pdf_url"], Path(t["dest"])): t
            for t in tasks
        }

        done = 0
        for future in as_completed(future_map):
            task = future_map[future]
            ok, msg = future.result()
            done += 1

            venue  = task["venue"]
            title  = task["title"][:55]
            marker = "✓" if ok else "✗"
            log(f"  [{done:3d}/{total}] {marker} [{venue}] {title}")
            if not ok:
                log(f"           Gagal: {msg}")

            results.append({**task, "ok": ok, "msg": msg})

    return results


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 2: Download PDF paper IR 2025")
    parser.add_argument("--input",   default=str(DEFAULT_INPUT),
                        help=f"File metadata JSON (default: {DEFAULT_INPUT})")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                        help=f"Folder output (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--report",  default=str(DEFAULT_REPORT),
                        help=f"File laporan JSON (default: {DEFAULT_REPORT})")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Jumlah thread paralel (default: {DEFAULT_WORKERS})")
    parser.add_argument("--venues",  nargs="+", default=None,
                        help="Filter venue tertentu (default: semua)")
    parser.add_argument("--limit",   type=int, default=None,
                        help="Batas paper per venue (default: semua)")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir    = Path(args.out_dir)
    report_path = Path(args.report)

    # Baca metadata
    if not input_path.exists():
        print(f"[ERROR] File tidak ditemukan: {input_path}")
        print("Jalankan collect.py terlebih dahulu.")
        return

    with open(input_path, encoding="utf-8") as f:
        all_papers = json.load(f)

    # Filter venue
    if args.venues:
        all_papers = [p for p in all_papers if p["venue"] in args.venues]

    # Filter yang punya PDF URL
    all_papers = [p for p in all_papers if p.get("pdf_url")]

    # Grup per venue, terapkan limit
    by_venue: dict[str, list] = {}
    for p in all_papers:
        by_venue.setdefault(p["venue"], []).append(p)

    if args.limit:
        for vk in by_venue:
            by_venue[vk] = by_venue[vk][: args.limit]

    # Susun task — skip yang sudah ada
    tasks    = []
    skipped  = 0
    counters: dict[str, int] = {}

    for venue_key, papers in by_venue.items():
        counters[venue_key] = 0
        for paper in papers:
            counters[venue_key] += 1
            idx      = counters[venue_key]
            filename = safe_filename(paper["title"], idx)
            dest     = out_dir / venue_key / filename

            if dest.exists() and dest.stat().st_size >= MIN_PDF_BYTES:
                skipped += 1
                continue

            tasks.append({
                "idx":     idx,
                "venue":   venue_key,
                "title":   paper["title"],
                "pdf_url": paper["pdf_url"],
                "dest":    str(dest),
                "doi":     paper.get("doi", ""),
                "source":  paper.get("source", ""),
            })

    total_planned = sum(len(v) for v in by_venue.values())
    print(f"Metadata  : {len(all_papers)} paper")
    print(f"Akan diunduh: {len(tasks)} | Sudah ada (skip): {skipped}")
    print(f"Workers   : {args.workers}")
    print(f"Output    : {out_dir.resolve()}\n")

    if not tasks:
        print("Tidak ada yang perlu diunduh.")
        return

    # Jalankan download
    t0      = time.time()
    results = run_downloads(tasks, workers=args.workers)
    elapsed = time.time() - t0

    # Hitung statistik
    ok_list   = [r for r in results if r["ok"]]
    fail_list = [r for r in results if not r["ok"]]

    by_venue_ok: dict[str, int] = {}
    for r in ok_list:
        by_venue_ok[r["venue"]] = by_venue_ok.get(r["venue"], 0) + 1

    # Simpan laporan
    report = {
        "total_attempted": len(tasks),
        "success": len(ok_list),
        "failed": len(fail_list),
        "skipped": skipped,
        "elapsed_seconds": round(elapsed, 1),
        "by_venue": {
            vk: {
                "attempted": len(papers),
                "success":   by_venue_ok.get(vk, 0),
            }
            for vk, papers in by_venue.items()
        },
        "failures": [
            {"venue": r["venue"], "title": r["title"],
             "pdf_url": r["pdf_url"], "error": r["msg"]}
            for r in fail_list
        ],
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Ringkasan akhir
    print(f"\n{'='*60}")
    print("RINGKASAN DOWNLOAD")
    print(f"{'='*60}")
    for vk, papers in by_venue.items():
        ok_n  = by_venue_ok.get(vk, 0)
        all_n = len(papers)
        bar   = "█" * ok_n + "░" * (all_n - ok_n)
        print(f"  {vk:8s}: {ok_n:3d}/{all_n} {bar}")

    print(f"\nBerhasil  : {len(ok_list)}/{len(tasks)}")
    print(f"Gagal     : {len(fail_list)}")
    print(f"Skip      : {skipped} (sudah ada)")
    print(f"Waktu     : {elapsed:.1f} detik ({elapsed/60:.1f} menit)")
    print(f"Laporan   : {report_path}")
    print(f"Folder PDF: {out_dir.resolve()}")

    if fail_list:
        print(f"\nGagal download ({len(fail_list)} paper):")
        for r in fail_list[:10]:
            print(f"  [{r['venue']}] {r['title'][:60]} — {r['msg']}")
        if len(fail_list) > 10:
            print(f"  ... (+{len(fail_list)-10} lagi, lihat {report_path})")


if __name__ == "__main__":
    main()

"""
Stage 3 — Analisis kelayakan paper IR 2025 menggunakan DeepSeek API.

Input  : all_papers.jsonl dari HuggingFace repo (stream via HTTP)
Output : analysis_partial_P{N}.jsonl  (checkpoint per partisi)
         analysis_results.csv / .json (hasil merge semua partisi)

Usage:
  # Test 5 paper dulu
  python3 analyze.py --test 5

  # Kaggle: partisi 0 dari 3 notebook paralel
  python3 analyze.py --partition 0 --num-partitions 3

  # Merge semua partisi → CSV final
  python3 analyze.py --merge
"""

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from openai import AsyncOpenAI
from tqdm import tqdm

# ─── CONFIG ──────────────────────────────────────────────────────────────────

DEEPSEEK_BASE_URL   = "https://api.deepseek.com"
DEEPSEEK_MODEL      = "deepseek-chat"
MAX_TEXT_CHARS      = 32_000
DEFAULT_CONCURRENCY = 10
DEFAULT_CHECKPOINT  = 50
PARTIAL_PREFIX      = "analysis_partial"
FINAL_CSV           = Path("analysis_results.csv")
FINAL_JSON          = Path("analysis_results.json")

HF_JSONL_URL = "https://huggingface.co/datasets/{dataset}/resolve/main/all_papers.jsonl"

# ─── PROMPT ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a research assistant helping undergraduate students at Universitas Indonesia \
select an Information Retrieval paper to re-implement and extend for their final project.
They have 5 weeks, a $20 compute budget, and are most interested in Indonesian / \
low-resource language applications. They lean toward small models, fine-tuning (not \
pretraining from scratch), and text-only methods.

You will be given the full text of a paper. Analyze it and respond ONLY with a single \
valid JSON object — no markdown fences, no extra text — matching the schema below.

Schema:
{
  "title": "string — paper title",
  "venue": "string — conference/journal name or 'not mentioned'",
  "year": "string — publication year or 'not mentioned'",
  "is_ir_relevant": "Yes | No",
  "specific_task": "string — e.g. dense retrieval, reranking, RAG, etc. or 'not mentioned'",
  "has_code_repo": "Yes | No",
  "code_url": "string — URL or 'not mentioned'",
  "datasets_used": ["list of dataset names"],
  "datasets_public": "Yes | No | Partial | Not mentioned",
  "model_size": "Small (<1B) | Medium (1B-7B) | Large (7B-70B) | Very Large (>70B) | Not mentioned",
  "training_type": "From scratch | Fine-tuning | Inference only | Not mentioned",
  "compute_cost_quoted": "string — exact quote from paper or 'not mentioned'",
  "limitations": ["list of limitations / future work the authors mention"],
  "applicable_indonesian": "Yes | No | Possibly",
  "applicable_indonesian_reason": "string — one sentence explanation",
  "extension_difficulty": "Easy | Medium | Hard",
  "extension_difficulty_reason": "string — one sentence explanation",
  "modality": "Text only | Text + Image | Other",
  "has_distillation": "Yes | No",
  "has_peft": "Yes | No",
  "has_synthetic_data": "Yes | No",
  "benchmarks_include_indonesian": "Yes | No | Not mentioned",
  "doability_score": 3,
  "extension_score": 4,
  "summary": "string — one sentence on why this paper is or is not a good fit for the team"
}

Rules:
- Only state what is explicitly written in the paper. If something is not mentioned, use "not mentioned".
- doability_score: 1=nearly impossible to reimplement in 5 weeks; 5=trivially reproducible.
- extension_score: 1=no obvious angle to extend; 5=many clear extension opportunities.
- Respond ONLY with the JSON object."""

USER_PROMPT_TEMPLATE = """\
Analyze the following paper and respond with a JSON object as instructed.

---PAPER TEXT START---
{paper_text}
---PAPER TEXT END---"""

# ─── LOAD DATA ───────────────────────────────────────────────────────────────

def load_papers(hf_dataset: str) -> list[dict]:
    """
    Stream all_papers.jsonl dari HF repo.
    Return list of dicts dengan field: _group_id, text, venue, title, authors, year, doi, abstract.
    """
    url = HF_JSONL_URL.format(dataset=hf_dataset)
    print(f"Downloading all_papers.jsonl dari {hf_dataset} ...", flush=True)

    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()

    papers = []
    buf = b""
    total_bytes = 0

    with tqdm(unit="B", unit_scale=True, unit_divisor=1024, desc="Downloading", ncols=80) as pbar:
        for chunk in resp.iter_content(chunk_size=65_536):
            buf += chunk
            total_bytes += len(chunk)
            pbar.update(len(chunk))

            # Parse baris-baris yang sudah lengkap
            lines = buf.split(b"\n")
            buf = lines[-1]  # simpan sisa yang belum lengkap
            for line in lines[:-1]:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    papers.append({
                        "_group_id": obj.get("filename") or str(len(papers)),
                        "text":      str(obj.get("full_text") or "")[:MAX_TEXT_CHARS],
                        "venue":     obj.get("venue", ""),
                        "title":     obj.get("title", ""),
                        "authors":   obj.get("authors", []),
                        "year":      obj.get("year", ""),
                        "doi":       obj.get("doi", ""),
                        "abstract":  obj.get("abstract", ""),
                    })
                except json.JSONDecodeError:
                    pass

    # Flush sisa buffer
    if buf.strip():
        try:
            obj = json.loads(buf)
            papers.append({
                "_group_id": obj.get("filename") or str(len(papers)),
                "text":      str(obj.get("full_text") or "")[:MAX_TEXT_CHARS],
                "venue":     obj.get("venue", ""),
                "title":     obj.get("title", ""),
                "authors":   obj.get("authors", []),
                "year":      obj.get("year", ""),
                "doi":       obj.get("doi", ""),
                "abstract":  obj.get("abstract", ""),
            })
        except json.JSONDecodeError:
            pass

    print(f"  Loaded {len(papers)} papers ({total_bytes/1024/1024:.1f} MB)", flush=True)
    return papers


# ─── CHECKPOINT ──────────────────────────────────────────────────────────────

def load_checkpoint(partial_path: Path) -> set[str]:
    done: set[str] = set()
    if not partial_path.exists():
        return done
    with open(partial_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "_group_id" in rec:
                    done.add(rec["_group_id"])
            except json.JSONDecodeError:
                pass
    print(f"  Checkpoint: {len(done)} paper sudah selesai (skip).", flush=True)
    return done


# ─── JSON PARSING ─────────────────────────────────────────────────────────────

def parse_json_response(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(raw[start:end])
        raise


# ─── ASYNC ANALYSIS ──────────────────────────────────────────────────────────

async def analyze_one(
    paper: dict,
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    retries: int = 3,
) -> dict:
    user_msg = USER_PROMPT_TEMPLATE.format(paper_text=paper["text"])

    result: dict = {
        "_group_id":   paper["_group_id"],
        "_venue_meta": paper.get("venue", ""),
        "_error":      None,
    }

    raw = ""
    for attempt in range(retries):
        try:
            async with sem:
                response = await client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_msg},
                    ],
                    temperature=0.0,
                    max_tokens=1024,
                )
            raw = response.choices[0].message.content or ""
            parsed = parse_json_response(raw)
            result.update(parsed)
            result["_error"] = None
            return result

        except json.JSONDecodeError as e:
            result["_error"] = f"JSONDecodeError: {e} | raw: {raw[:200]}"
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)

        except Exception as e:
            result["_error"] = f"{type(e).__name__}: {e}"
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                return result

    return result


async def analyze_all(
    papers: list[dict],
    client: AsyncOpenAI,
    concurrency: int,
    partial_path: Path,
    checkpoint_every: int,
) -> list[dict]:
    sem        = asyncio.Semaphore(concurrency)
    total      = len(papers)
    done_count = [0]
    fout       = open(partial_path, "a", encoding="utf-8")
    pbar       = tqdm(total=total, desc="Analyzing", unit="paper", ncols=80)

    async def run_one(paper: dict) -> dict:
        t0  = time.perf_counter()
        res = await analyze_one(paper, client, sem)
        elapsed = time.perf_counter() - t0

        done_count[0] += 1
        n      = done_count[0]
        status = "ERR" if res.get("_error") else "OK"
        title  = (res.get("title") or paper.get("title") or paper["_group_id"])[:50]
        pbar.set_postfix_str(f"{status} | {title}", refresh=False)
        pbar.update(1)

        fout.write(json.dumps(res, ensure_ascii=False) + "\n")
        if n % checkpoint_every == 0:
            fout.flush()

        return res

    tasks   = [asyncio.create_task(run_one(p)) for p in papers]
    results = await asyncio.gather(*tasks)

    pbar.close()
    fout.flush()
    fout.close()
    return list(results)


# ─── MERGE ───────────────────────────────────────────────────────────────────

def merge_partials(out_csv: Path, out_json: Path):
    partials = sorted(Path(".").glob(f"{PARTIAL_PREFIX}_P*.jsonl"))
    if not partials:
        print("[WARN] Tidak ada file partial ditemukan.")
        return

    rows = []
    seen: set[str] = set()

    for path in partials:
        print(f"  Reading {path} ...", flush=True)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    gid = rec.get("_group_id", "")
                    if gid not in seen:
                        seen.add(gid)
                        rows.append(rec)
                except json.JSONDecodeError:
                    pass

    if not rows:
        print("[WARN] Tidak ada baris valid.")
        return

    df = pd.DataFrame(rows)

    for col in ("doability_score", "extension_score"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "doability_score" in df.columns and "extension_score" in df.columns:
        df["combined_score"] = df["doability_score"].fillna(0) + df["extension_score"].fillna(0)
        df = df.sort_values("combined_score", ascending=False)

    df.to_csv(out_csv, index=False, encoding="utf-8")
    df.to_json(out_json, orient="records", force_ascii=False, indent=2)

    ok  = df["_error"].isna().sum() if "_error" in df.columns else len(df)
    err = len(df) - ok
    print(f"\nMerge selesai: {len(df)} papers ({ok} OK, {err} error)")
    print(f"  CSV  : {out_csv.resolve()}")
    print(f"  JSON : {out_json.resolve()}")

    cols = ["title", "_venue_meta", "doability_score", "extension_score", "combined_score", "summary"]
    cols = [c for c in cols if c in df.columns]
    print(f"\nTop 10 candidates:")
    print(df[cols].head(10).to_string(index=False))


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 3: Analisis paper IR dengan DeepSeek")

    parser.add_argument("--hf-dataset",       default="fassabilf/ir2025-papers-text")
    parser.add_argument("--partition",         type=int, default=0)
    parser.add_argument("--num-partitions",    type=int, default=1)
    parser.add_argument("--api-key",           default="sk-0989c4f9d50f46bf93ac214bed520dbb")
    parser.add_argument("--concurrency",       type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--checkpoint-every",  type=int, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--test",              type=int, default=None, metavar="N")
    parser.add_argument("--merge",             action="store_true")
    parser.add_argument("--out-csv",           default=str(FINAL_CSV))
    parser.add_argument("--out-json",          default=str(FINAL_JSON))

    args = parser.parse_args()

    if args.merge:
        print("Mode: MERGE")
        merge_partials(Path(args.out_csv), Path(args.out_json))
        return

    if not (0 <= args.partition < args.num_partitions):
        print(f"[ERROR] --partition harus 0..{args.num_partitions-1}")
        sys.exit(1)

    partial_path = Path(f"{PARTIAL_PREFIX}_P{args.partition}.jsonl")

    print(f"{'='*60}")
    print(f"Stage 3: Analisis Paper IR — DeepSeek")
    print(f"{'='*60}")
    print(f"Dataset     : {args.hf_dataset}")
    print(f"Partisi     : {args.partition} / {args.num_partitions}")
    print(f"Concurrency : {args.concurrency}")
    print(f"Checkpoint  : setiap {args.checkpoint_every} paper")
    print(f"Output      : {partial_path}")
    if args.test:
        print(f"MODE TEST   : hanya {args.test} paper")
    print()

    # Load
    papers   = load_papers(args.hf_dataset)
    done_ids = load_checkpoint(partial_path)
    papers   = [p for p in papers if p["_group_id"] not in done_ids]

    if args.num_partitions > 1:
        papers = [p for i, p in enumerate(papers) if i % args.num_partitions == args.partition]
        print(f"  Partisi {args.partition}: {len(papers)} papers", flush=True)

    if args.test:
        papers = papers[:args.test]

    if not papers:
        print("Tidak ada paper yang perlu dianalisis.")
        return

    # Estimasi
    n = len(papers)
    est_input  = n * (MAX_TEXT_CHARS // 4)
    est_output = n * 600
    est_cost   = (est_input / 1e6 * 0.27) + (est_output / 1e6 * 1.10)
    est_min    = n / (args.concurrency * 3)
    print(f"\nAkan menganalisis {n} paper")
    print(f"  Estimasi biaya : ~${est_cost:.3f} USD")
    print(f"  Estimasi waktu : ~{est_min:.1f} menit\n")

    client = AsyncOpenAI(api_key=args.api_key, base_url=DEEPSEEK_BASE_URL)

    t0      = time.time()
    results = asyncio.run(analyze_all(
        papers          = papers,
        client          = client,
        concurrency     = args.concurrency,
        partial_path    = partial_path,
        checkpoint_every= args.checkpoint_every,
    ))
    elapsed = time.time() - t0

    ok_n  = sum(1 for r in results if not r.get("_error"))
    err_n = len(results) - ok_n

    print(f"\n{'='*60}")
    print(f"SELESAI: {ok_n}/{len(results)} OK  |  {elapsed:.1f}s ({elapsed/60:.1f} menit)")
    print(f"Output : {partial_path.resolve()}")
    print(f"Merge  : python3 analyze.py --merge")
    if err_n:
        errs = [r for r in results if r.get("_error")]
        print(f"\nContoh error ({err_n} total):")
        for r in errs[:3]:
            print(f"  {r['_group_id'][:50]}: {r['_error'][:80]}")


if __name__ == "__main__":
    main()

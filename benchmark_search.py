"""Is Faiss ``IndexFlatIP`` worth replacing with an approximate index?

``search_all_chars_in_corpus.py`` uses a brute-force inner-product index. Brute force is *exact*
but O(N) per query, so the obvious question is whether an approximate (ANN) index would be faster.
This script answers it with measurements rather than intuition, in two separable parts:

Recall  (real embeddings only)
    How many of the exact top-K does each ANN index actually return? Recall depends on how the
    vectors are distributed, so this is measured only on real glyph embeddings, at several corpus
    sizes obtained by subsampling.

Latency  (real embeddings, then synthetic vectors for sizes we do not have)
    Milliseconds per query against an index of N vectors. Timing is almost independent of whether
    the vectors are meaningful, so sizes beyond the real corpus use random unit vectors -- clearly
    reported as synthetic, and never used for recall.

Together these give the crossover: the corpus size at which giving up exactness starts to buy
enough speed to be worth it.

Example
-------
    python benchmark_search.py                          # uses chinese-clip-large
    python benchmark_search.py --backend chinese-clip --max-synthetic 500000
"""

import argparse
import os
import time

import faiss
import numpy as np
import pandas as pd

from feature_extractors import BACKENDS, DTYPES
from search_all_chars_in_corpus import cache_tag, set_deterministic

#: Query batch used for every latency measurement, so numbers are comparable across index types.
#: Large enough that Faiss' 16-thread dispatch overhead does not dominate -- at 1000 queries the
#: small-N timings were noise (N=5k measured *slower* than N=10k).
N_QUERIES = 4000

#: Timed runs per measurement; the fastest is reported, which is the standard way to read through
#: interference from the vLLM workers sharing this machine.
N_REPEATS = 3


def load_embeddings(output_dir, backend, limit=None):
    """Find whichever cached embedding matrix exists for this backend, fp32 or fp16."""
    for dtype in sorted(DTYPES):
        path = os.path.join(output_dir, f"embeddings_{cache_tag(backend, dtype, limit)}.npz")
        if os.path.exists(path):
            emb = np.load(path, allow_pickle=True)["embeddings"].astype(np.float32)
            print(f"Loaded {emb.shape[0]} x {emb.shape[1]} embeddings ({backend}, {dtype})")
            return emb
    raise SystemExit(f"No embedding cache for {backend!r} in {output_dir} -- run the search script first.")


def build_flat(vectors):
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def build_hnsw(vectors, m=32, ef_construction=80):
    index = faiss.IndexHNSWFlat(vectors.shape[1], m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.add(vectors)
    return index


def build_ivf_flat(vectors, nlist):
    quantizer = faiss.IndexFlatIP(vectors.shape[1])
    index = faiss.IndexIVFFlat(quantizer, vectors.shape[1], nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(vectors)
    index.add(vectors)
    return index


def build_ivf_pq(vectors, nlist, m, nbits=8):
    quantizer = faiss.IndexFlatIP(vectors.shape[1])
    index = faiss.IndexIVFPQ(quantizer, vectors.shape[1], nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
    index.train(vectors)
    index.add(vectors)
    return index


def largest_divisor_at_most(value, cap):
    """PQ needs the sub-quantizer count to divide the dimension exactly."""
    for candidate in range(cap, 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def time_search(index, queries, top_k):
    """Return (indices, best-of-N_REPEATS milliseconds per query), after a warmup pass."""
    index.search(queries[:64], top_k)  # warm caches/threads so the first repeat is not an outlier
    best = float("inf")
    indices = None
    for _ in range(N_REPEATS):
        start = time.perf_counter()
        _, indices = index.search(queries, top_k)
        best = min(best, time.perf_counter() - start)
    return indices, best / len(queries) * 1000


def recall_at_k(approx_idx, exact_idx):
    """Mean fraction of the exact top-K that the approximate index also returned."""
    return float(
        np.mean([len(set(a) & set(e)) / len(e) for a, e in zip(approx_idx, exact_idx)])
    )


def index_variants(vectors, top_k):
    """Yield (name, build_fn, apply_param_fn, [param values]) for each index type."""
    n, d = vectors.shape
    nlist = max(1, min(4096, int(4 * np.sqrt(n))))
    pq_m = largest_divisor_at_most(d, 64)

    yield "HNSW32", lambda v: build_hnsw(v), \
        (lambda ix, p: setattr(ix.hnsw, "efSearch", p)), [16, 32, 64, 128, 256], "efSearch"
    yield f"IVFFlat(nlist={nlist})", lambda v: build_ivf_flat(v, nlist), \
        (lambda ix, p: setattr(ix, "nprobe", p)), [1, 4, 8, 16, 32, 64], "nprobe"
    yield f"IVFPQ(nlist={nlist},m={pq_m})", lambda v: build_ivf_pq(v, nlist, pq_m), \
        (lambda ix, p: setattr(ix, "nprobe", p)), [1, 8, 32, 64], "nprobe"


def evaluate(vectors, top_k, label, measure_recall):
    """Build every index over ``vectors`` and record recall (optional) and latency."""
    n, d = vectors.shape
    rng = np.random.default_rng(0)
    queries = vectors[rng.choice(n, size=min(N_QUERIES, n), replace=False)]

    rows = []
    start = time.perf_counter()
    flat = build_flat(vectors)
    flat_build = time.perf_counter() - start
    exact_idx, flat_ms = time_search(flat, queries, top_k)
    rows.append({
        "corpus": label, "N": n, "dim": d, "index": "Flat (exact)", "param": "-",
        "recall@k": 1.0 if measure_recall else None,
        "build_s": round(flat_build, 2), "ms_per_query": round(flat_ms, 3), "speedup": 1.0,
    })
    print(f"  {'Flat (exact)':26s}            recall=1.0000  build={flat_build:6.2f}s  {flat_ms:8.3f} ms/q")

    for name, build_fn, set_param, values, param_name in index_variants(vectors, top_k):
        start = time.perf_counter()
        index = build_fn(vectors)
        build_s = time.perf_counter() - start
        for value in values:
            set_param(index, value)
            approx_idx, ms = time_search(index, queries, top_k)
            recall = recall_at_k(approx_idx, exact_idx) if measure_recall else None
            rows.append({
                "corpus": label, "N": n, "dim": d, "index": name,
                "param": f"{param_name}={value}",
                "recall@k": round(recall, 4) if recall is not None else None,
                "build_s": round(build_s, 2), "ms_per_query": round(ms, 3),
                "speedup": round(flat_ms / ms, 2),
            })
            shown = f"{recall:.4f}" if recall is not None else "  n/a "
            print(f"  {name:26s} {param_name}={value:<4d} recall={shown}  "
                  f"build={build_s:6.2f}s  {ms:8.3f} ms/q  ({flat_ms / ms:5.2f}x)")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", default="chinese-clip-large", choices=BACKENDS)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--limit", type=int, default=None, help="match the --limit used when embedding")
    parser.add_argument("--max-synthetic", type=int, default=1_000_000,
                        help="largest synthetic corpus for the latency curve (0 disables)")
    args = parser.parse_args()

    set_deterministic()
    embeddings = load_embeddings(args.output_dir, args.backend, args.limit)
    n, d = embeddings.shape
    rng = np.random.default_rng(42)

    rows = []

    # --- Recall + latency on real data, at several corpus sizes -------------------------------
    real_sizes = [size for size in (5_000, 10_000, n) if size <= n]
    for size in real_sizes:
        subset = embeddings if size == n else embeddings[rng.choice(n, size=size, replace=False)]
        print(f"\n=== REAL embeddings, N={size} ===")
        rows += evaluate(subset, args.top_k, f"real-{size}", measure_recall=True)

    # --- Latency only, on synthetic vectors, for sizes we do not have --------------------------
    synthetic_sizes = [s for s in (100_000, 500_000, 1_000_000) if s <= args.max_synthetic]
    for size in synthetic_sizes:
        print(f"\n=== SYNTHETIC vectors (latency only, NOT a recall measurement), N={size} ===")
        vectors = rng.normal(size=(size, d)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        rows += evaluate(vectors, args.top_k, f"synthetic-{size}", measure_recall=False)
        del vectors

    df = pd.DataFrame(rows)
    out_path = os.path.join(args.output_dir, "benchmark_search.xlsx")
    with pd.ExcelWriter(out_path) as writer:
        df.to_excel(writer, sheet_name="All", index=False)
        df[df["index"] == "Flat (exact)"].to_excel(writer, sheet_name="Flat scaling", index=False)
    print(f"\nWrote {out_path}")

    # Headline: exact search cost as the corpus grows.
    flat = df[df["index"] == "Flat (exact)"]
    print("\nExact Flat search, ms per query:")
    for _, row in flat.iterrows():
        print(f"  N={row['N']:>9,}  {row['ms_per_query']:8.3f} ms  ({row['corpus']})")


if __name__ == "__main__":
    main()

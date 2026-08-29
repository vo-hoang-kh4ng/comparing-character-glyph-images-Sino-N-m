"""Corpus-wide top-K visual similarity search over Sino-Nom glyph images.

For every character in ``final_characteristics-v2.xlsx`` that has an image in ``./images/``, this
embeds the glyph, indexes all embeddings with Faiss (``IndexFlatIP`` == cosine similarity on
L2-normalized vectors), and writes each character's top-K nearest neighbours.

The feature extractor is pluggable -- see ``feature_extractors.py``. The original ResNet18
pipeline is still the ``resnet18`` backend, so old results remain reproducible.

Examples
--------
    python search_all_chars_in_corpus.py                          # default: chinese-clip
    python search_all_chars_in_corpus.py --backend resnet18       # original baseline
    python search_all_chars_in_corpus.py --backend dinov2 --limit 500   # quick trial run

Embeddings are cached to ``output/embeddings_<backend>.npz`` and reused on later runs (pass
``--refresh`` to recompute), which is what makes ``benchmark_extractors.py`` cheap to iterate on.
"""

import argparse
import os
import random
import time
import warnings

import faiss
import numpy as np
import pandas as pd
import torch

from feature_extractors import BACKENDS, DTYPES, build_extractor

warnings.filterwarnings("ignore")

CHAR_TABLE = "final_characteristics-v2.xlsx"


def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cache_tag(backend, dtype="fp32", limit=None):
    """Filename stem shared by the embedding cache and the result files.

    fp32 and no-limit stay unsuffixed so the original filenames keep working.
    ``benchmark_extractors.py`` builds the same tag to find the caches.
    """
    tag = backend + (f"_{dtype}" if dtype != "fp32" else "")
    return tag + (f"_limit{limit}" if limit else "")


def load_corpus(image_folder, limit=None):
    """Return the (UNICODE, CHAR, image path) rows that actually have a glyph image on disk."""
    df = pd.read_excel(CHAR_TABLE)[["UNICODE", "CHAR"]].copy()
    df["path"] = df["UNICODE"].map(lambda u: os.path.join(image_folder, f"{u}.jpg"))
    exists = df["path"].map(os.path.exists)
    missing = int((~exists).sum())
    df = df[exists].reset_index(drop=True)
    if limit:
        df = df.head(limit).copy()
    print(f"Corpus: {len(df)} characters with images ({missing} entries have no image file)")
    return df


def compute_embeddings(df, backend, batch_size, device, cache_path, refresh=False, dtype="fp32"):
    """Embed every glyph, caching the result keyed by the UNICODE list it was built from."""
    if cache_path and os.path.exists(cache_path) and not refresh:
        cached = np.load(cache_path, allow_pickle=True)
        if list(cached["unicodes"]) == list(df["UNICODE"]):
            print(f"Loaded cached embeddings from {cache_path}")
            return cached["embeddings"], float(cached["elapsed"])
        print(f"Cache {cache_path} does not match the current corpus -- recomputing")

    extractor = build_extractor(backend, device=device, dtype=dtype)
    print(f"Embedding {len(df)} glyphs with '{backend}' (dim={extractor.dim}, batch={batch_size})")
    start = time.perf_counter()
    embeddings = extractor.embed_paths(
        df["path"].tolist(), batch_size=batch_size, progress_every=20
    )
    elapsed = time.perf_counter() - start
    print(f"Done in {elapsed:.1f}s ({len(df) / elapsed:.1f} img/s)")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(
            cache_path,
            embeddings=embeddings,
            unicodes=np.array(df["UNICODE"], dtype=object),
            elapsed=elapsed,
        )
        print(f"Cached embeddings to {cache_path}")
    return embeddings, elapsed


def search_top_k(embeddings, top_k):
    """Faiss inner-product search returning, per row, the top-K neighbours excluding self."""
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    # Ask for K+1 because a vector is always its own nearest neighbour.
    scores, indices = index.search(embeddings, top_k + 1)

    neighbour_idx = np.empty((len(embeddings), top_k), dtype=np.int64)
    neighbour_score = np.empty((len(embeddings), top_k), dtype=np.float32)
    for row in range(len(embeddings)):
        # Dropping self leaves either K or K+1 hits (duplicate glyphs can outrank self), so
        # slicing to top_k is always safe.
        keep = indices[row] != row
        neighbour_idx[row] = indices[row][keep][:top_k]
        neighbour_score[row] = scores[row][keep][:top_k]
    return neighbour_idx, neighbour_score


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", default="chinese-clip", choices=BACKENDS,
                        help="feature extractor to use (default: chinese-clip)")
    parser.add_argument("--top-k", type=int, default=20, help="neighbours per character (default: 20)")
    parser.add_argument("--images", default="./images", help="folder of <UNICODE>.jpg glyphs")
    parser.add_argument("--output-dir", default="./output", help="where results are written")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu", help="'cpu' or 'cuda'")
    parser.add_argument("--dtype", default="fp32", choices=sorted(DTYPES),
                        help="weight precision for the HuggingFace backends (default: fp32)")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N characters")
    parser.add_argument("--refresh", action="store_true", help="ignore cached embeddings")
    args = parser.parse_args()

    set_deterministic()
    os.makedirs(args.output_dir, exist_ok=True)

    df = load_corpus(args.images, limit=args.limit)
    if df.empty:
        raise SystemExit(f"No glyph images found under {args.images!r} -- extract images.zip first.")

    tag = cache_tag(args.backend, args.dtype, args.limit)
    cache_path = os.path.join(args.output_dir, f"embeddings_{tag}.npz")
    embeddings, _ = compute_embeddings(
        df, args.backend, args.batch_size, args.device, cache_path,
        refresh=args.refresh, dtype=args.dtype,
    )

    top_k = min(args.top_k, len(df) - 1)
    neighbour_idx, neighbour_score = search_top_k(embeddings, top_k)

    chars = df["CHAR"].to_numpy()
    output_df = pd.DataFrame(
        {
            "Input Character": chars,
            f"Top {top_k} Similar Characters": [list(chars[row]) for row in neighbour_idx],
            "Similarity Scores": [
                [round(float(s), 4) for s in row] for row in neighbour_score
            ],
        }
    )

    xlsx_path = os.path.join(args.output_dir, f"output_top_k_similar_char_{tag}.xlsx")
    csv_path = os.path.join(args.output_dir, f"output_top_k_similar_char_{tag}.csv")
    output_df.to_excel(xlsx_path, index=False)
    output_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {xlsx_path}\nWrote {csv_path}")
    return output_df


if __name__ == "__main__":
    main()

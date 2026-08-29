"""Top-K visually similar Sino-Nom characters *within one Quoc Ngu reading*.

Part 2 of the project. Given a Vietnamese word and a query image, this ranks only the characters
that share that Quoc Ngu reading -- a handful of candidates rather than the whole corpus, which is
why no Faiss index is needed here.

Two scoring methods, selectable with ``--method``:

histogram
    The original pipeline, kept bit-for-bit: grayscale -> resize 100x100 -> Gaussian blur -> Canny
    edges -> Otsu threshold -> contrast stretch -> 256-bin histogram, then L2 distance turned into
    ``1 / (1 + distance)``. It compares *edge-pixel counts*, not shape, so two characters with
    similar amounts of ink score alike however differently they are drawn.

embedding
    The same feature extractors Part 1 uses (``feature_extractors.py``), scored by cosine
    similarity. Candidate glyphs are read straight out of the ``output/embeddings_<backend>.npz``
    cache when it exists, so this usually costs one forward pass for the query image alone.

``--method both`` runs the two side by side and reports how much their rankings agree -- the
comparison the report needs.

Examples
--------
    python search_use_QuocNgu_mapping.py --word ta --image ./test_images/ta.jpg
    python search_use_QuocNgu_mapping.py --word ta --image ./images/50AC.jpg --method both
    python search_use_QuocNgu_mapping.py --word trăm --backend chinese-clip-large --dtype fp16
"""

import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

from feature_extractors import BACKENDS, DTYPES, build_extractor
from search_all_chars_in_corpus import CHAR_TABLE, cache_tag

QUOCNGU_TABLE = "QuocNgu_SinoNom_Dic.xlsx"


def load_candidates(word, image_folder):
    """Characters sharing this Quoc Ngu reading that also have a glyph image on disk."""
    quocngu = pd.read_excel(QUOCNGU_TABLE)
    chars = pd.read_excel(CHAR_TABLE)

    sino_nom = quocngu[quocngu["QuocNgu"] == word]["SinoNom"].tolist()
    if not sino_nom:
        raise SystemExit(f"No SinoNom entry for {word!r} in {QUOCNGU_TABLE}.")

    df = chars[chars["CHAR"].isin(sino_nom)][["UNICODE", "CHAR"]].copy()
    df["path"] = df["UNICODE"].map(lambda u: os.path.join(image_folder, f"{u}.jpg"))
    exists = df["path"].map(os.path.exists)
    missing = int((~exists).sum())
    df = df[exists].reset_index(drop=True)
    print(f"'{word}': {len(sino_nom)} SinoNom entries, {len(df)} with a glyph image "
          f"({missing} without)")
    if df.empty:
        raise SystemExit("No candidate has an image -- nothing to rank.")
    return df


# --- method 1: the original histogram pipeline ------------------------------------------------

def extract_histogram(image_path):
    """Unchanged from the original script -- see the module docstring for what it actually measures."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (100, 100))
    image = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(image, 100, 200)
    _, binary = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contrast_stretched = cv2.normalize(binary, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    hist = cv2.calcHist([contrast_stretched], [0], None, [256], [0, 256])
    return cv2.normalize(hist, hist).flatten()


def score_histogram(df, query_path):
    query = extract_histogram(query_path)
    distances = np.array([
        np.linalg.norm(query - extract_histogram(path)) for path in df["path"]
    ])
    return 1.0 / (1.0 + distances)


# --- method 2: embeddings, reusing Part 1's cache ----------------------------------------------

def candidate_embeddings(df, backend, dtype, output_dir, device, batch_size):
    """Look the candidates up in the corpus cache; embed only what the cache is missing."""
    cache_path = os.path.join(output_dir, f"embeddings_{cache_tag(backend, dtype)}.npz")
    if os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=True)
        row_of = {str(u): i for i, u in enumerate(cached["unicodes"])}
        rows = [row_of.get(str(u)) for u in df["UNICODE"]]
        if all(row is not None for row in rows):
            print(f"Reused all {len(rows)} candidate embeddings from {cache_path}")
            return cached["embeddings"][rows], None
        print(f"Cache {cache_path} covers only part of the candidates -- embedding them directly")

    extractor = build_extractor(backend, device=device, dtype=dtype)
    return extractor.embed_paths(df["path"].tolist(), batch_size=batch_size), extractor


def score_embedding(df, query_path, backend, dtype, output_dir, device, batch_size):
    embeddings, extractor = candidate_embeddings(
        df, backend, dtype, output_dir, device, batch_size
    )
    if extractor is None:
        extractor = build_extractor(backend, device=device, dtype=dtype)
    query = extractor.embed_paths([query_path], batch_size=1)[0]
    # Both sides are L2-normalized by embed_paths, so the dot product is cosine similarity.
    return embeddings @ query


def main():
    # Sino-Nom glyphs crash a cp1252 Windows console otherwise.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--word", default="ta", help="Quoc Ngu reading to restrict the search to")
    parser.add_argument("--image", default="./test_images/ta.jpg", help="query glyph image")
    parser.add_argument("--method", default="embedding", choices=("embedding", "histogram", "both"))
    parser.add_argument("--backend", default="chinese-clip", choices=BACKENDS)
    parser.add_argument("--dtype", default="fp32", choices=sorted(DTYPES))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--images", default="./images")
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu", help="'cpu' or 'cuda'")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise SystemExit(f"Query image {args.image!r} not found.")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Query: {args.image}  (word: {args.word})")
    df = load_candidates(args.word, args.images)

    methods = ("embedding", "histogram") if args.method == "both" else (args.method,)
    scores = {}
    for method in methods:
        if method == "histogram":
            scores[method] = score_histogram(df, args.image)
        else:
            scores[method] = score_embedding(
                df, args.image, args.backend, args.dtype,
                args.output_dir, args.device, args.batch_size,
            )

    top_k = min(args.top_k, len(df))
    for method, values in scores.items():
        order = np.argsort(-values)[:top_k]
        label = args.backend if method == "embedding" else "histogram"
        out = pd.DataFrame({
            "UNICODE": df["UNICODE"].to_numpy()[order],
            "SinoNom": df["CHAR"].to_numpy()[order],
            "Similarity": [round(float(v), 4) for v in values[order]],
        })
        print(f"\n--- top {top_k} by {method} ({label}) ---")
        print(out.to_string(index=False))

        suffix = "histogram" if method == "histogram" else f"embedding_{args.backend}"
        stem = os.path.join(args.output_dir, f"output_{suffix}_similarity")
        out.to_excel(f"{stem}.xlsx", index=False)
        out.to_csv(f"{stem}.csv", index=False, encoding="utf-8-sig")
        print(f"Wrote {stem}.xlsx / .csv")

    if len(scores) == 2:
        ranks = {m: set(np.argsort(-v)[:top_k]) for m, v in scores.items()}
        shared = ranks["embedding"] & ranks["histogram"]
        print(f"\nAgreement: the two methods share {len(shared)}/{top_k} of the top-{top_k} "
              f"({len(shared) / top_k * 100:.0f}%)")


if __name__ == "__main__":
    main()

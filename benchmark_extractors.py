"""Compare feature-extractor backends for Sino-Nom glyph similarity.

Reads the embedding caches written by ``search_all_chars_in_corpus.py`` (run that once per
backend first), re-runs the Faiss top-K search, and reports for each backend:

Speed
    Wall-clock embedding time and throughput recorded during extraction.

Quality proxies (weak labels derived from ``final_characteristics-v2.xlsx``)
    radical@k    share of the top-K neighbours built on the same Kangxi radical as the query
    stroke_mae@k mean |stroke-count difference| between query and its top-K neighbours
    ids_jaccard@k mean Jaccard overlap of IDS decomposition components (``SHAPE_MORPH``)

    None of these is a human relevance judgement -- they are cheap stand-ins that correlate with
    "looks alike", useful for ranking backends against each other. The full evaluation is a
    separate work item.

Agreement
    Mean overlap of the top-K lists between every pair of backends, i.e. how differently the
    backends actually rank the corpus.

Outputs
    output/benchmark_summary.xlsx   metrics + per-backend sample neighbour lists
    output/contact_sheet_<backend>.png  query glyph + its top-5 neighbours, for eyeballing

Example
-------
    python search_all_chars_in_corpus.py --backend resnet18
    python search_all_chars_in_corpus.py --backend chinese-clip
    python benchmark_extractors.py --backends resnet18 chinese-clip
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from feature_extractors import BACKENDS, DTYPES
from search_all_chars_in_corpus import (
    CHAR_TABLE,
    cache_tag,
    load_corpus,
    search_top_k,
    set_deterministic,
)

#: Ideographic Description Characters -- structural operators in SHAPE_MORPH, not components.
IDS_OPERATORS = set("⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻")


def parse_radical(value):
    """'nhất 一 (+3 nét)' -> 'nhất 一'. The '(+n nét)' suffix is the residual stroke count."""
    if not isinstance(value, str):
        return None
    return re.sub(r"\s*\(.*\)\s*$", "", value).strip() or None


def parse_ids_components(value):
    """'⿱乛丄' -> {'乛', '丄'}: the leaf components, with IDS operators stripped."""
    if not isinstance(value, str):
        return frozenset()
    return frozenset(ch for ch in value if ch not in IDS_OPERATORS and not ch.isspace())


def load_metadata(df):
    """Attach the weak-label columns to the corpus frame, aligned on UNICODE."""
    meta = pd.read_excel(CHAR_TABLE)[["UNICODE", "RADICAL", "STROKE_NUM", "SHAPE_MORPH"]]
    merged = df.merge(meta, on="UNICODE", how="left")
    return (
        merged["RADICAL"].map(parse_radical).to_numpy(dtype=object),
        merged["STROKE_NUM"].to_numpy(dtype=float),
        merged["SHAPE_MORPH"].map(parse_ids_components).to_numpy(dtype=object),
    )


def quality_metrics(neighbour_idx, radicals, strokes, components):
    """Mean weak-label agreement between each query and its retrieved neighbours."""
    radical_hits, stroke_diffs, jaccards = [], [], []

    for row, neighbours in enumerate(neighbour_idx):
        if radicals[row] is not None:
            valid = [radicals[n] for n in neighbours if radicals[n] is not None]
            if valid:
                radical_hits.append(np.mean([r == radicals[row] for r in valid]))

        if not np.isnan(strokes[row]):
            valid = strokes[neighbours]
            valid = valid[~np.isnan(valid)]
            if len(valid):
                stroke_diffs.append(np.mean(np.abs(valid - strokes[row])))

        query_parts = components[row]
        if query_parts:
            scores = []
            for n in neighbours:
                other = components[n]
                if other:
                    union = len(query_parts | other)
                    scores.append(len(query_parts & other) / union if union else 0.0)
            if scores:
                jaccards.append(np.mean(scores))

    return {
        "radical@k": float(np.mean(radical_hits)) if radical_hits else float("nan"),
        "stroke_mae@k": float(np.mean(stroke_diffs)) if stroke_diffs else float("nan"),
        "ids_jaccard@k": float(np.mean(jaccards)) if jaccards else float("nan"),
    }


def contact_sheet(df, neighbour_idx, rows, out_path, thumb=88, n_neighbours=5):
    """Render 'query | top-N neighbours' strips so the ranking can be judged by eye."""
    pad, label_h = 6, 14
    cols = n_neighbours + 1
    width = cols * (thumb + pad) + pad
    height = len(rows) * (thumb + label_h + pad) + pad
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)

    for r, query in enumerate(rows):
        y = pad + r * (thumb + label_h + pad)
        for c, idx in enumerate([query] + list(neighbour_idx[query][:n_neighbours])):
            x = pad + c * (thumb + pad)
            glyph = Image.open(df["path"].iloc[idx]).convert("L").resize((thumb, thumb))
            sheet.paste(glyph, (x, y))
            # UNICODE values are ASCII hex, so the bitmap default font renders them fine.
            draw.text((x, y + thumb), str(df["UNICODE"].iloc[idx]), fill="black")
            if c == 0:
                # Separate the query from its results with a vertical rule.
                draw.rectangle([x - 1, y - 1, x + thumb, y + thumb], outline="red")
    sheet.save(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backends", nargs="+", default=list(BACKENDS), choices=BACKENDS)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--images", default="./images")
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--limit", type=int, default=None, help="match the --limit used when embedding")
    parser.add_argument("--samples", type=int, default=25, help="characters shown in the visual comparison")
    args = parser.parse_args()

    set_deterministic()
    df = load_corpus(args.images, limit=args.limit)
    radicals, strokes, components = load_metadata(df)

    suffix = f"_limit{args.limit}" if args.limit else ""
    rng = np.random.default_rng(42)
    sample_rows = np.sort(rng.choice(len(df), size=min(args.samples, len(df)), replace=False))

    summary, neighbours_by_backend, sample_tables = [], {}, {}

    for backend in args.backends:
        # Backends were not all embedded at the same precision (the large ViTs need fp16 to fit),
        # so accept whichever cache exists rather than forcing one dtype across the comparison.
        found = [
            (dt, os.path.join(args.output_dir, f"embeddings_{cache_tag(backend, dt, args.limit)}.npz"))
            for dt in sorted(DTYPES)
        ]
        found = [(dt, path) for dt, path in found if os.path.exists(path)]
        if not found:
            print(f"[skip] {backend}: no cache in {args.output_dir} -- run search_all_chars_in_corpus.py first")
            continue
        dtype, cache_path = found[0]
        if len(found) > 1:
            print(f"[note] {backend}: caches for {[d for d, _ in found]} exist, using {dtype}")
        cached = np.load(cache_path, allow_pickle=True)
        embeddings = cached["embeddings"]
        elapsed = float(cached["elapsed"])

        neighbour_idx, _ = search_top_k(embeddings, min(args.top_k, len(df) - 1))
        neighbours_by_backend[backend] = neighbour_idx

        metrics = quality_metrics(neighbour_idx, radicals, strokes, components)
        summary.append({
            "backend": backend,
            "dtype": dtype,
            "dim": int(embeddings.shape[1]),
            "embed_seconds": round(elapsed, 1),
            "img_per_sec": round(len(df) / elapsed, 1),
            **{k: round(v, 4) for k, v in metrics.items()},
        })
        print(f"{backend:20s} {dtype:4s} dim={embeddings.shape[1]:5d} {len(df)/elapsed:6.1f} img/s  " +
              "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

        chars = df["CHAR"].to_numpy()
        sample_tables[backend] = [
            "".join(chars[neighbour_idx[row][:10]]) for row in sample_rows
        ]
        contact_sheet(df, neighbour_idx, sample_rows[:12],
                      os.path.join(args.output_dir, f"contact_sheet_{backend}{suffix}.png"))

    if not summary:
        raise SystemExit("No embedding caches found -- nothing to compare.")

    # Pairwise agreement: how much do two backends' top-K lists overlap on average?
    agreement = []
    names = list(neighbours_by_backend)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = [
                len(set(neighbours_by_backend[a][r]) & set(neighbours_by_backend[b][r]))
                for r in range(len(df))
            ]
            agreement.append({
                "backend_a": a,
                "backend_b": b,
                f"mean_overlap@{args.top_k}": round(float(np.mean(overlap)), 2),
                "overlap_ratio": round(float(np.mean(overlap)) / args.top_k, 3),
            })

    samples_df = pd.DataFrame({
        "UNICODE": df["UNICODE"].to_numpy()[sample_rows],
        "Query": df["CHAR"].to_numpy()[sample_rows],
        **{f"top10_{b}": v for b, v in sample_tables.items()},
    })

    out_path = os.path.join(args.output_dir, f"benchmark_summary{suffix}.xlsx")
    with pd.ExcelWriter(out_path) as writer:
        pd.DataFrame(summary).to_excel(writer, sheet_name="Metrics", index=False)
        pd.DataFrame(agreement).to_excel(writer, sheet_name="Agreement", index=False)
        samples_df.to_excel(writer, sheet_name="Samples", index=False)
    print(f"\nWrote {out_path}")
    print(f"Contact sheets: {args.output_dir}/contact_sheet_<backend>{suffix}.png")


if __name__ == "__main__":
    main()

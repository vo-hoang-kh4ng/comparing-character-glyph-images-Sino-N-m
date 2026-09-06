"""Does metadata reranking actually help? Three suites, only one of which produces a number.

``rerank.py`` scores candidates with RADICAL, STROKE_NUM and SHAPE_MORPH. ``benchmark_extractors.py``
scores retrieval with ``radical@k``, ``stroke_mae@k`` and ``ids_jaccard@k`` -- derived from those
same three columns. Measuring the reranker with those metrics is circular, so this script exists to
keep the circular measurement and the honest one visibly apart.

--suite proxy   The circular measurement, run ON PURPOSE and labelled as invalid.
                It exists to show how large the illusion is: radical@20 climbs toward its 0.978
                ceiling because the reranker was handed the answer. Output columns are suffixed
                [CIRCULAR] so a number pasted into a slide carries its own warning. This suite is
                a negative result for the report, never a result.

--suite render  The honest number. Queries are re-rendered glyphs from ``render_fonts.py``; the
                label is the character's own Unicode codepoint, which owes nothing to the three
                metadata columns. n is in the thousands, so the 95% CI is around a point --
                small enough to resolve the kind of improvement reranking might plausibly give.
                Every query is an image of unknown identity, so this suite necessarily exercises
                ``infer_query_meta`` rather than looking metadata up.

--suite scans   The final read-out on the 59 real scans in ``test_images/``, using the human
                labels in ``output/label_sheets/label_template.csv``. Report it, do not tune on
                it: at n=59 the 95% CI is about +/-13 points, so nothing short of a 15-point swing
                is distinguishable from noise. Reported with CIs for exactly that reason.

The four configurations worth running on --suite render::

    python evaluate_rerank.py --suite render --augment --normalize none --no-rerank   # 1 baseline
    python evaluate_rerank.py --suite render --augment --oracle-meta                  # 2 oracle
    python evaluate_rerank.py --suite render --augment                                # 3 real
    python evaluate_rerank.py --suite render --augment --normalize none               # 4 pre-B2

(2) deliberately cheats: it reads the query's true metadata off the label. It is a diagnostic, not
a result, and separates two different failures. If (2) is no better than (1), metadata cannot help
this task and there is no point tuning weights. If (2) beats (1) but (3) does not, the idea is
sound and ``infer_query_meta`` is the weak link. (4) shows whether the B2 normalization fix earns
its keep.

Nothing here modifies the existing scripts; helpers are imported from them.
"""

import argparse
import math
import os
import random
import sys

import numpy as np
import pandas as pd

from rerank import NORMALIZERS, Reranker, parse_weights, warn_circular

SUITES = ("proxy", "render", "scans")


def wilson_ci(hits, n, z=1.96):
    """Wilson score interval. Beats the normal approximation at the small n this project has."""
    if n == 0:
        return 0.0, 0.0
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def report_hits(label, ranks, n, ks=(1, 5, 20)):
    """Print hit@k with CIs plus MRR. ranks: 1-based rank of the correct answer, or None."""
    print(f"\n  {label}  (n = {n})")
    for k in ks:
        hits = sum(1 for r in ranks if r and r <= k)
        lo, hi = wilson_ci(hits, n)
        print(f"    hit@{k:<3d} = {hits:5d}/{n} = {hits / n:.4f}   95% CI [{lo:.4f}, {hi:.4f}]"
              f"   +/-{(hi - lo) / 2 * 100:.1f} pts")
    mrr = float(np.mean([1 / r if r else 0.0 for r in ranks]))
    print(f"    MRR      = {mrr:.4f}")
    return {"hit@1": sum(1 for r in ranks if r == 1) / n, "MRR": mrr}


def load_corpus_cache(output_dir, backend, dtype, limit=None):
    """The embedding cache written by ``search_all_chars_in_corpus.py``."""
    from search_all_chars_in_corpus import cache_tag

    tag = cache_tag(backend, dtype, limit)
    path = os.path.join(output_dir, f"embeddings_{tag}.npz")
    if not os.path.exists(path):
        raise SystemExit(
            f"No {path} -- run: python search_all_chars_in_corpus.py --backend {backend} "
            f"--dtype {dtype}"
        )
    cached = np.load(path, allow_pickle=True)
    embeddings = cached["embeddings"].astype("float32")
    unicodes = np.array([str(u) for u in cached["unicodes"]])
    print(f"Corpus: {len(unicodes)} embeddings (dim {embeddings.shape[1]}) from {path}")
    return embeddings, unicodes, tag


def build_reranker(args):
    reranker = Reranker.from_excel(
        weights=parse_weights(args.weights), normalize=args.normalize)
    print(f"Reranker: normalize={args.normalize} weights={reranker.weights}")
    return reranker


def rerank_one(reranker, args, candidates, neighbours, true_unicode=None):
    """Rerank one candidate list for a query image of unknown identity.

    candidates: [(unicode, cosine)] -- the list to reorder.
    neighbours: [(unicode, cosine)] -- WHOLE-CORPUS top-N, the evidence for inferring the query's
                metadata. Never pass a Part 2 same-reading group here: those characters share a
                reading, not a shape, so voting inside them is noise.
    """
    if args.oracle_meta:
        # Deliberate leak: reads the answer's metadata off the label. Diagnostic only -- it is the
        # ceiling that a perfect infer_query_meta would reach, never a reportable result.
        meta = reranker.meta_for(true_unicode)
    else:
        meta = reranker.infer_query_meta(neighbours, top_n=args.infer_top_n,
                                         min_margin=args.min_margin)
    ranked = reranker.rerank(meta, candidates)
    return [r.unicode for r in ranked], meta


# --- suite: proxy (circular on purpose) -------------------------------------------------------

def suite_proxy(args):
    from benchmark_extractors import load_metadata, quality_metrics
    from search_all_chars_in_corpus import load_corpus, search_top_k, set_deterministic

    set_deterministic(args.seed)
    df = load_corpus(args.images, limit=args.limit)
    embeddings, unicodes, tag = load_corpus_cache(
        args.output_dir, args.backend, args.dtype, args.limit)
    if list(unicodes) != list(df["UNICODE"].astype(str)):
        raise SystemExit("Cache and corpus disagree -- pass a matching --limit.")

    reranker = build_reranker(args)
    tainted = reranker.circularity_report()
    if tainted:
        print(warn_circular(tainted))

    radicals, strokes, components = load_metadata(df)
    pool = min(args.pool, len(df) - 1)
    print(f"\nStage 1: top-{pool} for {len(df)} characters")
    neighbour_idx, neighbour_score = search_top_k(embeddings, pool)

    row_of = {u: i for i, u in enumerate(unicodes)}
    before = quality_metrics(neighbour_idx[:, :args.top_k], radicals, strokes, components)

    print(f"Reranking {len(df)} queries...")
    reranked = np.empty((len(df), args.top_k), dtype=np.int64)
    for i, u in enumerate(unicodes):
        candidates = [(unicodes[j], float(s))
                      for j, s in zip(neighbour_idx[i], neighbour_score[i])]
        ranked = reranker.rerank(u, candidates, top_k=args.top_k)
        reranked[i] = [row_of[r.unicode] for r in ranked]
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{len(df)}", flush=True)
    after = quality_metrics(reranked, radicals, strokes, components)

    # 24 = len("ids_jaccard@k [CIRCULAR]"), the widest label this loop can print. At the old width
    # of 16 every flagged row overflowed and shunted the numbers right, so the one table whose
    # whole job is to be read sceptically was the one that came out misaligned.
    label_width = max(len(m) + len(" [CIRCULAR]") for m in
                      ("radical@k", "stroke_mae@k", "ids_jaccard@k"))
    print(f"\n{'metric':<{label_width}s} {'stage 1':>10s} {'reranked':>10s} {'change':>10s}")
    rows = []
    for metric in ("radical@k", "stroke_mae@k", "ids_jaccard@k"):
        a, b = before[metric], after[metric]
        flag = " [CIRCULAR]" if metric in tainted else ""
        print(f"{metric + flag:<{label_width}s} {a:10.4f} {b:10.4f} {b - a:+10.4f}")
        rows.append({"metric": metric + flag, "stage1": round(a, 4),
                     "reranked": round(b, 4), "change": round(b - a, 4)})

    print("\nRead this as a demonstration, not a result. Every metric above is computed from a"
          "\ncolumn the reranker was given as an input feature, so an improvement here is the"
          "\nreranker agreeing with itself. The honest number is --suite render.")

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, f"rerank_proxy_{tag}_{args.normalize}.xlsx")
    with pd.ExcelWriter(out) as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="Circular", index=False)
        pd.DataFrame([{"warning": "Every metric in this file is circular: the reranker uses "
                                  "RADICAL/STROKE_NUM/SHAPE_MORPH as input features and these "
                                  "metrics are derived from those same columns. Do not report. "
                                  "See evaluate_rerank.py --suite render."}]
                     ).to_excel(writer, sheet_name="Warnings", index=False)
    print(f"Wrote {out}")


# --- suite: render (the honest number) --------------------------------------------------------

def load_manifest(rendered_dir, augmented, sample, seed):
    """Query rows from ``render_fonts.py``'s manifest: rendered views of a known codepoint.

    The label is the Unicode codepoint, which is independent of RADICAL/STROKE_NUM/SHAPE_MORPH --
    that independence is the entire reason this suite can measure the reranker honestly.
    """
    path = os.path.join(rendered_dir, "manifest.csv")
    if not os.path.exists(path):
        raise SystemExit(f"No {path} -- run: python render_fonts.py --augment 2")
    manifest = pd.read_csv(path, dtype={"unicode": str})
    queries = manifest[manifest["view"] != "corpus"]
    queries = queries[queries["aug"] > 0] if augmented else queries[queries["aug"] == 0]
    if queries.empty:
        raise SystemExit(
            "No matching rows in the manifest"
            + (" -- re-run render_fonts.py with --augment N." if augmented else "."))
    queries = queries[queries["path"].map(os.path.exists)]
    if sample and sample < len(queries):
        queries = queries.sample(n=sample, random_state=seed)
    return queries.reset_index(drop=True)


def embed_queries(paths, keys, args, tag):
    """Embed the sampled render views, caching so the four configurations embed only once."""
    from feature_extractors import build_extractor

    cache = os.path.join(
        args.output_dir,
        f"embeddings_render_{tag}_{'aug' if args.augment else 'clean'}_{len(paths)}.npz")
    if os.path.exists(cache) and not args.refresh:
        cached = np.load(cache, allow_pickle=True)
        if list(cached["keys"]) == list(keys):
            print(f"Reused {len(paths)} query embeddings from {cache}")
            return cached["embeddings"].astype("float32")
        print(f"{cache} does not match this sample -- recomputing")

    extractor = build_extractor(args.backend, device=args.device, dtype=args.dtype)
    print(f"Embedding {len(paths)} rendered queries with '{args.backend}'")
    embeddings = extractor.embed_paths(paths, batch_size=args.batch_size, progress_every=20)
    np.savez(cache, embeddings=embeddings, keys=np.array(keys, dtype=object))
    print(f"Cached to {cache}")
    return embeddings.astype("float32")


def suite_render(args):
    from search_all_chars_in_corpus import set_deterministic

    set_deterministic(args.seed)
    corpus, unicodes, tag = load_corpus_cache(args.output_dir, args.backend, args.dtype)
    in_corpus = set(unicodes)

    queries = load_manifest(args.rendered_dir, args.augment, args.sample, args.seed)
    keep = queries["unicode"].isin(in_corpus)
    if not keep.all():
        print(f"  {(~keep).sum()} query rows name a character absent from the corpus cache -- dropped")
    queries = queries[keep].reset_index(drop=True)
    print(f"Queries: {len(queries)} rendered views "
          f"({'degraded' if args.augment else 'clean'}), "
          f"{queries['unicode'].nunique()} distinct characters")

    embeddings = embed_queries(queries["path"].tolist(),
                               (queries["unicode"] + "|" + queries["path"]).tolist(), args, tag)

    reranker = None if args.no_rerank else build_reranker(args)
    if args.oracle_meta:
        print("\n  *** --oracle-meta: query metadata is read from the LABEL. This is a\n"
              "      deliberate leak measuring the ceiling of infer_query_meta.\n"
              "      It is a diagnostic. Never report it as a result. ***")

    pool = min(args.pool, len(unicodes))
    row_of = {u: i for i, u in enumerate(unicodes)}
    ranks, base_ranks, margins, radical_hits = [], [], [], []

    print(f"\nRetrieving top-{pool} and reranking...")
    for i in range(len(queries)):
        truth = queries["unicode"].iloc[i]
        scores = corpus @ embeddings[i]
        top = np.argpartition(-scores, pool - 1)[:pool]
        top = top[np.argsort(-scores[top])]
        candidates = [(unicodes[j], float(scores[j])) for j in top]

        found = [r for r, (u, _) in enumerate(candidates) if u == truth]
        base_ranks.append(found[0] + 1 if found else None)

        if reranker is None:
            ranks.append(base_ranks[-1])
        else:
            ordered, meta = rerank_one(reranker, args, candidates, candidates, truth)
            hit = [r for r, u in enumerate(ordered) if u == truth]
            ranks.append(hit[0] + 1 if hit else None)
            margins.append(meta.margin)
            if not args.oracle_meta:
                radical_hits.append(meta.radical == reranker.radical.get(truth))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(queries)}", flush=True)

    n = len(ranks)
    print(f"\n=== Suite render: {args.backend}/{args.dtype}, "
          f"{'degraded' if args.augment else 'clean'} views ===")
    report_hits("stage 1 (visual only)", base_ranks, n)
    if reranker is not None:
        label = "reranked [ORACLE METADATA - NOT A RESULT]" if args.oracle_meta else \
                f"reranked (normalize={args.normalize})"
        report_hits(label, ranks, n)
        moved = sum(1 for a, b in zip(base_ranks, ranks) if a != b)
        print(f"\n    candidates whose rank changed: {moved}/{n} = {moved / n:.4f}")
        if radical_hits:
            acc = float(np.mean(radical_hits))
            lo, hi = wilson_ci(sum(radical_hits), len(radical_hits))
            print(f"    infer_query_meta radical accuracy: {acc:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
            print(f"    (queries where the vote was too split to call: "
                  f"{sum(1 for m in margins if m < args.min_margin)}/{n})")
            if acc < 0.40:
                print("    ! Below 0.40 the inferred radical is worse than useless -- the rerank\n"
                      "      is being steered by a wrong guess. Fix inference before tuning weights.")


# --- suite: scans (final read-out) ------------------------------------------------------------

def load_labels(path, candidates_of, min_confidence, confidence_levels):
    """Read the human labels from ``label_template.csv``.

    Reimplemented here rather than imported: the original lives inside
    ``evaluate_test_images.main()`` (around line 325) and is not importable. Keep the two in step
    if the label format ever changes.
    """
    table = pd.read_csv(path)
    rank_of = {level: i for i, level in enumerate(confidence_levels)}
    floor = rank_of.get(min_confidence, len(confidence_levels))
    truth, dropped = {}, []
    for row in table.itertuples():
        candidates = candidates_of.get(row.file)
        if not candidates:
            continue
        confidence = str(getattr(row, "confidence", "") or "").strip().lower()
        if min_confidence and confidence in rank_of and rank_of[confidence] > floor:
            dropped.append(row.file)
            continue
        unicode_value = getattr(row, "correct_unicode", None)
        if isinstance(unicode_value, str) and unicode_value.strip():
            truth[row.file] = unicode_value.strip()
        elif not pd.isna(getattr(row, "correct_index", np.nan)):
            position = int(row.correct_index) - 1
            if 0 <= position < len(candidates):
                truth[row.file] = candidates[position]
    return truth, dropped


def suite_scans(args):
    from evaluate_test_images import CONFIDENCE_LEVELS, decode_filenames, ground_truth
    from feature_extractors import build_extractor
    from search_all_chars_in_corpus import CHAR_TABLE, set_deterministic
    from search_use_QuocNgu_mapping import QUOCNGU_TABLE, extract_histogram

    set_deterministic(args.seed)
    quocngu = pd.read_excel(QUOCNGU_TABLE)
    chars = pd.read_excel(CHAR_TABLE)
    corpus, unicodes, _ = load_corpus_cache(args.output_dir, args.backend, args.dtype)
    row_of = {u: i for i, u in enumerate(unicodes)}
    have_image = set(unicodes)

    matched, unmatched = decode_filenames(args.test_images, quocngu)
    print(f"Filenames decoded: {len(matched)} matched, {len(unmatched)} skipped")
    labelled = [(p, s, w, gt) for p, s, w in matched
                if (gt := ground_truth(quocngu, chars, w, have_image))]
    if not labelled:
        raise SystemExit("Nothing to evaluate.")

    candidates_of = {stem: sorted(gt) for _, stem, _, gt in labelled}
    truth, dropped = load_labels(args.labels, candidates_of, args.min_confidence,
                                 CONFIDENCE_LEVELS)
    if dropped:
        print(f"Dropped by --min-confidence {args.min_confidence}: {len(dropped)} "
              f"({', '.join(sorted(dropped))})")
    if not truth:
        raise SystemExit(f"No usable labels in {args.labels}.")

    extractor = build_extractor(args.backend, device=args.device, dtype=args.dtype)
    queries = extractor.embed_paths([p for p, _, _, _ in labelled], batch_size=args.batch_size)

    reranker = build_reranker(args)
    if args.oracle_meta:
        print("\n  *** --oracle-meta reads the answer's metadata off the label. Diagnostic only. ***")

    emb_ok = rerank_ok = hist_ok = 0
    emb_ranks, rerank_ranks, hist_ranks = [], [], []
    radical_hits, scored = [], 0

    for i, (path, stem, word, gt) in enumerate(labelled):
        if stem not in truth:
            continue
        scored += 1
        gold = truth[stem]
        group = sorted(gt)

        # Part 2: rank inside the same-reading group.
        group_scores = corpus[[row_of[u] for u in group]] @ queries[i]
        candidates = sorted(zip(group, (float(s) for s in group_scores)),
                            key=lambda kv: -kv[1])
        emb_order = [u for u, _ in candidates]

        # Part 1 whole-corpus neighbours -- the evidence infer_query_meta needs. The same-reading
        # group is useless for that: its members share a reading, not a shape.
        corpus_scores = corpus @ queries[i]
        top = np.argpartition(-corpus_scores, args.infer_top_n)[:args.infer_top_n]
        top = top[np.argsort(-corpus_scores[top])]
        neighbours = [(unicodes[j], float(corpus_scores[j])) for j in top]

        rerank_order, meta = rerank_one(reranker, args, candidates, neighbours, gold)
        if not args.oracle_meta:
            radical_hits.append(meta.radical == reranker.radical.get(gold))

        query_hist = extract_histogram(path)
        distances = [np.linalg.norm(query_hist - extract_histogram(
            os.path.join(args.images, f"{u}.jpg"))) for u in group]
        hist_order = [group[j] for j in np.argsort(distances)]

        for order, ranks in ((emb_order, emb_ranks), (rerank_order, rerank_ranks),
                             (hist_order, hist_ranks)):
            ranks.append(order.index(gold) + 1 if gold in order else None)
        emb_ok += emb_order[0] == gold
        rerank_ok += rerank_order[0] == gold
        hist_ok += hist_order[0] == gold

    n = scored
    print(f"\n=== Suite scans: Part 2, same-reading group, {args.backend}/{args.dtype} ===")
    print(f"Labels: {args.labels}"
          + (f", min-confidence {args.min_confidence}" if args.min_confidence else ""))
    baseline = float(np.mean([1 / len(gt) for _, stem, _, gt in labelled if stem in truth]))
    for label, ok, ranks in (("embedding", emb_ok, emb_ranks),
                             ("embedding + rerank", rerank_ok, rerank_ranks),
                             ("histogram", hist_ok, hist_ranks)):
        lo, hi = wilson_ci(ok, n)
        mrr = float(np.mean([1 / r for r in ranks if r])) if any(ranks) else 0.0
        print(f"  {label:<20s} top-1 {ok:2d}/{n} = {ok / n:.4f}  "
              f"95% CI [{lo:.4f}, {hi:.4f}] (+/-{(hi - lo) / 2 * 100:.1f} pts)  MRR {mrr:.4f}")
    print(f"  {'random in group':<20s}            = {baseline:.4f}")

    if radical_hits:
        acc = float(np.mean(radical_hits))
        lo, hi = wilson_ci(sum(radical_hits), len(radical_hits))
        print(f"\n  infer_query_meta radical accuracy: {acc:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

    lo, hi = wilson_ci(rerank_ok, n)
    print(f"\n  At n={n} the 95% CI is about +/-{(hi - lo) / 2 * 100:.0f} points, so a difference"
          f"\n  of {abs(rerank_ok - emb_ok)} image(s) between methods is not evidence either way."
          f"\n  Report this number with its interval; do not tune against it.")


# --- CLI --------------------------------------------------------------------------------------

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", required=True, choices=SUITES)
    parser.add_argument("--backend", default="chinese-clip-large")
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--images", default="./images")
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--normalize", default="zscore", choices=NORMALIZERS)
    parser.add_argument("--weights", default=None, help="e.g. 'visual=0.5,radical=0.3'")
    parser.add_argument("--no-rerank", action="store_true",
                        help="stage-1 baseline only (render suite)")
    parser.add_argument("--pool", type=int, default=100,
                        help="stage-1 candidates handed to the reranker")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--infer-top-n", type=int, default=50,
                        help="corpus-wide neighbours voting on the query's metadata")
    parser.add_argument("--min-margin", type=float, default=0.15,
                        help="below this vote margin the radical signal is dropped")
    parser.add_argument("--oracle-meta", action="store_true",
                        help="DIAGNOSTIC: read query metadata from the label. Cheating on purpose")

    parser.add_argument("--limit", type=int, default=None, help="proxy suite: match the cache")
    parser.add_argument("--rendered-dir", default="./output/rendered")
    parser.add_argument("--sample", type=int, default=5000,
                        help="render suite: query views to sample (0 = all)")
    parser.add_argument("--augment", action="store_true",
                        help="render suite: use the degraded views, which are far closer to "
                             "real scans than clean renders are")
    parser.add_argument("--refresh", action="store_true", help="ignore cached query embeddings")

    parser.add_argument("--test-images", default="./test_images")
    parser.add_argument("--labels", default="./output/label_sheets/label_template.csv")
    parser.add_argument("--min-confidence", default=None, choices=("high", "med", "low"))
    args = parser.parse_args()

    if args.oracle_meta and args.no_rerank:
        raise SystemExit("--oracle-meta and --no-rerank contradict each other.")
    if args.pool <= args.top_k:
        raise SystemExit(f"--pool ({args.pool}) must exceed --top-k ({args.top_k}).")

    random.seed(args.seed)
    {"proxy": suite_proxy, "render": suite_render, "scans": suite_scans}[args.suite](args)


if __name__ == "__main__":
    main()

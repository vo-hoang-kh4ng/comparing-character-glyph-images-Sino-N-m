"""Metadata-based reranking layer for Sino-Nom glyph similarity search.

Stage 1 (``feature_extractors.py`` + Faiss, or the histogram search in Part 2) proposes candidates
from pixels alone. This module rescores those candidates using the linguistic metadata in
``final_characteristics-v2.xlsx`` that the image models cannot see: radical, stroke count, and the
IDS decomposition in ``SHAPE_MORPH``.

Usage -- query is a KNOWN corpus character (Part 1, corpus exploration)::

    from rerank import Reranker

    reranker = Reranker.from_excel()                     # reads final_characteristics-v2.xlsx
    ranked = reranker.rerank("4E00", candidates, top_k=20)

Usage -- query is an IMAGE of unknown identity (a real scan, a rendered glyph)::

    meta = reranker.infer_query_meta(corpus_neighbours)  # [(unicode, cosine)], top-50 CORPUS-WIDE
    ranked = reranker.rerank(meta, candidates, top_k=20)

where ``candidates`` is a list of ``(unicode, visual_similarity)`` pairs and the result is a list of
``RerankedCandidate`` sorted by combined score, descending.

Four design decisions, each fixing a defect found in review. The labels B1-B4 are used in the
review notes, in ``test_rerank.py`` and in ``evaluate_rerank.py``:

B1  The three weak-label metrics in ``benchmark_extractors.py`` (``radical@k``, ``stroke_mae@k``,
    ``ids_jaccard@k``) are derived from the very columns this module uses as features. Reranking
    and then reporting them is circular -- ``radical@k`` would climb toward its 0.978 ceiling
    while meaning nothing. ``circularity_report()`` names the tainted metrics so callers can
    refuse to print them; honest numbers come from ``evaluate_rerank.py --suite render``.

B2  Signals are standardized WITHIN each candidate list before mixing (``normalize='zscore'``).
    Mixing raw scores does not work: with the default weights a candidate must win visual
    similarity by more than 0.364 cosine to overcome a single radical mismatch, and top-20
    neighbour cosines never span that. Raw mixing is therefore a lexicographic sort -- radical,
    then stroke, then shape -- with the image model demoted to a tie-breaker. ``normalize='none'``
    reproduces that old behaviour and is kept only as a baseline to measure the fix against.

B3  A query image of unknown identity has no metadata -- and that is exactly the case for
    ``test_images/`` and for rendered glyphs. Passing ``None`` used to make every metadata signal
    drop out, silently turning the reranker into a no-op. It now raises. Use
    ``infer_query_meta()`` to vote the query's metadata out of its corpus-wide visual neighbours.

B4  ``load_corpus()`` in ``search_all_chars_in_corpus.py`` returns only UNICODE/CHAR/path, so
    ``Reranker(df)`` on that frame used to die with ``KeyError: 'RADICAL'``. Use
    ``Reranker.from_excel()``; the constructor validates columns and says so.

API note: ``score()`` is gone. It scored one pair in isolation, which B2 makes meaningless. Its
replacement is ``raw_signals()`` (unnormalized, for debugging) plus ``rerank()`` (two-pass).

Import weight: this module deliberately imports only the standard library and pandas. Do NOT
import ``search_all_chars_in_corpus`` at module level -- it pulls in torch and faiss (its lines
26-29), which would make the pure-logic unit tests in ``test_rerank.py`` need a 200 MB torch
install. Heavy imports live inside ``main()``.
"""

import argparse
import math
import os
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field

import pandas as pd

#: Master character table. Declared here rather than imported from ``search_all_chars_in_corpus``
#: on purpose -- see the import-weight note in the module docstring.
CHAR_TABLE = "final_characteristics-v2.xlsx"

# Ideographic Description Characters: the structural operators in an IDS string
# (e.g. the leading char of '⿰木口'). Stripped out before treating the rest of
# the string as a bag of components.
IDC_CHARS = set('⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻')

# 'dai 大 (+1 net)' -> reading, radical glyph, added stroke count.
# Verified to match all 31073 non-null RADICAL values in final_characteristics-v2.xlsx.
RADICAL_RE = re.compile(r'^(\S+)\s+(\S+)\s*\(\+?(-?\d+)\s*nét\)$')

#: The signals mixed into the final score, in a fixed order so output columns stay stable.
SIGNALS = ('visual', 'radical', 'stroke', 'shape')

# Relative contribution of each signal. Tune these against a NON-CIRCULAR metric only
# (evaluate_rerank.py --suite render), never against radical@k -- see B1.
DEFAULT_WEIGHTS = {
    'visual': 0.55,   # cosine similarity carried over from stage 1
    'radical': 0.20,  # same Kangxi radical or not
    'stroke': 0.15,   # closeness in total stroke count
    'shape': 0.10,    # IDF-weighted overlap of IDS components
}

#: Which spreadsheet column feeds which signal. Drives ``circularity_report()``.
FEATURE_COLUMNS = {'radical': 'RADICAL', 'stroke': 'STROKE_NUM', 'shape': 'SHAPE_MORPH'}

#: Weak-label metrics computed by ``benchmark_extractors.py`` and the column each is derived from.
#: A metric whose column is also a rerank feature is circular -- see B1.
CIRCULAR_METRICS = {
    'radical@k': 'RADICAL',
    'stroke_mae@k': 'STROKE_NUM',
    'ids_jaccard@k': 'SHAPE_MORPH',
}

#: ``radical@20`` cannot exceed this (measured from the radical group-size distribution: 30394 of
#: 31073 characters have at least 20 same-radical siblings). A reranked run approaching it is
#: measuring its own features, not retrieval quality.
RADICAL_AT_20_CEILING = 0.978

REQUIRED_COLUMNS = ('UNICODE', 'CHAR', 'RADICAL', 'STROKE_NUM', 'SHAPE_MORPH')

NORMALIZERS = ('none', 'zscore', 'rrf')


def parse_radical(value):
    """Extract the radical glyph from a RADICAL cell. Returns None if unparseable.

    Returns the glyph ('一'), not reading+glyph ('nhất 一'). The readings are ambiguous: the table
    has 214 distinct radical glyphs but only 193 distinct readings, because spellings like
    'thuỷ'/'thủ' collapse different radicals onto one string.
    """
    if not isinstance(value, str):
        return None
    match = RADICAL_RE.match(value.strip())
    return match.group(2) if match else None


def parse_components(value):
    """Split an IDS string into its component multiset, dropping the IDC operators.

    '⿵大丶' -> Counter({'大': 1, '丶': 1}). A Counter rather than a set so that 林 = '⿰木木'
    stays distinguishable from 木 (502 characters in the table repeat a component). Counter's
    ``|`` and ``&`` are multiset union and intersection, which is exactly what ``_shape_sim``
    needs, and they work with float weights too -- that is how ``infer_query_meta`` returns a
    soft bag.

    Returns None when SHAPE_MORPH is absent, which is the case for 1698 characters (mostly
    atomic ones like 大, 木 that do not decompose).
    """
    if not isinstance(value, str) or not value.strip():
        return None
    components = Counter(ch for ch in value.strip() if ch not in IDC_CHARS)
    return components or None


def _softmax(values):
    """Numerically stable softmax over a plain list."""
    peak = max(values)
    exps = [math.exp(v - peak) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def _weighted_median(pairs):
    """Weighted median of [(value, weight)]. Robust to the outlier a hub match introduces."""
    pairs = sorted((v, w) for v, w in pairs if w > 0)
    if not pairs:
        return None
    half = sum(w for _, w in pairs) / 2.0
    running = 0.0
    for value, weight in pairs:
        running += weight
        if running >= half:
            return float(value)
    return float(pairs[-1][0])


@dataclass
class QueryMeta:
    """The query side of a comparison.

    Either read off a known corpus character (``inferred=False``) or voted out of that query
    image's visual neighbours (``inferred=True``) -- see B3.
    """
    radical: str = None
    stroke_num: float = None
    components: Counter = None
    inferred: bool = False
    #: Vote margin between the top two radicals. 0.0 when not inferred.
    margin: float = 0.0
    #: Full radical vote tally, for debugging why a query was assigned its radical.
    votes: dict = field(default_factory=dict)


@dataclass
class RerankedCandidate:
    unicode: str
    char: str
    score: float
    visual_sim: float
    #: Raw per-signal values (None where metadata is missing), for debugging and the report.
    parts: dict = field(default_factory=dict)
    #: The same signals after within-list normalization -- this is what actually drove the
    #: ranking, so it is what explains a surprising order.
    parts_z: dict = field(default_factory=dict)


class Reranker:
    def __init__(self, df_meta, weights=None, normalize='zscore', rrf_k=60):
        """df_meta: a DataFrame carrying REQUIRED_COLUMNS -- see ``from_excel``.

        normalize: 'zscore' (default, see B2), 'rrf', or 'none' (the pre-B2 behaviour, kept as a
        measurement baseline).
        """
        if normalize not in NORMALIZERS:
            raise ValueError(f"normalize must be one of {NORMALIZERS}, got {normalize!r}")

        missing = [c for c in REQUIRED_COLUMNS if c not in df_meta.columns]
        if missing:
            raise ValueError(
                f"df_meta is missing {missing}. If this frame came from "
                f"search_all_chars_in_corpus.load_corpus() it only carries UNICODE/CHAR/path -- "
                f"use Reranker.from_excel() instead (B4)."
            )

        self.weights = dict(weights) if weights else dict(DEFAULT_WEIGHTS)
        self.normalize = normalize
        self.rrf_k = rrf_k

        unicodes = df_meta['UNICODE'].astype(str)
        self.char = dict(zip(unicodes, df_meta['CHAR']))
        self.radical = {u: parse_radical(v) for u, v in zip(unicodes, df_meta['RADICAL'])}
        self.components = {u: parse_components(v) for u, v in zip(unicodes, df_meta['SHAPE_MORPH'])}

        strokes = pd.to_numeric(df_meta['STROKE_NUM'], errors='coerce')
        self.stroke_num = {
            u: (None if pd.isna(v) else float(v))
            for u, v in zip(unicodes, strokes)
        }

        self.idf = self._build_idf()
        self._default_idf = max(self.idf.values()) if self.idf else 1.0

    @classmethod
    def from_excel(cls, path=CHAR_TABLE, **kwargs):
        """Build straight from the master table -- the correct entry point (B4).

        ``load_corpus()`` returns only UNICODE/CHAR/path, so passing its frame to ``Reranker(df)``
        raises. Reading only REQUIRED_COLUMNS also avoids the 31208x10 full-table copy the old
        constructor made.
        """
        return cls(pd.read_excel(path, usecols=list(REQUIRED_COLUMNS)), **kwargs)

    # --- circularity guard (B1) -------------------------------------------------------------

    def active_feature_columns(self):
        """Spreadsheet columns that actually move the ranking (weight > 0)."""
        return {col for signal, col in FEATURE_COLUMNS.items() if self.weights.get(signal, 0.0) > 0}

    def circularity_report(self):
        """Weak-label metrics invalidated by this configuration: {metric: source column}.

        Reranking on RADICAL and then reporting radical@k measures the reranker against its own
        input. Callers must refuse to publish anything this returns -- see B1 and
        ``evaluate_rerank.py --suite proxy``, which exists to demonstrate the effect rather than
        to produce a result.
        """
        used = self.active_feature_columns()
        return {metric: col for metric, col in CIRCULAR_METRICS.items() if col in used}

    def _build_idf(self):
        """Inverse document frequency over IDS components.

        A character is a 'document', its components are the 'terms'. Sharing a rare component is
        far stronger evidence of similarity than sharing a common one (口 occurs in 1520 of the
        29510 decomposable characters; 1303 components occur exactly once), so plain Jaccard
        overlap would over-reward the common cases.
        """
        doc_freq = Counter()
        n_docs = 0
        for components in self.components.values():
            if components:
                n_docs += 1
                # .keys(), not the Counter itself: document frequency counts each character once,
                # so 林 = '⿰木木' must contribute 1 to 木, not 2.
                doc_freq.update(components.keys())
        if not n_docs:
            return {}
        return {term: math.log(n_docs / freq) for term, freq in doc_freq.items()}

    def _shape_sim(self, a, b):
        """IDF-weighted multiset Jaccard between two component bags. None if either is missing."""
        if not a or not b:
            return None
        # Unseen components fall back to the maximum IDF: absent from the corpus
        # means maximally rare, not worthless.
        union = sum(self.idf.get(t, self._default_idf) * c for t, c in (a | b).items())
        if union == 0:
            return None
        intersection = sum(self.idf.get(t, self._default_idf) * c for t, c in (a & b).items())
        return intersection / union

    def _stroke_sim(self, a, b):
        """Soft penalty on stroke-count difference: 0 -> 1.0, 1 -> 0.5, 2 -> 0.33."""
        if a is None or b is None:
            return None
        return 1.0 / (1.0 + abs(a - b))

    def _radical_sim(self, a, b):
        if a is None or b is None:
            return None
        return 1.0 if a == b else 0.0

    # --- query side (B3) --------------------------------------------------------------------

    def meta_for(self, unicode_value):
        """Metadata of a known corpus character."""
        u = str(unicode_value)
        if u not in self.char:
            raise KeyError(
                f"{u!r} is not in the character table. An unknown UNICODE would leave every "
                f"metadata signal empty and silently disable the reranker (B3)."
            )
        return QueryMeta(
            radical=self.radical.get(u),
            stroke_num=self.stroke_num.get(u),
            components=self.components.get(u),
            inferred=False,
        )

    def infer_query_meta(self, neighbours, temperature=0.05, min_margin=0.15, top_n=50):
        """Vote the query's metadata out of its corpus-wide visual neighbours (B3).

        neighbours: [(unicode, cosine)] -- the top-N of a WHOLE-CORPUS search, N ~ 50.

        This is pseudo-relevance feedback. It is not circular: the inputs are image cosines plus
        the CANDIDATES' metadata; the query's own metadata never appears, which is the whole
        point -- for a real scan nobody knows it.

        Feed it whole-corpus neighbours, never a Part 2 same-reading group: those ~7 characters
        share a Quoc Ngu reading and have no reason at all to share a radical, so voting inside
        them is noise.

        A radical whose vote margin falls below ``min_margin`` returns None, dropping the radical
        signal for that query entirely. Declining to rerank beats reranking toward a guess.
        """
        neighbours = [(str(u), float(s)) for u, s in neighbours][:top_n]
        if not neighbours:
            raise ValueError("infer_query_meta needs at least one neighbour")

        weights = _softmax([s / temperature for _, s in neighbours])

        votes = Counter()
        for (u, _), w in zip(neighbours, weights):
            radical = self.radical.get(u)
            if radical:
                votes[radical] += w
        radical, margin = None, 0.0
        if votes:
            ranked = votes.most_common(2)
            margin = ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0.0)
            if margin >= min_margin:
                radical = ranked[0][0]

        stroke = _weighted_median([
            (self.stroke_num[u], w) for (u, _), w in zip(neighbours, weights)
            if self.stroke_num.get(u) is not None
        ])

        components = Counter()
        for (u, _), w in zip(neighbours, weights):
            bag = self.components.get(u)
            if bag:
                for term, count in bag.items():
                    components[term] += w * count
        if components:
            # Rescale so the strongest component weighs 1.0, matching the integer counts a real
            # decomposition carries. Without this the soft bag is uniformly tiny and the min/max
            # multiset Jaccard in _shape_sim collapses toward zero for every candidate alike.
            peak = max(components.values())
            components = Counter({t: v / peak for t, v in components.items()})
        else:
            components = None

        return QueryMeta(radical=radical, stroke_num=stroke, components=components,
                         inferred=True, margin=margin, votes=dict(votes))

    def _resolve_query(self, query):
        if isinstance(query, QueryMeta):
            return query
        if query is None:
            raise ValueError(
                "rerank() needs query metadata. Passing None drops every metadata signal and "
                "turns the reranker into a no-op that still looks like it ran (B3). For a query "
                "image of unknown identity, call infer_query_meta(corpus_neighbours) first."
            )
        return self.meta_for(query)

    # --- scoring ----------------------------------------------------------------------------

    def raw_signals(self, query_meta, candidate_unicode, visual_sim):
        """Unnormalized per-signal values for one pair. None where metadata is missing.

        Not a ranking score on its own -- mixing these raw is exactly the defect B2 fixes. Use it
        to explain a ranking, not to produce one.
        """
        u = str(candidate_unicode)
        # Map cosine [-1, 1] onto [0, 1] instead of clamping negatives to 0. The old clamp was
        # written for ResNet18, whose post-ReLU features never go negative; the current default
        # backend is Chinese-CLIP, which is not post-ReLU, and clamping collapsed every
        # negative-similarity candidate onto an identical 0.0.
        visual = (max(-1.0, min(1.0, float(visual_sim))) + 1.0) / 2.0
        return {
            'visual': visual,
            'radical': self._radical_sim(query_meta.radical, self.radical.get(u)),
            'stroke': self._stroke_sim(query_meta.stroke_num, self.stroke_num.get(u)),
            'shape': self._shape_sim(query_meta.components, self.components.get(u)),
        }

    @staticmethod
    def _standardize(values):
        """z-score across one candidate list; None -> 0.0, i.e. that list's own mean.

        Two properties fall out, both wanted:

        - A candidate missing SHAPE_MORPH is scored at the expected value of that signal rather
          than dropped from the mixture. The old code renormalized the remaining weights, which
          made scores from candidates with different metadata coverage incomparable.
        - A signal every candidate agrees on has zero variance and contributes nothing. This is
          common in Part 2, where a same-reading group can share one radical outright.
        """
        present = [v for v in values if v is not None]
        if len(present) < 2:
            return [0.0] * len(values)
        mean = statistics.fmean(present)
        std = statistics.pstdev(present)
        if std == 0.0:
            return [0.0] * len(values)
        return [0.0 if v is None else (v - mean) / std for v in values]

    def _reciprocal_rank(self, values):
        """Reciprocal Rank Fusion: drop magnitudes, keep order only. Missing -> ranked last.

        More robust than z-score when a signal has outliers, less discriminative on short
        candidate lists -- with the standard rrf_k=60 a 7-candidate Part 2 group spreads over
        only 1/61..1/67, so lower rrf_k if you use this there.
        """
        order = sorted(
            range(len(values)),
            key=lambda i: (values[i] is None, -(values[i] if values[i] is not None else 0.0)),
        )
        out = [0.0] * len(values)
        for rank, i in enumerate(order, start=1):
            out[i] = 0.0 if values[i] is None else 1.0 / (self.rrf_k + rank)
        return out

    def _normalize_signal(self, values):
        if self.normalize == 'zscore':
            return self._standardize(values)
        if self.normalize == 'rrf':
            return self._reciprocal_rank(values)
        return [0.0 if v is None else v for v in values]

    def rerank(self, query, candidates, top_k=None, exclude_query=True):
        """Rescore and reorder stage-1 candidates.

        query:      UNICODE of a known corpus character, or a QueryMeta from infer_query_meta().
                    None raises -- see B3.
        candidates: iterable of (unicode, visual_similarity).

        Pass in more candidates than you intend to keep -- reranking a list that is already
        trimmed to top_k can only shuffle it, never pull a better match in.

        Two passes, because normalizing a signal requires seeing the whole candidate list (B2).
        With normalize='zscore' the returned score is a weighted sum of z-scores: a ranking score
        centred near zero, NOT a similarity in [0, 1].
        """
        meta = self._resolve_query(query)
        pairs = [(str(u), float(v)) for u, v in candidates]
        if exclude_query and not isinstance(query, QueryMeta):
            pairs = [(u, v) for u, v in pairs if u != str(query)]
        if not pairs:
            return []

        raw = {signal: [] for signal in SIGNALS}
        for u, visual in pairs:
            signals = self.raw_signals(meta, u, visual)
            for signal in SIGNALS:
                raw[signal].append(signals[signal])

        normalized = {signal: self._normalize_signal(raw[signal]) for signal in SIGNALS}

        results = []
        for i, (u, visual) in enumerate(pairs):
            score = sum(self.weights.get(s, 0.0) * normalized[s][i] for s in SIGNALS)
            results.append(RerankedCandidate(
                unicode=u,
                char=self.char.get(u, ''),
                score=score,
                visual_sim=visual,
                parts={s: raw[s][i] for s in SIGNALS},
                parts_z={s: normalized[s][i] for s in SIGNALS},
            ))

        # UNICODE as the tie-break keeps the order reproducible across runs and machines, the
        # same promise set_deterministic() makes for everything else in this project.
        results.sort(key=lambda r: (-r.score, r.unicode))
        return results[:top_k] if top_k else results


def warn_circular(tainted):
    """Banner naming the metrics a configuration invalidates (B1)."""
    line = "=" * 78
    body = "\n".join(f"    {metric:<16s} derived from {col}"
                     for metric, col in sorted(tainted.items()))
    return (
        f"\n{line}\n"
        f"  CIRCULAR METRICS -- {len(tainted)} weak label(s) are invalid for this configuration:\n"
        f"{body}\n"
        f"  The reranker uses these columns as input features, so scoring it with metrics\n"
        f"  derived from them measures it against itself. radical@20 will climb toward its\n"
        f"  {RADICAL_AT_20_CEILING} ceiling and mean nothing. Do not put these in the report.\n"
        f"  Honest numbers: evaluate_rerank.py --suite render\n"
        f"{line}"
    )


# --- CLI ------------------------------------------------------------------------------------

def parse_weights(text):
    """'visual=0.5,radical=0.3' -> the default weights with those two overridden."""
    weights = dict(DEFAULT_WEIGHTS)
    if not text:
        return weights
    for item in text.split(','):
        key, _, value = item.partition('=')
        key = key.strip()
        if key not in SIGNALS:
            raise SystemExit(f"unknown signal {key!r}; expected one of {SIGNALS}")
        weights[key] = float(value)
    return weights


def main():
    # Sino-Nom glyphs crash a cp1252 Windows console otherwise.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Rerank Part 1's corpus-wide top-K using character metadata.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", default="chinese-clip-large")
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--top-k", type=int, default=20, help="neighbours kept per character")
    parser.add_argument("--pool", type=int, default=100,
                        help="stage-1 candidates fed to the reranker (must exceed --top-k)")
    parser.add_argument("--normalize", default="zscore", choices=NORMALIZERS)
    parser.add_argument("--weights", default=None, help="e.g. 'visual=0.5,radical=0.3'")
    parser.add_argument("--images", default="./images")
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--limit", type=int, default=None,
                        help="match the --limit used when the embeddings were built")
    args = parser.parse_args()

    if args.pool <= args.top_k:
        raise SystemExit(
            f"--pool ({args.pool}) must exceed --top-k ({args.top_k}); reranking a list already "
            f"trimmed to top-k can only shuffle it."
        )

    # Heavy imports stay here: importing search_all_chars_in_corpus pulls in torch and faiss.
    import numpy as np
    from search_all_chars_in_corpus import cache_tag, load_corpus, search_top_k, set_deterministic

    set_deterministic()
    df = load_corpus(args.images, limit=args.limit)
    tag = cache_tag(args.backend, args.dtype, args.limit)
    cache_path = os.path.join(args.output_dir, f"embeddings_{tag}.npz")
    if not os.path.exists(cache_path):
        raise SystemExit(
            f"No {cache_path} -- run search_all_chars_in_corpus.py --backend {args.backend} first."
        )
    cached = np.load(cache_path, allow_pickle=True)
    embeddings = cached["embeddings"]
    unicodes = [str(u) for u in cached["unicodes"]]
    if unicodes != list(df["UNICODE"].astype(str)):
        raise SystemExit(
            f"{cache_path} was built from a different corpus -- pass a matching --limit."
        )

    reranker = Reranker.from_excel(weights=parse_weights(args.weights), normalize=args.normalize)
    tainted = reranker.circularity_report()
    if tainted:
        print(warn_circular(tainted))

    pool = min(args.pool, len(df) - 1)
    print(f"Stage 1: top-{pool} for {len(df)} characters")
    neighbour_idx, neighbour_score = search_top_k(embeddings, pool)

    print(f"Reranking (normalize={args.normalize}, weights={reranker.weights})")
    rows = []
    for i, u in enumerate(unicodes):
        candidates = [(unicodes[j], float(s))
                      for j, s in zip(neighbour_idx[i], neighbour_score[i])]
        ranked = reranker.rerank(u, candidates, top_k=args.top_k)
        rows.append({
            "Input Character": reranker.char.get(u, u),
            f"Top {args.top_k} Similar Characters": [r.char for r in ranked],
            "Rerank Scores": [round(r.score, 4) for r in ranked],
            "Similarity Scores": [round(r.visual_sim, 4) for r in ranked],
        })
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(unicodes)}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    out = pd.DataFrame(rows)
    stem = os.path.join(args.output_dir,
                        f"output_top_k_similar_char_{tag}_reranked_{args.normalize}")
    out.to_excel(f"{stem}.xlsx", index=False)
    out.to_csv(f"{stem}.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote {stem}.xlsx / .csv")
    print("\nThese are RESULTS, not an evaluation. For a number, run evaluate_rerank.py.")


if __name__ == "__main__":
    main()

"""Unit tests for ``rerank.py``.

The core suite runs on **pandas + pytest alone** -- no torch, no faiss, no glyph images, no GPU,
and no Excel. That is deliberate: ``rerank.py`` is pure logic, so its tests should not need the
200 MB of machine learning stack the rest of this project does. The fixture below is a
hand-built ten-character corpus whose radicals, stroke counts and IDS strings were chosen so the
expected ranking can be worked out on paper.

Tests marked ``data`` additionally read ``final_characteristics-v2.xlsx`` and need openpyxl. They
freeze the corpus statistics the review measured, so a later change to the parsing rules shows up
as a failing test instead of a silently different number::

    python -m pytest test_rerank.py -v            # core suite, no setup at all
    pip install openpyxl
    python -m pytest test_rerank.py -v -m data    # plus the real-table checks

Test names carry the B1-B4 labels from ``rerank.py``'s module docstring.
"""

import math
from collections import Counter

import pandas as pd
import pytest

from rerank import (
    CHAR_TABLE,
    DEFAULT_WEIGHTS,
    QueryMeta,
    RADICAL_RE,
    REQUIRED_COLUMNS,
    Reranker,
    parse_components,
    parse_radical,
)

# Column order is REQUIRED_COLUMNS: UNICODE, CHAR, RADICAL, STROKE_NUM, SHAPE_MORPH.
# (RADICAL before STROKE_NUM -- swapping them makes parse_radical see an int and to_numeric see
# a string, so both signals silently go None while the shape signal keeps working.)
#
# Chosen so that, for query Q001:
#   B001 matches on every metadata signal (same radical, same strokes, identical components)
#   A001 matches on none of them but can be given a higher visual similarity
# which is the exact configuration that exposes B2.
FIXTURE_ROWS = [
    ("Q001", "校", "mộc 木 (+6 nét)", 10, "⿰木交"),      # the query
    ("A001", "洨", "thuỷ 水 (+6 nét)", 10, "⿰⺡交"),     # different radical, shares only 交
    ("B001", "楌", "mộc 木 (+6 nét)", 10, "⿰木交"),      # same radical, identical components
    ("C001", "林", "mộc 木 (+4 nét)", 8, "⿰木木"),       # repeated component -> multiplicity
    ("D001", "木", "mộc 木 (+0 nét)", 4, None),           # atomic: no SHAPE_MORPH
    ("E001", "沐", "thuỷ 水 (+4 nét)", 7, "⿰⺡木"),
    ("F001", "咬", "khẩu 口 (+6 nét)", 9, "⿰口交"),
    ("G001", "檄", "mộc 木 (+16 nét)", 20, "⿰木交"),      # same radical, far stroke count
    ("H001", "〓", None, None, None),                     # every metadata field missing
    ("I001", "江", "thuỷ 水 (+3 nét)", 6, "⿰⺡工"),
]

#: Component document frequencies implied by FIXTURE_ROWS, over the 8 rows that decompose.
#: 木 is 5, not 6: 林 = '⿰木木' repeats it but is still a single document.
FIXTURE_DOCS = 8
FIXTURE_DF = {"木": 5, "交": 5, "⺡": 3, "口": 1, "工": 1}


def frame():
    return pd.DataFrame(FIXTURE_ROWS, columns=list(REQUIRED_COLUMNS))


def build(**kwargs):
    return Reranker(frame(), **kwargs)


def order(results):
    return [r.unicode for r in results]


# --- parsing --------------------------------------------------------------------------------

def test_parse_radical_returns_the_glyph_not_the_reading():
    # 214 distinct glyphs vs only 193 distinct readings in the real table, so the glyph is the
    # unambiguous half.
    assert parse_radical("mộc 木 (+6 nét)") == "木"
    assert parse_radical("nhất 一 (+3 nét)") == "一"


def test_parse_radical_rejects_non_strings_and_garbage():
    assert parse_radical(None) is None
    assert parse_radical(float("nan")) is None
    assert parse_radical("khong dung dinh dang") is None


def test_parse_components_keeps_multiplicity():
    # A set would collapse 林 onto {木} and make it indistinguishable from 木 itself.
    assert parse_components("⿰木木") == Counter({"木": 2})
    assert parse_components("⿰木交") == Counter({"木": 1, "交": 1})


def test_parse_components_returns_none_when_absent():
    assert parse_components(None) is None
    assert parse_components("") is None
    assert parse_components("   ") is None
    # An IDS string of nothing but operators leaves no components behind.
    assert parse_components("⿰") is None


# --- IDF ------------------------------------------------------------------------------------

def test_idf_weighs_rare_components_more():
    idf = build().idf
    assert idf["口"] > idf["⺡"] > idf["木"]


def test_idf_document_frequency_counts_each_character_once():
    # 林 = '⿰木木' contributes 1 to 木's document frequency, not 2. Counting the multiset here
    # would inflate df and quietly deflate the IDF of every repeated component.
    idf = build().idf
    for term, freq in FIXTURE_DF.items():
        assert idf[term] == pytest.approx(math.log(FIXTURE_DOCS / freq))


# --- B4: constructor rejects the wrong frame ------------------------------------------------

def test_b4_load_corpus_frame_is_rejected_with_a_useful_message():
    # This is exactly what search_all_chars_in_corpus.load_corpus() returns.
    wrong = pd.DataFrame({"UNICODE": ["Q001"], "CHAR": ["校"], "path": ["./images/Q001.jpg"]})
    with pytest.raises(ValueError) as excinfo:
        Reranker(wrong)
    message = str(excinfo.value)
    assert "RADICAL" in message
    assert "from_excel" in message


def test_b4_unknown_normalizer_is_rejected():
    with pytest.raises(ValueError):
        build(normalize="minmax")


# --- B3: a query with no metadata must not silently disable the reranker ---------------------

def test_b3_none_query_raises_instead_of_becoming_a_noop():
    with pytest.raises(ValueError) as excinfo:
        build().rerank(None, [("A001", 0.9), ("B001", 0.8)])
    assert "no-op" in str(excinfo.value)


def test_b3_unknown_unicode_raises():
    with pytest.raises(KeyError):
        build().rerank("NOPE", [("A001", 0.9)])


# --- B2: the scale mismatch -----------------------------------------------------------------

def test_b2_raw_mixing_lets_the_radical_flag_override_visual_similarity():
    """normalize='none' is the pre-fix behaviour, kept only as a measurement baseline.

    A001 wins visual by 0.05 -- a wide margin by the standards of real top-20 cosines -- and
    still loses, because the radical flag is worth 0.20 of a score whose visual term can only
    move by 0.55 * 0.05 / 2 = 0.014.
    """
    ranked = build(normalize="none").rerank("Q001", [("A001", 0.95), ("B001", 0.90)])
    assert order(ranked) == ["B001", "A001"]


def test_b2_zscore_lets_visual_similarity_win_the_same_comparison():
    """Same inputs, same weights, only the normalization changes -- and the answer flips.

    Under z-score each signal is measured in standard deviations of *this candidate list*, so
    visual's 0.55 weight actually outranks radical's 0.20 plus shape's 0.10.
    """
    ranked = build(normalize="zscore").rerank("Q001", [("A001", 0.95), ("B001", 0.90)])
    assert order(ranked) == ["A001", "B001"]


def test_b2_raw_mixing_needs_an_impossible_cosine_gap_to_flip():
    """Locate the crossover of normalize='none' with only visual and radical in play.

    It sits at 0.727 raw cosine (0.20 / (0.55 / 2)). Real Chinese-CLIP top-20 neighbours span a
    few hundredths, so in practice the flag is never overridden: the mixture is a lexicographic
    sort wearing the costume of a weighted sum.
    """
    weights = {"visual": 0.55, "radical": 0.20, "stroke": 0.0, "shape": 0.0}
    reranker = build(normalize="none", weights=weights)

    below = reranker.rerank("Q001", [("A001", 0.95), ("B001", 0.25)])   # gap 0.70
    above = reranker.rerank("Q001", [("A001", 0.95), ("B001", 0.20)])   # gap 0.75
    assert order(below)[0] == "B001"
    assert order(above)[0] == "A001"


def test_b2_a_signal_every_candidate_agrees_on_contributes_nothing():
    # B001, C001 and G001 are all 木, so the radical signal has zero variance here and must not
    # tilt the ranking. Common in Part 2, where a same-reading group can share one radical.
    ranked = build(normalize="zscore").rerank(
        "Q001", [("B001", 0.90), ("C001", 0.80), ("G001", 0.70)])
    assert all(r.parts["radical"] == 1.0 for r in ranked)
    assert all(r.parts_z["radical"] == 0.0 for r in ranked)


def test_b2_missing_metadata_is_imputed_to_the_list_mean_not_dropped():
    # D001 has no SHAPE_MORPH. The old code renormalized the surviving weights, which made its
    # score incomparable with candidates that had all four signals. Now it sits at the mean.
    ranked = build(normalize="zscore").rerank(
        "Q001", [("B001", 0.90), ("A001", 0.85), ("D001", 0.80)])
    assert set(order(ranked)) == {"A001", "B001", "D001"}
    d001 = next(r for r in ranked if r.unicode == "D001")
    assert d001.parts["shape"] is None
    assert d001.parts_z["shape"] == 0.0


def test_b2_negative_cosines_keep_their_order():
    # The old clamp (max(v, 0.0)) was written for ResNet18's post-ReLU features. Chinese-CLIP is
    # not post-ReLU, and the clamp collapsed every negative-similarity candidate onto 0.0.
    weights = {"visual": 1.0, "radical": 0.0, "stroke": 0.0, "shape": 0.0}
    ranked = build(normalize="none", weights=weights).rerank(
        "Q001", [("A001", -0.90), ("B001", -0.50)])
    assert order(ranked) == ["B001", "A001"]
    assert ranked[0].parts["visual"] == pytest.approx(0.25)
    assert ranked[1].parts["visual"] == pytest.approx(0.05)


def test_b2_rrf_uses_rank_only_and_puts_missing_signals_last():
    reranker = build(normalize="rrf", rrf_k=60)
    assert reranker._reciprocal_rank([0.9, 0.1, None]) == pytest.approx(
        [1 / 61, 1 / 62, 0.0])
    # A hundredfold change in magnitude that preserves the order changes nothing.
    assert reranker._reciprocal_rank([90.0, 0.1, None]) == pytest.approx(
        [1 / 61, 1 / 62, 0.0])


# --- B3: inferring the query's metadata from its visual neighbours ---------------------------

def test_b3_infer_votes_the_dominant_radical():
    reranker = build()
    meta = reranker.infer_query_meta(
        [("B001", 0.90), ("C001", 0.88), ("G001", 0.87), ("A001", 0.50)])
    assert meta.inferred is True
    assert meta.radical == "木"
    assert meta.margin > 0.15


def test_b3_infer_declines_when_the_vote_is_split():
    # Declining to rerank beats reranking toward a guess: radical falls back to None, which makes
    # the radical signal drop out for this query alone.
    meta = build().infer_query_meta([("B001", 0.90), ("A001", 0.90)])
    assert meta.radical is None
    assert meta.margin == pytest.approx(0.0)
    assert meta.votes  # the tally is still exposed for debugging


def test_b3_infer_uses_a_weighted_median_for_stroke_count():
    # Equal weights over strokes {10, 20, 4} -> 10, and the far outlier at 20 does not drag it
    # the way a mean would (mean would be 11.33).
    meta = build().infer_query_meta([("B001", 0.9), ("G001", 0.9), ("D001", 0.9)])
    assert meta.stroke_num == pytest.approx(10.0)


def test_b3_infer_rescales_the_component_bag_to_peak_one():
    # Without this the soft bag is uniformly tiny and _shape_sim's min/max multiset Jaccard
    # collapses toward zero for every candidate alike.
    meta = build().infer_query_meta([("B001", 0.90), ("A001", 0.85)])
    assert max(meta.components.values()) == pytest.approx(1.0)


def test_b3_rerank_accepts_an_inferred_query():
    reranker = build(normalize="zscore")
    meta = reranker.infer_query_meta([("B001", 0.90), ("C001", 0.88), ("G001", 0.87)])
    ranked = reranker.rerank(meta, [("A001", 0.80), ("B001", 0.79), ("F001", 0.78)])
    assert len(ranked) == 3
    assert all(isinstance(r.score, float) for r in ranked)


def test_b3_infer_needs_at_least_one_neighbour():
    with pytest.raises(ValueError):
        build().infer_query_meta([])


# --- B1: circularity guard -------------------------------------------------------------------

def test_b1_default_weights_taint_all_three_proxy_metrics():
    tainted = build().circularity_report()
    assert tainted == {
        "radical@k": "RADICAL",
        "stroke_mae@k": "STROKE_NUM",
        "ids_jaccard@k": "SHAPE_MORPH",
    }


def test_b1_zeroing_a_weight_untaints_only_that_metric():
    weights = dict(DEFAULT_WEIGHTS, radical=0.0, shape=0.0)
    assert build(weights=weights).circularity_report() == {"stroke_mae@k": "STROKE_NUM"}


def test_b1_visual_only_reranking_is_not_circular():
    weights = {"visual": 1.0, "radical": 0.0, "stroke": 0.0, "shape": 0.0}
    assert build(weights=weights).circularity_report() == {}


# --- general behaviour -----------------------------------------------------------------------

def test_query_is_excluded_from_its_own_candidate_list():
    ranked = build().rerank("Q001", [("Q001", 1.0), ("B001", 0.9)])
    assert order(ranked) == ["B001"]


def test_inferred_queries_keep_every_candidate():
    # There is no query UNICODE to exclude, and dropping nothing is the safe default.
    reranker = build()
    meta = reranker.infer_query_meta([("B001", 0.9)])
    assert len(reranker.rerank(meta, [("Q001", 1.0), ("B001", 0.9)])) == 2


def test_top_k_trims_after_sorting():
    ranked = build().rerank("Q001", [("A001", 0.95), ("B001", 0.90), ("F001", 0.85)], top_k=2)
    assert len(ranked) == 2


def test_ties_break_on_unicode_so_runs_are_reproducible():
    # Identical inputs in a different order must produce the same ranking.
    reranker = build(normalize="zscore")
    forward = reranker.rerank("Q001", [("B001", 0.9), ("G001", 0.9), ("C001", 0.9)])
    backward = reranker.rerank("Q001", [("C001", 0.9), ("G001", 0.9), ("B001", 0.9)])
    assert order(forward) == order(backward)


def test_empty_candidate_list_returns_empty():
    assert build().rerank("Q001", []) == []


def test_candidate_with_no_metadata_at_all_still_ranks():
    ranked = build().rerank("Q001", [("H001", 0.9), ("B001", 0.8)])
    h001 = next(r for r in ranked if r.unicode == "H001")
    assert h001.parts["radical"] is None
    assert h001.parts["stroke"] is None
    assert h001.parts["shape"] is None


def test_shape_similarity_is_one_for_identical_decompositions():
    reranker = build()
    meta = reranker.meta_for("Q001")
    assert reranker.raw_signals(meta, "B001", 0.5)["shape"] == pytest.approx(1.0)
    # A001 shares only 交, so it lands well below.
    assert reranker.raw_signals(meta, "A001", 0.5)["shape"] < 0.5


# --- data-backed checks (need openpyxl + the real table) --------------------------------------

@pytest.mark.data
def test_radical_regex_covers_the_whole_real_table():
    """All 31073 non-null RADICAL values parse, yielding the 214 Kangxi radicals."""
    pytest.importorskip("openpyxl")
    column = pd.read_excel(CHAR_TABLE, usecols=["RADICAL"])["RADICAL"]
    present = [v for v in column if isinstance(v, str) and v.strip()]
    assert len(present) == 31073
    assert [v for v in present if not RADICAL_RE.match(v.strip())] == []
    assert len({parse_radical(v) for v in present}) == 214


@pytest.mark.data
def test_real_corpus_statistics_match_what_the_review_measured():
    pytest.importorskip("openpyxl")
    reranker = Reranker.from_excel()
    assert len(reranker.char) == 31208
    decomposable = [c for c in reranker.components.values() if c]
    assert len(decomposable) == 29510            # 1698 characters do not decompose
    assert len(reranker.idf) == 3915             # distinct components
    # 口 is the most common component, in 1520 of the 29510 decomposable characters.
    assert reranker.idf["口"] == pytest.approx(math.log(29510 / 1520))
    assert reranker._default_idf == pytest.approx(math.log(29510 / 1))


@pytest.mark.data
def test_from_excel_accepts_the_master_table_unaided():
    pytest.importorskip("openpyxl")
    reranker = Reranker.from_excel()
    ranked = reranker.rerank("20001", [("20003", 0.9), ("20004", 0.8)])
    assert len(ranked) == 2

"""Tests for the pure helpers in ``evaluate_rerank.py``.

Like ``test_rerank.py`` this runs on pandas + pytest alone. The suite bodies need torch, faiss and
embedding caches, so they are not covered here; what *is* covered is the logic that decides which
images get scored and how confidently -- the part where a silent mistake would corrupt a reported
number rather than crash.

``load_labels`` gets the most attention because it is a **reimplementation**: the original lives
inside ``evaluate_test_images.main()`` (around line 325) and is not importable, so the two copies
can drift. These tests pin the reimplementation against the real
``output/label_sheets/label_template.csv``, which is checked into the repo.
"""

import os
import re

import pandas as pd
import pytest

from evaluate_rerank import load_labels, load_manifest, wilson_ci

CONFIDENCE_LEVELS = ("high", "med", "low")
LABELS = os.path.join("output", "label_sheets", "label_template.csv")

needs_labels = pytest.mark.skipif(
    not os.path.exists(LABELS), reason=f"{LABELS} not present")


def candidates_from_sheet(path=LABELS):
    """Rebuild {file: [unicode, ...]} from the sheet's own '1=X(hex) 2=Y(hex)' column.

    ``label_sheets()`` writes candidates in the order of ``sorted(gt)``, which is exactly the
    order ``correct_index`` refers to, so parsing it back gives the list load_labels expects.
    """
    table = pd.read_csv(path)
    out = {}
    for row in table.itertuples():
        out[row.file] = re.findall(r"\(([0-9A-Fa-f]+)\)", row.candidates)
    return out


# --- Wilson intervals -------------------------------------------------------------------------

def test_wilson_ci_handles_empty_samples():
    assert wilson_ci(0, 0) == (0.0, 0.0)


def test_wilson_ci_stays_inside_zero_one():
    for hits, n in ((0, 59), (59, 59), (1, 5)):
        lo, hi = wilson_ci(hits, n)
        assert 0.0 <= lo <= hi <= 1.0


def test_wilson_ci_brackets_the_observed_rate():
    lo, hi = wilson_ci(28, 59)
    assert lo < 28 / 59 < hi


def test_wilson_ci_width_matches_the_projects_sample_size_claims():
    """Why --suite scans cannot settle anything and --suite render can.

    59 scans is the whole of ``test_images/`` and the repo has no other source of real scanned
    glyphs, so this width is a property of the data, not of the method.
    """
    lo, hi = wilson_ci(28, 59)
    assert 0.11 < (hi - lo) / 2 < 0.14        # about +/-12 points

    lo, hi = wilson_ci(2500, 5000)
    assert (hi - lo) / 2 < 0.02               # about +/-1.4 points


def test_wilson_ci_narrows_as_n_grows():
    widths = [wilson_ci(n // 2, n)[1] - wilson_ci(n // 2, n)[0] for n in (59, 300, 5000)]
    assert widths == sorted(widths, reverse=True)


# --- load_labels ------------------------------------------------------------------------------

@needs_labels
def test_load_labels_reads_every_row_of_the_real_sheet():
    truth, dropped = load_labels(LABELS, candidates_from_sheet(), None, CONFIDENCE_LEVELS)
    assert len(truth) == 59
    assert dropped == []


@needs_labels
def test_load_labels_min_confidence_high_drops_the_six_uncertain_rows():
    # The sheet carries 53 high, 4 med, 2 low -- see BENCHMARK.md 10.6.1.
    truth, dropped = load_labels(LABELS, candidates_from_sheet(), "high", CONFIDENCE_LEVELS)
    assert len(truth) == 53
    assert len(dropped) == 6


@needs_labels
def test_load_labels_min_confidence_med_keeps_high_and_med():
    truth, _ = load_labels(LABELS, candidates_from_sheet(), "med", CONFIDENCE_LEVELS)
    assert len(truth) == 57


@needs_labels
def test_real_sheet_index_and_unicode_agree_on_every_row():
    """correct_index and correct_unicode must name the same candidate.

    They disagreed once already: the sheet used to truncate its candidate list at 24 while
    ``n_candidates`` reported the true total, so several answers sat past the end of what was
    printed. This is the check that would have caught it.
    """
    candidates = candidates_from_sheet()
    table = pd.read_csv(LABELS)
    for row in table.itertuples():
        listed = candidates[row.file]
        assert len(listed) == row.n_candidates, f"{row.file}: sheet truncates its candidate list"
        assert listed[int(row.correct_index) - 1].upper() == str(row.correct_unicode).upper()


def test_load_labels_prefers_correct_unicode_over_correct_index(tmp_path):
    sheet = tmp_path / "labels.csv"
    sheet.write_text(
        "file,correct_index,correct_unicode,confidence\nimg,1,BEEF,high\n", encoding="utf-8")
    truth, _ = load_labels(str(sheet), {"img": ["DEAD", "BEEF"]}, None, CONFIDENCE_LEVELS)
    assert truth == {"img": "BEEF"}


def test_load_labels_falls_back_to_correct_index(tmp_path):
    sheet = tmp_path / "labels.csv"
    sheet.write_text(
        "file,correct_index,correct_unicode,confidence\nimg,2,,high\n", encoding="utf-8")
    truth, _ = load_labels(str(sheet), {"img": ["DEAD", "BEEF"]}, None, CONFIDENCE_LEVELS)
    assert truth == {"img": "BEEF"}


def test_load_labels_ignores_rows_with_no_candidate_list(tmp_path):
    # A labelled file whose image is absent from this run must not enter the denominator.
    sheet = tmp_path / "labels.csv"
    sheet.write_text(
        "file,correct_index,correct_unicode,confidence\nghost,1,BEEF,high\n", encoding="utf-8")
    truth, _ = load_labels(str(sheet), {"img": ["DEAD"]}, None, CONFIDENCE_LEVELS)
    assert truth == {}


def test_load_labels_ignores_an_out_of_range_index(tmp_path):
    sheet = tmp_path / "labels.csv"
    sheet.write_text(
        "file,correct_index,correct_unicode,confidence\nimg,9,,high\n", encoding="utf-8")
    truth, _ = load_labels(str(sheet), {"img": ["DEAD", "BEEF"]}, None, CONFIDENCE_LEVELS)
    assert truth == {}


# --- load_manifest ----------------------------------------------------------------------------

def write_manifest(tmp_path, rows):
    directory = tmp_path / "rendered"
    directory.mkdir()
    written = []
    for unicode_value, char, view, aug in rows:
        image = directory / f"{view}_{unicode_value}_{aug}.png"
        image.write_bytes(b"")
        written.append((unicode_value, char, view, aug, str(image)))
    pd.DataFrame(written, columns=["unicode", "char", "view", "aug", "path"]).to_csv(
        directory / "manifest.csv", index=False)
    return str(directory)


def test_load_manifest_drops_corpus_views_and_keeps_clean_renders(tmp_path):
    # The corpus image is a 'view' of the class for training purposes, but it is the retrieval
    # target here -- using it as a query would be asking the model to find an image it is holding.
    directory = write_manifest(tmp_path, [
        ("4E00", "一", "corpus", 0),
        ("4E00", "一", "hanamin", 0),
        ("4E00", "一", "hanamin", 1),
    ])
    clean = load_manifest(directory, augmented=False, sample=0, seed=42)
    assert list(clean["view"]) == ["hanamin"]
    assert list(clean["aug"]) == [0]


def test_load_manifest_augmented_selects_only_degraded_views(tmp_path):
    directory = write_manifest(tmp_path, [
        ("4E00", "一", "corpus", 0),
        ("4E00", "一", "hanamin", 0),
        ("4E00", "一", "hanamin", 1),
        ("4E00", "一", "lxgw-kai", 2),
    ])
    augmented = load_manifest(directory, augmented=True, sample=0, seed=42)
    assert sorted(augmented["aug"]) == [1, 2]


def test_load_manifest_keeps_unicode_as_a_string(tmp_path):
    # '20001' must not become the integer 20001, or it will never match a corpus UNICODE.
    directory = write_manifest(tmp_path, [("20001", "𠀁", "hanamin", 0)])
    assert load_manifest(directory, False, 0, 42)["unicode"].iloc[0] == "20001"


def test_load_manifest_sampling_is_reproducible(tmp_path):
    directory = write_manifest(
        tmp_path, [(f"{0x4E00 + i:04X}", "x", "hanamin", 0) for i in range(20)])
    first = load_manifest(directory, False, 5, seed=42)["path"].tolist()
    second = load_manifest(directory, False, 5, seed=42)["path"].tolist()
    assert first == second and len(first) == 5


def test_load_manifest_without_a_manifest_says_what_to_run(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        load_manifest(str(tmp_path), False, 0, 42)
    assert "render_fonts.py" in str(excinfo.value)


def test_load_manifest_without_augmented_rows_says_so(tmp_path):
    directory = write_manifest(tmp_path, [("4E00", "一", "hanamin", 0)])
    with pytest.raises(SystemExit) as excinfo:
        load_manifest(directory, augmented=True, sample=0, seed=42)
    assert "--augment" in str(excinfo.value)

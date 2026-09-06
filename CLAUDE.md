# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Repo: <https://github.com/vo-hoang-kh4ng/comparing-character-glyph-images-Sino-N-m>
Model: <https://huggingface.co/vohoangkh4ng/chinese-clip-ft-sinonom>

**The repo itself is the submitted artefact** — the marker reproduces from a clean clone, so
`README.md`'s "Tái lập" section is a deliverable, not a courtesy. Every command in it must run
as written from the repo root, and each step carries the expected value to check against. If you
change a script's CLI or a headline number, update that section in the same commit.

## What this is

A university group project ("Image Comparison") for finding visually-similar Sino-Nôm (Chữ Nôm /
Hán-Nôm) characters by comparing character glyph images. This is not a conventional application —
it's a set of standalone analysis scripts plus their input data (Excel dictionaries and a corpus of
character images). There is no build system and no package manifest beyond `requirements.txt`.
There *is* a test suite — `python -m pytest -q` → 55 passed, pandas only, no torch needed; see the
Tests section below.

Two search modes exist, mirroring how the assignment was handed out:
- **Part 1** — top-K similar characters across the *whole* corpus (`search_all_chars_in_corpus.py`).
- **Part 2** — top-K restricted to characters sharing one Quốc Ngữ reading
  (`search_use_QuocNgu_mapping.py`).

### Team split (3 people)

Recorded in `phan_cong_cong_viec.xlsx` (gitignored — it is an `.xlsx`). Summary, because it explains
why the code is shaped the way it is:

| Person | Scope |
|---|---|
| 1 | Swap ResNet18 for a Transformer feature extractor and benchmark it against the baseline. **This is the work that produced `feature_extractors.py` and `benchmark_extractors.py`.** |
| 2 | Rerank/filter top-K using the unused metadata columns (`RADICAL`, `STROKE_NUM`, `SHAPE_MORPH`); upgrade Part 2 from Canny+histogram to embeddings. |
| 3 | Real evaluation (Precision@k with weak labels), report/slides, and the stretch goal of structural decomposition per the FudanOCR paper. |

**All three scopes have now been worked.** Person 1 produced `feature_extractors.py` /
`benchmark_extractors.py` and the full-corpus numbers; Person 2 produced `rerank.py` /
`evaluate_rerank.py` plus the Part 2 embedding upgrade; Person 3 produced `evaluate_test_images.py`,
the hand-assigned Part 2 labels, and the report. The one thing that remains explicitly *not* done is
a **human relevance evaluation** — the proxy metrics in `benchmark_extractors.py` exist only to rank
backends against each other and are not that, which every results table in the repo says out loud.

## Data files (read-only inputs, do not regenerate by hand)

- `images.zip` — one glyph image per character, named `<UNICODE_CODEPOINT_HEX>.jpg` (e.g.
  `2A0AF.jpg`). Scripts expect this extracted to an `./images/` folder alongside them.
- `final_characteristics-v2.xlsx` — master character table. Key columns used by the scripts:
  `UNICODE` (hex codepoint, matches image filenames) and `CHAR` (the actual glyph). Also carries
  linguistic metadata (`AM_NOM`, `STROKE_NUM`, `RADICAL`, `SHAPE_MORPH`, `STROKE`, `VARIATION`,
  `P_ANCIENT`, `P_MODERN`). The retrieval pipeline never reads these; `benchmark_extractors.py`
  uses three of them as weak labels and `rerank.py` uses the *same* three as features — which is
  exactly why scoring the reranker with those metrics is circular (B1).
- `QuocNgu_SinoNom_Dic.xlsx` — mapping table with columns `QuocNgu` (modern Vietnamese romanized
  word) and `SinoNom` (corresponding Sino-Nôm character(s)). Used to restrict similarity search to
  characters sharing a given Quốc Ngữ reading.
- `materials/2309.01083v1.pdf` — reference paper ("Sino-Nom character decomposition") cited as a
  direction for future improvement (see README.txt).

Neither xlsx file has been observed to require anything beyond `pandas.read_excel` (openpyxl
engine); don't add unnecessary preprocessing.

## The scripts

### `feature_extractors.py` — pluggable embedding backends

All backends share one contract:
`build_extractor(name, device, dtype).embed_paths(paths) -> (N, D) float32`, L2-normalized so inner
product == cosine similarity (what the Faiss `IndexFlatIP` downstream assumes). Backends:

| name | model | dim | notes |
|---|---|---|---|
| `resnet18` | ImageNet ResNet18 | 512 | **Original baseline, kept bit-for-bit.** Patching `conv1` to 1 channel *discards* the pretrained first-layer filters and re-initializes them randomly — everything downstream is built on a random edge detector. This is the flaw the other backends fix; keep it faithful so "before" numbers stay honest. |
| `resnet18-gray` | ImageNet ResNet18 | 512 | Same net, but `conv1` is seeded by summing the pretrained RGB filters, so the ImageNet prior survives. Isolates how much of the baseline's weakness is that one line. |
| `chinese-clip` | OFA-Sys/chinese-clip-vit-base-patch16 | 512 | Image tower, projected embedding (`get_image_features`). Pretrained on Chinese image–text pairs, so its prior is closest to Han glyphs. |
| **`chinese-clip-large`** | OFA-Sys/chinese-clip-vit-large-patch14 | 768 | **Best quality measured so far.** Same family as base, so base→large→huge isolates the effect of scale alone. |
| `chinese-clip-huge` | OFA-Sys/chinese-clip-vit-huge-patch14 | 1024 | Bigger than large but *not* better on radical@k — see the saturation note in the results. |
| **`chinese-clip-ft`** | ViT-B/16 + `output/finetune/best.pt` | 512 | **Best on real scans by a wide margin** — hit@1 0.5932 vs large's 0.0847. Same architecture and embedding width as `chinese-clip`, so every downstream script works unchanged; only the weights differ. Loading fails loudly if the checkpoint is missing, because a silent fallback would report zero-shot numbers under a finetuned name. Override the path with `$GLYPH_FT_CKPT`. |
| `dinov2` | facebook/dinov2-base | 1536 | CLS token ⊕ mean-pooled patch tokens — CLS carries global shape, patch mean carries stroke texture. |

The three `chinese-clip*` backends all run through one `ChineseClipExtractor`; the embedding width
is read from `config.projection_dim`, never hardcoded, so adding another size is a one-line entry in
`_HF_MODEL_IDS` plus `BACKENDS`.

Two memory tricks make the large models fit next to the vLLM workers, both in
`ChineseClipExtractor.__init__`:

- **The text tower is deleted before the model moves to the GPU.** `get_image_features` only uses
  `vision_model` + `visual_projection`, but `ChineseCLIPModel` also loads a full RoBERTa — ~1.3 GB
  of pure waste on ViT-H. `del model.text_model, model.text_projection` while still on CPU means
  only the vision half is ever transferred. Do not "restore" these; nothing calls them.
- **`--dtype fp16`** halves the weights. Validated, not assumed: fp16 vs fp32 on `chinese-clip`
  agrees to cosine ≥ 0.99999 with **99.31%** top-20 overlap, so the comparison stays fair. ViT-H
  needs it (fp32 would be ~3.8 GB against ~3.05 GB free).

Version handling: `transformers` v5 returns an output object from `get_image_features` (the
projection lives in `pooler_output`) while v4 returned the tensor directly, and the `torch_dtype`
kwarg was renamed to `dtype` around 4.56. `ChineseClipExtractor` and `_from_pretrained` handle both
splits.

### `search_all_chars_in_corpus.py` — corpus-wide top-K similarity

For every character in `final_characteristics-v2.xlsx` that has an image, finds the top-K
(default 20) most visually similar other characters across the *entire* corpus.

Pipeline: load `./images/<UNICODE>.jpg` → embed via the chosen backend → Faiss `IndexFlatIP` →
query top-(K+1) and drop the self-match. Determinism is enforced via `set_deterministic()` (seeds
random/numpy/torch, disables cudnn benchmarking) — preserve this if you touch the model or data
pipeline, since results are expected to be reproducible.

CLI (no more editing constants): `--backend` (default `chinese-clip`), `--top-k`, `--images`,
`--output-dir`, `--batch-size`, `--device`, `--limit N` (quick trial on the first N chars),
`--refresh` (ignore cache).

Embeddings are cached to `output/embeddings_<tag>.npz`, keyed by the UNICODE list they were built
from, and reused across runs — this is what makes `benchmark_extractors.py` cheap. The `<tag>` is
the backend name, plus a `_limit<N>` suffix when `--limit` is passed, so trial runs never clobber
the full-corpus cache (and `benchmark_extractors.py` must be given the *same* `--limit` to find
them). Output goes to `output/output_top_k_similar_char_<tag>.xlsx`/`.csv` — columns are
`Input Character`, `Top <k> Similar Characters` (the k is interpolated into the header, so it is
`Top 20 …` by default), and `Similarity Scores`. `./output/` is created automatically.

Note: 26,044 of the 31,208 table rows have a glyph image; the other 5,164 are skipped silently
(count is reported, not one line per miss).

### `benchmark_extractors.py` — compare backends

Reads the embedding caches (run the search script once per backend first), re-runs the search, and
reports speed plus three **weak-label quality proxies** derived from columns the search pipeline
itself never uses:

- `radical@k` — share of top-K neighbours with the same Kangxi radical (`RADICAL`, parsed to strip
  the `(+n nét)` residual-stroke suffix).
- `stroke_mae@k` — mean |stroke-count difference| (`STROKE_NUM`); lower is better.
- `ids_jaccard@k` — mean Jaccard overlap of IDS decomposition components (`SHAPE_MORPH`, e.g.
  `⿱乛丄`, with the ⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻ operators stripped).

These are cheap stand-ins that correlate with "looks alike", adequate for *ranking backends against
each other* — they are not human relevance judgements, and the full evaluation is a separate work
item.

Writes `output/benchmark_summary.xlsx` with three sheets — `Metrics` (per-backend speed + proxies),
`Agreement` (mean pairwise top-K overlap, i.e. how differently two backends rank the corpus), and
`Samples` (top-10 neighbour strings for 25 seeded-random query characters) — plus
`contact_sheet_<backend>.png` (query glyph in a red box + its top-5 neighbours) for eyeballing.
A backend with no cache file is skipped with a message, not an error; only an empty run aborts.

### `benchmark_search.py` — is exact Faiss worth replacing?

Compares `IndexFlatIP` against HNSW / IVFFlat / IVFPQ. Reads the embedding caches, so it is nearly
free to re-run. Deliberately separates two kinds of number: **recall** only from real embeddings
(it depends on how vectors are distributed), **latency** from real data up to 26k and then random
vectors for 100k–1M (timing barely depends on the data). Do not merge the two. Conclusion is in the
Faiss section below — the short version is *keep the exact index*.

### `analyze_metric_ceiling.py` — is `radical@k` saturating, or is the *label*?

Written to answer feedback that the numbers "plateau at 0.5". Three stages, three hypotheses:

- `--stage ceiling` — the theoretical ceiling of `radical@k` from the radical-group size
  distribution. It is **0.9862** at k=20, and only 3% of queries are capped at all, so a too-small
  top-K is *not* the bottleneck.
- `--stage curve` — `radical@k` for k=1…100 per backend. `chinese-clip-large` gets **0.6331 at
  k=1**, decaying gently to 0.5361 at k=20. The bottleneck is at rank 1; widening or narrowing k
  cannot fix it.
- `--stage oracle` — the ceiling that actually matters. Ranking by Jaccard over the *true* IDS
  components (a ranker that already knows the character's structure, which no image model can beat)
  scores only **0.7338**. `RADICAL` is not a purely visual label: the radical appears in the
  character's own IDS decomposition just **56.7%** of the time.

So `chinese-clip-large` sits at **73% of the achievable ceiling** and **24.4× random** (0.0220) —
headroom is ~+37% relative, not 2×. Full write-up in BENCHMARK.md §9, including why the
large→huge regression is evidence for the *pretraining-domain* hypothesis rather than a capacity
one, and the circularity trap to avoid when fine-tuning (never train on `RADICAL` and then report
`radical@k`).

Needs `scipy` (sparse component matrix); the oracle stage is O(N²) in chunks, ~2 min for 26k.

### `render_fonts.py` — synthesising positive pairs for fine-tuning

The corpus has **exactly one image per character** (26,044 images / 26,044 characters), so *no*
"same character, different style" pair exists anywhere in the data — and that is precisely the pair
metric learning needs. This script manufactures them by re-rendering every character from its
Unicode codepoint through 7 typefaces (mincho, song, **kai/brush**, hei, fangsong).

Two things that matter if you touch it:

- **Fonts are not installed on this machine** (`fc-list :lang=zh` = 0) and are not committed
  (~200 MB, `fonts/` is gitignored). Run `bash download_fonts.sh` first. `FONT_GROUPS` maps a style
  tag to an ordered *fallback list* — `hanamin` is HanaMinA (BMP) then HanaMinB (Ext-B), because
  neither alone covers the corpus.
- **Coverage is checked against the font's `cmap`, never assumed.** A codepoint the font lacks
  renders as a `.notdef` tofu box — identical for every missing character — which would poison the
  positive pairs. Characters outside the cmap are skipped, and a blank render is dropped too.

Union coverage is 26,033/26,044 = **100.0%** (11 characters no font can draw), averaging 5.05
typefaces per character → 131,604 rendered images. Output is `output/rendered/<tag>/<UNICODE>.png`
plus `manifest.csv`, where the class label is the Unicode codepoint and the original corpus image is
listed as one more view (`view=corpus`).

**Measured, and the number is the point:** with the current `chinese-clip-large`, rendered glyphs
retrieve their own character at hit@1 = 0.7018 — but real scans manage 0.0847 (BENCHMARK.md §10.2).
The hardest font is `lxgw-kai`, the most brush-like one, at 0.6350, so the data does span the right
axis; it is just an order of magnitude easier than the real failure. `--augment N` runs each render
through `scan_augment.degrade` (below), which pulls hit@1 down to 0.1986 — within 2.8x of the real
scans instead of 9.4x. **Do not present multi-font rendering alone as having closed the handwriting
gap** — see BENCHMARK.md §13.3-§13.5.

### `scan_augment.py` — degrading renders to match real scans, calibrated against measurements

The gap between a clean render and a real scan is not font choice, and it is not guesswork: six
statistics measured over the 59 `test_images/` scans say where it is. Blur is the largest
(`edge` 15.6 vs 102.2, a 6.55x gap), then contrast (91.6 vs 232.8), stroke thickness (0.051 vs
0.028), ink fill, and sensor noise (10.1 vs 2.7).

`degrade()` reproduces those six axes in the physical order a scanned page acquires them: elastic
warp → stroke dilation → downsample/upsample (this is what makes adjacent strokes *merge*) →
Gaussian blur → paper tint and contrast reduction → sensor noise → non-square stretch. Nothing here
touches radical structure, so the Unicode label stays valid.

**The parameter ranges are calibrated, not tuned by eye.** `python scan_augment.py --calibrate`
prints the scan/render/degraded comparison and a mean log-error: currently 0.880 → 0.103 (8.5x
better), with all six statistics landing within 0.77-1.00x of the real scans. If you change a
constant in `degrade()`, re-run `--calibrate`; do not adjust them by feel.

The residual 2.8x difficulty gap is not fixable here. Every operation in `degrade()` is *image
noise*; real scans are additionally written with a brush in running script — merged strokes, shifted
component proportions, dropped strokes. That is a *structural* difference no image-processing
pipeline can synthesise from printed type. Closing it needs real brush data (MCCD) or real Hán-Nôm
scans (NomNaOCR / IHR-NomDB).

### `evaluate_test_images.py` — the only non-proxy evaluation in the project

`test_images/` filenames are Telex-typed Vietnamese (`cofn.jpg` = "còn", `nguwowsi.jpg` = "người"),
so decoding them and looking the word up in `QuocNgu_SinoNom_Dic.xlsx` yields **real ground truth**:
the set of Sino-Nôm characters that actually read that way. Telex allows the tone letter directly
after the vowel (`cofn`) or at the end (`conf`), so the decoder generates every tone position and
matches — **59/59 filenames resolve, none ambiguously**.

This was 56/59 until the three stragglers were traced. Only one was a data problem: `phong_1` was a
**code bug** — `decode_filenames` called `words.pop()` on the set living inside `lookup`, so after
`phong.jpg` matched, `lookup["phong"]` was empty and the *second* image of any repeated word was
silently dropped (now `next(iter(words))`). The other two are filename typos, corrected via
`FILENAME_FIXES` rather than by renaming files, so `test_images/` still matches the original zip
byte-for-byte: `dau` → "dâu" (Kiều's "bể dâu", distinct from `ddau.jpg` = "đau"), `nguwowsi` →
"người" (typed `s` for `f`).

**Part 1, whole-corpus retrieval on real scans: hit@1 = 0.0847, hit@20 = 0.2034, MRR = 0.1143**
(n=59, `chinese-clip-large`/fp16). That is 179× the random baseline but low in absolute terms — and
much worse than the proxy metrics suggest. Report it honestly; it is the project's first real number.

**After fine-tuning this becomes 0.5932** (`chinese-clip-ft`, BENCHMARK.md §14) — the two 95% CIs
are disjoint, so this is the one comparison in the project that n=59 can actually carry. Keep the
zero-shot number in every table anyway: it is what makes the gain legible.

The cause is **distribution shift, not a weak model.** Symptom: a hub effect — 攅 is returned top-1
for 6 different queries, 劕/湸/旦/刟 for 3 each. Measured gaps: ink fill 0.353 vs 0.184, background
249 vs 255, and the scans are **not square** (96×128, 88×96) while every corpus image is 70×70, so a
straight resize to 224×224 distorts the glyph. But the dominant gap is only visible by eye: **the
queries are handwritten/cursive, the corpus is printed regular script.** `moojt.jpg` is a flowing
沒; its nine candidates are all upright 楷書.

`--normalize` (ink-tight crop → square pad → fixed fill ratio) fixes the *geometric* half. It cuts
the hub effect clearly (41 → 48 distinct top-1s) but the hit@k change is **inside the noise at
n=59** — 5 images vs 6, a one-image difference. So it is **off by default**; do not promote it to a
default on this evidence.

Part 2 on the same scans works far better, because it ranks ~7 candidates instead of 26,044. It also
widens the gap to the histogram: the two methods agree on top-1 only **23/59 (39.0%)**, versus 6/10
on clean corpus queries. Histogram scores collapse into a 0.003 band across a whole candidate group.

**Part 2 is now labelled.** `output/label_sheets/label_template.csv` carries `correct_index`,
`correct_unicode` and a `confidence` column for all 59 scans (53 `high`, 4 `med`, 2 `low`). The
scans are the opening lines of *Truyện Kiều*, so labels were assigned by shape matching first and
cross-checked against the poem's standard Nôm orthography — where the two disagreed the **image
won** (the scan writes "qua" as 戈 not 過, "năm" as 𢆥 not 年, "một" as 没 not 𠬠).

Measured: embedding 28/59 = **0.4746**, histogram 26/59 = 0.4407, random-in-group 0.1987. Both beat
random by >2x, so Part 2 genuinely works. All three newly recovered scans are missed by *both*
methods, so they only grew the denominator.

With `chinese-clip-ft` the same measurement gives **48/59 = 0.8136** (MRR 0.8983), and 45/53 =
0.8491 under `--min-confidence high`. That is a 22-image gap over histogram with disjoint CIs —
unlike the zero-shot 2-image gap, this one supports a conclusion on its own.

**The 11 `med`/`low` rows were re-reviewed** (BENCHMARK.md §10.6.1) by re-cropping the deciding half
of each 94×104-to-178×162 scan at 460–520px. Five rows rose to `high`, `gia` dropped to `low`, and
**one label was wrong**: `coix` was 揆 (扌) but the left radical is 土 → 𡎝, which is also the standard
Kiều spelling. That single fix cost embedding one image and gave histogram one — 29/25 became 28/26,
halving the gap from 4 images to 2. Treat that as the calibration for how much weight the table can
carry. `trari` and `gia` remain unresolved and need the original page scan, not the isolated glyph.

**`confidence` is a human judgement, not a score.** Nothing computes it — deliberately. The
sensitivity check below filters on this column and then re-scores the model; if `confidence` were
derived from the model's similarity the check would be circular. The rubric lives in
`CONFIDENCE_LEVELS` in `evaluate_test_images.py`: `high` = the deciding stroke is readable and every
rival candidate differs by radical or by a clearly visible stroke count; `med` = one or more rivals
are near-identical variants differing only where the scan is blurred (𣘛 vs 橷, both 木+兜); `low` =
the deciding stroke is unreadable and the pick leans on context. When shape and the poem disagree
the shape wins **and** the row drops to `med`. Filter with `--min-confidence high`.

**The sheet used to hide the answer (fixed).** `label_sheets()` defaulted to `max_candidates=24`
while `n_candidates` reported the true total, so `gia` listed 50 candidates, printed 24, and had its
answer at index **32**. Same for `lujc` (35/40), `phong` (30/35), `tuw` (27/97) — the labeller could
not see the right answer on the sheet. The labels themselves are fine (assigned off the full list,
and the scorer always used an untruncated `sorted(gt)`; all 59 rows re-verified index↔unicode), but
the sheets were unusable. Now `max_candidates=None` and each candidate carries its codepoint
(`32=笳(7B33)`). Do not set it back.

**Do not cite that gap as evidence that embeddings beat the histogram.** The 95% CIs overlap almost
entirely and the whole margin is **2 images out of 59**. Restricting to the 53 `high`-confidence
labels keeps it at exactly 2 (0.5094 vs 0.4717, MRR 0.6808 vs 0.6402). With a ±13-point CI on each
side, 2 images is not evidence. What the number does establish is that narrowing to the same-reading
group is what makes the task tractable, not the choice of scoring function.

**59 is the hard ceiling of this test set.** `test_images/` holds 59 scans and the repo has no other
source of real scanned glyphs. At n=59 around p≈0.5 the 95% CI is **±13.3 points**; ±8 points needs
n≈150, ±6 points needs n≈300. Any improvement smaller than ~15 points — which almost certainly
includes finetuning — **cannot be demonstrated on this data.** Growing n requires new scans, not new
methods.

Labels came from shape matching, not from a Hán-Nôm philologist; the 6 remaining `med`/`low` rows
(`coix`, `dau`, `gia`, `moojt`, `trari`, `xanh`) should be reviewed by someone who reads Nôm. Also
note `test_images/` holds isolated glyphs with no line numbers, so "cross-checking against Kiều" is
inference from the vocabulary of the opening lines, not a lookup. Re-run with
`--part 2 --labels output/label_sheets/label_template.csv [--min-confidence high]`.

### `search_use_QuocNgu_mapping.py` — same-reading top-K similarity

Ranks only the characters sharing one Quốc Ngữ reading (looked up via `QuocNgu_SinoNom_Dic.xlsx`,
then mapped to `UNICODE` via `final_characteristics-v2.xlsx`). The candidate set is a few dozen
characters, so there is no Faiss index — a plain sort is enough.

Two scoring methods via `--method`:

- **`embedding`** (default) — the same `feature_extractors.py` backends Part 1 uses, scored by
  cosine. Candidate glyphs are a subset of the corpus, so it **reads their vectors straight out of
  `output/embeddings_<backend>.npz`** and only runs a forward pass for the query image.
- **`histogram`** — the original Canny + 256-bin histogram pipeline, kept bit-for-bit.
- **`both`** — runs the two and prints their top-K agreement.

**Use `embedding`; `histogram` is kept for comparison, not for results.** The histogram only counts
*edge pixels*, so it cannot tell apart two characters that carry a similar amount of ink: querying
with the image of 咱 makes it return 些 at similarity **1.0000**, ranked above the query character
itself. Measured, on a self-retrieval test (query with a candidate's own image, so the correct
answer must be itself): embedding **64/64**, histogram 62/64, the two failures both ties.

Everything is CLI-driven now (`--word`, `--image`, `--method`, `--backend`, `--dtype`, `--top-k`,
`--device`) — the old hardcoded `input_text` / `test_image_path` globals are gone. `./output/` is
created automatically, and CSVs are written `utf-8-sig` like the other script.

Output: `output/output_embedding_<backend>_similarity.{xlsx,csv}` and/or
`output/output_histogram_similarity.{xlsx,csv}`, columns `UNICODE`, `SinoNom`, `Similarity`.

**Now validated on real scans** (`test_images/`, 59 images, added later than the Drive bundle). The
predicted widening happened: on clean corpus queries the two methods agreed on 6/10 of the top-10;
on real scans they agree on top-1 only 23/59 (39.0%). See `evaluate_test_images.py` above.

### `rerank.py` — metadata reranking, and the four defects it was built around

Stage 1 proposes candidates from pixels alone. `Reranker` rescores them with the three columns the
image models cannot see: `RADICAL`, `STROKE_NUM`, `SHAPE_MORPH`. Weights default to
visual 0.55 / radical 0.20 / stroke 0.15 / shape 0.10.

Deliberately imports **only stdlib + pandas** — never `search_all_chars_in_corpus`, which would pull
in torch and faiss and make the pure-logic unit tests need a 200 MB install. Heavy imports live
inside `main()`. Keep it that way. (`analyze_metric_ceiling.py` does *not* keep it that way — it
reaches `parse_radical` out of `benchmark_extractors`, dragging torch in for a pandas-only stage.)

The labels B1–B4 are used consistently in the module docstring, `test_rerank.py` and
`evaluate_rerank.py`:

- **B1 — circularity.** The three proxy metrics in `benchmark_extractors.py` are derived from the
  exact three columns this module consumes. `circularity_report()` names the metrics a given weight
  configuration invalidates, and `warn_circular()` prints a banner. **Never tune weights against
  `radical@k`.**
- **B2 — normalize before mixing.** Signals are z-scored *within each candidate list*
  (`normalize='zscore'`). Mixing raw is not a mixture at all: at default weights a candidate must
  win visual similarity by more than **0.364 cosine** to overcome one radical mismatch, and top-20
  cosines never span that, so raw mixing degenerates into a lexicographic sort (radical, then
  stroke) with the image model demoted to tie-breaker. Measured cost of getting this wrong: **55.8
  points of hit@1** (0.7467 → 0.1883). `normalize='none'` reproduces the old behaviour and exists
  only as the baseline that measures the fix; `'rrf'` is a rank-only alternative.
- **B3 — a query image has no metadata.** That is the real case for `test_images/` and for rendered
  glyphs. Passing `None` used to silently drop every metadata signal, turning the reranker into a
  no-op that still looked like it ran; it now raises. Use `infer_query_meta(corpus_neighbours)`,
  which votes the query's radical/strokes/components out of its **whole-corpus** top-50 (softmax
  over cosine, temperature 0.05). Feed it corpus-wide neighbours, **never** a Part 2 same-reading
  group — those characters share a reading, not a shape. A radical whose vote margin is under
  `min_margin=0.15` returns `None`: declining to rerank beats reranking toward a guess.
- **B4 — the wrong frame.** `load_corpus()` returns only UNICODE/CHAR/path, so `Reranker(df)` on it
  died with `KeyError: 'RADICAL'`. Use `Reranker.from_excel()`; the constructor validates columns
  and says so.

`score()` is gone — it scored one pair in isolation, which B2 makes meaningless. Its replacement is
`raw_signals()` (unnormalized, for explaining a ranking) plus `rerank()` (two-pass, because
normalizing needs the whole candidate list).

Two details worth not re-deriving: components are a `Counter`, not a set, so 林 `⿰木木` stays
distinguishable from 木, and `_shape_sim` is **IDF-weighted** — sharing a rare component is much
stronger evidence than sharing 口 (which appears in 1,520 of 29,510 decomposable characters).
`parse_radical` returns the glyph `一`, not `nhất 一`; note this differs from
`benchmark_extractors.parse_radical`, which keeps reading+glyph. Verified they induce the **same
214-group partition** with the same 135 nulls, so the two are interchangeable for grouping — but
they are two independent parsers of one column, so keep them in step.

### `evaluate_rerank.py` — three suites, only one of which produces a number

```bash
S="--suite render --sample 600 --backend chinese-clip-large --dtype fp16"
python evaluate_rerank.py $S --no-rerank        # stage-1 baseline
python evaluate_rerank.py $S --oracle-meta      # ceiling — DIAGNOSTIC, never a result
python evaluate_rerank.py $S                    # the real configuration
python evaluate_rerank.py $S --normalize none   # pre-B2, measures what B2 bought
```

- `--suite proxy` — the **circular** measurement, run on purpose and labelled invalid. Columns get a
  `[CIRCULAR]` suffix and the .xlsx carries a `Warnings` sheet, so a number pasted into a slide
  carries its own warning. A demonstration, never a result.
- `--suite render` — the honest number. Label is the character's own Unicode codepoint, which owes
  nothing to the three metadata columns. Every query is an image of unknown identity, so this suite
  necessarily exercises `infer_query_meta`.
- `--suite scans` — the read-out on the 59 real scans with the human labels. Report it, do not tune
  on it (±13 points at n=59).

Results, and the fact that they are a **negative result reported with its ceiling** — full write-up
in **BENCHMARK.md §15**:

| configuration | hit@1 (n=600) | vs stage 1 |
|---|---:|---:|
| stage 1, no rerank | 448/600 = 0.7467 | — |
| *rerank, oracle metadata* (**diagnostic**) | *553/600 = 0.9217* | *+105* |
| **rerank, real** | **434/600 = 0.7233** | **−14** |
| raw mixing (pre-B2) | 113/600 = 0.1883 | −335 |

The bottleneck is measured, not guessed: inferring the query's radical by neighbour voting is right
only **11.86%** of the time on real scans with a zero-shot stage 1 — 2.4× random but wrong 88% of
the time, while radical carries weight 0.20. With `chinese-clip-ft` as stage 1 it jumps to
**0.6610**. The weak link is the stage in front of it, not the inference design.

**Three caveats before anyone cites this as "metadata does not help":**

1. The table was measured on **clean** renders (`make_figures.py` records the command; there is no
   `--augment`), where stage 1 already scores 0.7467 — very little headroom. The script's own
   docstring recommends `--augment` for all four configurations; that has not been run.
2. The table and the 11.86% come from **two different suites** (render vs scans), so the causal
   chain joining them is not yet closed. `--suite render` prints its own voting accuracy; that
   number is not recorded in `make_figures.py`.
3. **Nobody has run it with `chinese-clip-ft` as stage 1**, which is the one configuration where the
   bottleneck is gone. `--suite scans --backend chinese-clip-ft` is uncontaminated (the 59 scans are
   not in training) and is the cheapest experiment that could overturn the conclusion.

### Tests — `test_rerank.py`, `test_evaluate_rerank.py`, `conftest.py`

`python -m pytest -q` → **55 passed**, ~25 s, and it needs **no torch and no GPU** — pandas only.
That is a deliberate property of `rerank.py`'s import discipline; do not break it.

The suite is not decorative: it pins each of B1–B4 as a named test (`test_b2_raw_mixing_needs_an_
impossible_cosine_gap_to_flip`, `test_b3_none_query_raises_instead_of_becoming_a_noop`, …), it
verifies `RADICAL_RE` matches **every** non-null value in the real table, and it checks that all 59
rows of `label_template.csv` agree between `correct_index` and `correct_unicode`.

`conftest.py` does two things, both load-bearing:

- `collect_ignore_glob = ["submission/*", ...]` — `package_submission.py` copies the test files into
  `submission/`, and without this pytest collects both copies and aborts with "import file
  mismatch". This recurs after every packaging run; fix it here, not by deleting `__pycache__`.
- Tests marked `data` **skip** rather than fail when `final_characteristics-v2.xlsx` is absent,
  because that table is course-given data and is not bundled in `submission.zip`. A marker running
  `pytest` on the unpacked submission should see skips, not three red tests.

**Coverage gap worth closing:** `telex_variants` / `decode_filenames` in `evaluate_test_images.py`
have **no tests**, despite being pure logic (no torch), being the source of ground truth for *every*
headline number, and having already carried a real bug — `words.pop()` mutated the set inside
`lookup`, so the second image of any repeated word was silently dropped.

### `finetune_glyph.py` — the fine-tune, and the one that failed first

Trains the Chinese-CLIP ViT-B/16 image tower for **typeface invariance**: same Unicode codepoint
across different fonts must land on the same vector. Label is the codepoint and nothing else —
deliberately *not* `RADICAL`, because BENCHMARK.md §9.4 warns that training on a label and then
reporting that label's metric is measuring the training set.

Results are in BENCHMARK.md §14. The short version: real-scan hit@1 0.0847 → **0.5932**, and
held-out Unicode classes score the same as trained ones (0.9939 vs 0.9954), so it is not memorising.

**ArcFace collapsed — do not retry it.** The first design used ArcFace over 23,440 classes. After
one epoch, render→corpus hit@1 went 0.79 → 0.0035 on *seen* classes and 0.80 → 0.0050 on unseen,
with loss flat near ln(N). Falling on both sides means collapse, not overfitting. The cause is the
data shape, not a hyperparameter: 23,440 classes with ~6 images each means the randomly-initialised
head never organises, so what it back-propagates is noise — and Adam normalises by gradient
magnitude, so "lr = 1e-5" still moves the weights far enough to destroy the pretrained structure in
~3,000 steps. `clip_grad_norm_` does not help; clipped noise is still noise.

SupCon has no learnable parameters in the loss, so there is no random head injecting noise.
`--loss arcface` is kept only to reproduce the failure.

**P×K sampling is load-bearing, not a tweak.** A random batch of 48 drawn from 23,440 classes
contains ~0.05 positive pairs in expectation — almost every batch would have nothing to pull
together. `PKSampler` guarantees 24 classes × 2 views.

**Degradation runs online in the DataLoader**, not pre-rendered to disk. `scan_augment.degrade` is
already calibrated (§13.4); calling it per `__getitem__` with a `(seed, epoch, index)` RNG gives a
fresh variant every epoch, reproducibly, and costs no disk. So `render_fonts.py --augment N` is
**not** needed for training.

**Three things exist because the collapse was caught late:**
- `--eval-steps 600` evaluates mid-epoch, so a collapse shows up in ~1 minute instead of an hour.
- The `cos=` column is the mean pairwise cosine of 2,000 random corpus vectors — it goes to 1.0 on
  collapse. Without it, "learning badly" and "collapsed" look identical on hit@1, and they are fixed
  differently. It ran **0.890 → 0.000**; the 0.890 at zero-shot is itself the explanation for why
  discrimination was weak before.
- SupCon's collapse signature is loss parked at ln(P·K−1) = ln 47 = 3.85.

**`-inf * 0 = nan`.** In `supcon_loss` the diagonal is masked to `-inf` before `logsumexp`; it must
then be zeroed *before* multiplying by the positives mask, or the loss is `nan` from step 0.

VRAM: ~3.7 GB at batch 48 with gradient checkpointing and bf16 — that is why the backbone is B/16
and not large. large does not fit for *training* alongside the vLLM engines. If the GPU frees up,
raise the batch size before changing the backbone, or the comparison table stops being like-for-like.

### `paper.tex`, `make_figures.py`, `build_report.py`, `package_submission.py` — the submission

`build_report.py` generates `report/BaoCao.docx` and `.pdf` (9 pages). It is a
script, not a hand-written document, so every number traces back to the command that produced it and
a stale figure is one edit away rather than a hunt.

**The submitted report is now `report/paper.tex`, and it does compile.** Tectonic 0.17.0 is
installed at `~/.local/bin/tectonic` — a single static binary, no sudo, downloads TeX packages on
demand — so the earlier "there is no TeX on this box" note is obsolete:

```bash
export PATH=$HOME/.local/bin:$PATH
tectonic -X compile report/paper.tex        # -> report/paper.pdf, 9 pages
```

`paper.tex` is IEEEtran two-column, 9 pages, 5 figures, with three appendices (reproduction
commands, environment/cost, package layout). It **must** be XeLaTeX or LuaLaTeX, never pdflatex:
the Nôm glyphs live outside the BMP and only reach the page through `fontspec`.

**Load fonts by filename, not by system name.** `\setmainfont{TeX Gyre Termes}` fails under tectonic
because a system name needs the font registered with fontconfig; `\setmainfont{texgyretermes}[Extension=.otf, UprightFont=*-regular, ...]` resolves through kpathsea and works on every TeX Live /
MiKTeX / Tectonic install. That is the difference between "compiles on my machine" and "compiles
anywhere", and it is why the Nôm faces are loaded with an explicit `Path=../fonts/`.

`make_figures.py` generates all five figures into `report/figs/` from real measurements — a
`MEASUREMENTS` dict holds every number together with the command that produced it. **Do not hand-edit
the PDFs in `figs/`**; change the script and re-run it.

`build_report.py` still generates `report/BaoCao.docx` / `.pdf` (9 pages, two-column) via
LibreOffice — the same content in the Doc format the course also accepts. `report/report.tex` is the
**superseded** long-form technical report, kept for history and never successfully compiled; its
header says so.

**Nôm fonts must be in `~/.fonts`** or every Ext-B glyph renders as an empty box in the PDF — and it
fails silently, the .docx still opens. `ensure_fonts()` copies them on every run.

The trained checkpoint is published at
<https://huggingface.co/vohoangkh4ng/chinese-clip-ft-sinonom> (fp16). `MODEL_URL` in
`package_submission.py` points at it, so the generated `model/MODEL.md` ships a live link instead
of the "missing link" warning.

`package_submission.py` builds `submission/`. Three deliberate exclusions, all documented in the
generated `data/DATA.md` and `model/MODEL.md`:

- **Pretrained models are not bundled** — they are external, several GB, and download themselves.
  Only exact HF identifiers are recorded.
- **The 26k corpus and the two .xlsx tables are not bundled by default** (106 MB) — they are the
  course's given data, not the group's. `--with-corpus` includes them.
- **The 131,604 rendered images are not bundled** (1.1 GB) — one deterministic command regenerates
  them, and `finetune_glyph.py` degrades online so the real training set never exists on disk.

`label_template.csv` is *always* bundled: hand-assigned labels, not regenerable, and every Part 2
number depends on it.

**The shipped checkpoint is fp16, not fp32.** Not an approximation: `ChineseClipExtractor` calls
`model.to(fp16)` immediately after loading, so storing fp16 just performs a cast the pipeline
already does. Verified by rebuilding all 26,044 corpus vectors from both — max absolute difference
0.0. Half the size for a bit-identical result. `--fp32` ships the full one, only useful for resuming
training.

## Setting up (nothing but code is in git)

Bulk data (`images.zip`, both `.xlsx` files, `materials/`, `phan_cong_cong_viec.xlsx`) is gitignored
— too large / not source. It is distributed via a Google Drive folder linked in README.md.
`./output/` and `./images/` are gitignored too (generated, or course-given bulk data).

**`test_images/` IS tracked** — 59 scans, 1.2 MB. It is the project's only real evaluation set and
the source of every headline number, so a clone has to carry it or nothing reproduces. Same reason
`output/label_sheets/label_template.csv` is tracked: hand-assigned labels, not regenerable.

So a fresh clone needs:

```bash
pip install -r requirements.txt
# download from the Drive link in README.md, then:
unzip images.zip -d images          # -> ./images/<UNICODE>.jpg
# final_characteristics-v2.xlsx and QuocNgu_SinoNom_Dic.xlsx go in the repo root
```

Both search scripts create `./output/` themselves. `test_images/` comes with the clone.

```bash
python search_all_chars_in_corpus.py --backend chinese-clip     # or resnet18 / resnet18-gray / dinov2
python benchmark_extractors.py --backends resnet18 resnet18-gray chinese-clip dinov2
python search_use_QuocNgu_mapping.py
```

The HuggingFace backends download weights on first use (~600 MB Chinese-CLIP, ~350 MB DINOv2) into
`~/.cache/huggingface`. On an offline node, pre-seed that cache or set `HF_HOME`.

### Environment notes

Verified working on Python **3.14** with torch 2.13, torchvision 0.28, transformers **5.15**,
faiss-cpu 1.15, opencv-python 5.0 — all have wheels for 3.14, so no need to downgrade Python.
`requirements.txt` is unpinned; the code accommodates both transformers v4 and v5 (see the
`pooler_output` note above).

**On the H100 server (the machine this work is moving to), the setup is different — use `.venv/`.**
That box runs Python 3.10.12 with torch 2.6.0+cu124 and transformers 4.57.6 (the *v4* branch, so
`get_image_features` returns a tensor directly) already installed in the **system** interpreter,
which four vLLM engines depend on. Do **not** `pip install` into system Python — a resolver that
decides to move torch will take vLLM down with it. The venv is built with
`--system-site-packages` so it reuses the system torch rather than pulling its own copy:

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install "torchvision==0.21.0" faiss-cpu opencv-python-headless
.venv/bin/python search_all_chars_in_corpus.py --backend chinese-clip --device cuda --batch-size 32
```

Three deviations from `requirements.txt` that are deliberate, not drift:

- **`torchvision` must be pinned to 0.21.0** — that is the build matching torch 2.6.0. Unpinned,
  pip drags in a torch upgrade.
- **`opencv-python-headless`, not `opencv-python`** — headless drops the `libGL.so.1` dependency the
  server does not have, and Part 2 only uses `imread`/`resize`/`Canny`/`calcHist`, no GUI calls.
- pandas, numpy, openpyxl, pillow, torch, transformers, scikit-learn come from system site-packages;
  only the three above live in the venv.

Verify the venv did not shadow torch — `ls .venv/lib/python3.10/site-packages/ | grep ^torch` should
show `torchvision` and nothing else.

### Running on GPU

Pass `--device cuda`. Everything here is **inference only** — no training — so it is cheap:
ViT-B/16 fp32 weights are ~0.4 GB, and `--batch-size 32` fits comfortably in ~3 GB of VRAM. That
matters because the target server's H100 is mostly occupied by vLLM workers (~3.5 GB free at last
check); if VRAM is tight, keep the batch small and consider
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

A contrastive fine-tune needs gradients and optimizer state on top of that. **Measured, it does
fit**: `finetune_glyph.py` runs at **3.7 GB** with ViT-B/16 + gradient checkpointing + bf16 at batch
48, ~4 min/epoch. What does *not* fit is training `large` — that is the reason the finetuned
backbone is B/16, and it is a VRAM constraint, not a modelling choice.

### Throughput (measured)

| backend | CPU 16-core, batch 64 | H100, batch 32–64 | full 26k pass on GPU |
|---|---|---|---|
| `resnet18` | ~347 img/s | **3950 img/s** | 6.6 s |
| `resnet18-gray` | ~403 img/s | **3937 img/s** | 6.6 s |
| `chinese-clip` | ~5–9 img/s | **358 img/s** | 73 s |
| `chinese-clip-large` (fp16) | not measured | **354 img/s** | 74 s |
| `chinese-clip-huge` (fp16) | not measured | **283 img/s** | 92 s |
| `dinov2` | ~7 img/s | **306 img/s** | 85 s |
| `chinese-clip-ft` (fp16) | not measured | **445 img/s** | 59 s |

Note that `chinese-clip-large` costs essentially nothing over base (354 vs 358 img/s) despite being
2× the parameters — fp16 pays for the extra size. That is what makes it the easy default.

The GPU move paid off exactly where it mattered: the ViT backends went from ~1 hour per pass to
~1 minute (**~45×**), so `--limit` is no longer needed for iteration. ResNet gained ~11×.

**`use_fast=True` has been tried and rejected — do not "fix" it again.** The ViT backends are
preprocessing-bound, not GPU-bound (image load/decode is only ~10,000 img/s, so it is not the
culprit), and the HF fast processor *is* 3.1× faster in isolation (1700 vs 548 img/s). It still
loses end to end:

| | isolated preprocess | full corpus, chinese-clip | interleaved A/B, 4k imgs |
|---|---|---|---|
| `use_fast=False` | 548 img/s | **358 img/s** | **595.9 ± 8.9 img/s** |
| `use_fast=True` | 1700 img/s | 311 img/s | 521.8 ± 144.6 img/s |

The fast path uses multi-threaded torch CPU ops that contend with the vLLM workers sharing this box;
its throughput swings roughly 2× run to run (320–650 img/s across three interleaved reps) and it was
slower on both backends at full scale. The slow PIL path is single-threaded and therefore stable.
Both processors are set explicitly to `use_fast=False` in `feature_extractors.py`, which also
silences the HF deprecation warning.

Quality is *not* the reason to prefer either: the two paths agree to cosine ≥ 0.99996 on all 26,044
embeddings, so HF's "minor differences in outputs" warning is immaterial here.

**Caveat on every throughput number in this file:** the H100 is shared with four vLLM engines whose
load varies, so timings are reproducible to maybe ±10%, not better. The *quality* metrics are exact
and deterministic; the speed metrics are indicative.

## Current status — pick up here

**The project has shipped**: all six backends benchmarked over the full corpus, a fine-tune that
moved real-scan hit@1 from 0.0847 to 0.5932, a labelled Part 2 evaluation, a metadata reranker with
a non-circular evaluation, and the submission package. Everything is written up in
**[`BENCHMARK.md`](BENCHMARK.md)** — the deliverable — with `report/paper.tex` as the submitted
report. Everything below is the condensed version; `BENCHMARK.md` has the full analysis, method,
and limitations.

**The two things actually left open**, in priority order:

1. **Run `evaluate_rerank.py --suite scans --backend chinese-clip-ft --dtype fp16`.** The reranker's
   negative result (BENCHMARK.md §15) was measured with a zero-shot stage 1, whose radical-voting
   step is right only 11.86% of the time. With the finetuned stage 1 that jumps to 0.6610, so the
   question "does reranking help" has not been asked where it can answer yes. Cheap, code is
   already there, and `--suite scans` is uncontaminated.
2. **A human relevance evaluation.** Backend agreement is 8–13%, so the proxy metrics are doing real
   separating work — which also means a human judgement would be decisive rather than confirmatory.

**One methodological caveat to carry into any write-up:** `finetune_glyph.py` selects `best.pt` by
maximising hit@1 on `test_images/` (line ~472), the same 59 scans every headline number is reported
on. No pixel of `test_images/` enters training — `_assert_test_images_unseen` guarantees that — but
the *checkpoint choice* does see them, across ~10–30 evaluations on a set with a ±13-point CI. The
0.5932 vs 0.0847 conclusion survives easily (35/59 vs 5/59 is far larger than any selection effect),
but the absolute figure is optimistically biased. Either say so in BENCHMARK.md §11 and the paper,
or select on the held-out classes instead and report the test set once.

Reproduce end to end (~6 minutes total on GPU, embeddings are cached afterwards):

```bash
.venv/bin/python search_all_chars_in_corpus.py --backend resnet18           --device cuda
.venv/bin/python search_all_chars_in_corpus.py --backend resnet18-gray      --device cuda
.venv/bin/python search_all_chars_in_corpus.py --backend chinese-clip       --device cuda --batch-size 32
.venv/bin/python search_all_chars_in_corpus.py --backend chinese-clip-large --device cuda --batch-size 32 --dtype fp16
.venv/bin/python search_all_chars_in_corpus.py --backend chinese-clip-huge  --device cuda --batch-size 32 --dtype fp16
.venv/bin/python search_all_chars_in_corpus.py --backend dinov2             --device cuda --batch-size 32
.venv/bin/python benchmark_extractors.py   # defaults to all six
.venv/bin/python benchmark_search.py       # exact-vs-ANN study, reads the caches
.venv/bin/python -m pytest -q              # 55 passed, needs neither GPU nor torch
```

`benchmark_extractors.py` finds each backend's cache whichever precision it was built at, so the
mixed fp32/fp16 comparison above works without extra flags.

### Results — full corpus (26,044 glyphs), top-K = 20

| backend | dtype | dim | radical@k ↑ | stroke_mae@k ↓ | ids_jaccard@k ↑ |
|---|---|---|---|---|---|
| `resnet18` (baseline) | fp32 | 512 | 0.2316 | 2.8942 | 0.1031 |
| `resnet18-gray` | fp32 | 512 | 0.2202 | 2.5658 | 0.1020 |
| `chinese-clip` | fp32 | 512 | 0.4843 | 2.6296 | 0.1997 |
| **`chinese-clip-large`** | fp16 | 768 | **0.5361** | 2.6992 | 0.2113 |
| `chinese-clip-huge` | fp16 | 1024 | 0.5147 | 2.6547 | **0.2144** |
| `dinov2` | fp32 | 1536 | 0.3081 | **2.3397** | 0.1251 |

Five things to carry forward:

1. **Chinese-CLIP wins by a wide margin, and `-large` is the pick.** `chinese-clip-large` reaches
   radical@k **2.31×** the baseline (0.5361 vs 0.2316) and ids_jaccard **2.05×** (0.2113 vs 0.1031).
   The earlier 200-character trial on the base model predicted 0.398 and was, if anything,
   *pessimistic*. This vindicates the original hypothesis that a Han-glyph-aware prior beats an
   ImageNet CNN.
2. **Scale helps, then saturates and reverses.** base → large is +10.7% relative on radical@k
   (0.4843 → 0.5361), but large → huge is **−4.0%** (0.5361 → 0.5147) while costing 33% more
   dimensions and 20% more time. `huge` edges out `large` only on ids_jaccard (0.2144 vs 0.2113).
   Do not assume the next bigger checkpoint is better — the curve has already turned. If someone
   proposes ViT-G, this is the evidence that it probably is not worth the VRAM.
3. **DINOv2 is complementary to the CLIP family, not simply worse.** Every `chinese-clip*` backend
   wins on radical and IDS overlap — the *structural/semantic* signals — while DINOv2 holds the best
   stroke-count MAE (2.3397), i.e. it tracks visual density better. Worth saying explicitly in the
   report instead of declaring one winner; an ensemble or a rerank is a natural follow-up (and
   overlaps Person 2's scope).
4. **The `conv1` random-init hypothesis still does not pan out** — confirmed at full scale.
   `resnet18-gray` improves stroke MAE (2.5658 vs 2.8942) but *hurts* radical@k (0.2202 vs 0.2316)
   and leaves IDS flat. Keep reporting this negative result honestly; it is what motivated moving to
   ViT backends in the first place.
5. **Backend agreement is very low — mean top-20 overlap is 1.5–2.6 of 20 (8–13%).** Even the two
   closest (`chinese-clip` / `dinov2`, 2.57) disagree on ~87% of retrieved neighbours. The backends
   are not converging on one "true" answer, which means the proxy metrics are doing real work in
   separating them, *and* that a human-judged evaluation (Person 3) will be decisive rather than
   confirmatory.

Qualitatively the contact sheets match the metrics: for a 犭-radical query, `chinese-clip` returns
猎狺猜獍獖獐猗 (nearly all same radical) where `resnet18` returns 猜指𢯡褃棈偝精 (mixed 犭/扌/礻/米).

**Reproducibility is confirmed across machines.** The `resnet18` / `resnet18-gray` numbers above
match the values previously recorded on a different box (Python 3.14 / torch 2.13) to within 3e-4 —
`set_deterministic()` is doing its job; do not weaken it.

Any `output/embeddings_*.npz` from a previous machine can be reused (the benchmark reads them by
filename), but recomputing on GPU is fast enough that copying is rarely worth it.

### Faiss: keep `IndexFlatIP`, the ANN question is already settled

`benchmark_search.py` compared exact Flat against HNSW / IVFFlat / IVFPQ. HNSW is the best ANN
option — 99.1% recall@20 at 20× the speed — but at this corpus size the speedup buys nothing:

| N | exact ms/query | full-corpus search | embed @354 img/s | search share |
|---|---|---|---|---|
| **26,044 (today)** | 0.195 | 5 s | 74 s | **6%** |
| 100,000 | 0.633 | 63 s | 282 s | 18% |
| 500,000 | 5.484 | 2,742 s | 1,412 s | 66% |
| 1,000,000 | 7.532 | 7,532 s | 2,825 s | 73% |

Embedding is O(N) but a full-corpus sweep is O(N²), so search's share only grows. **The crossover is
around N ≈ 300,000** — that is where search overtakes embedding. Below it, trading exactness for
speed optimizes 6% of the runtime while giving up the one thing that is actually scarce here
(retrieval quality). Above ~500k it becomes mandatory: at 1M, exact costs 2.1 hours per sweep versus
~4.7 minutes for HNSW at `efSearch=64`.

So: **do not swap the index.** If the corpus ever passes ~300k, switch to `IndexHNSWFlat` with
`efSearch=64` and re-measure recall — recall was only ever measured at 26k and HNSW recall degrades
as N grows, so the 99.1% figure does not transfer. IVFPQ is a dead end regardless: 0.60 recall.

Latency beyond 26k was measured on random vectors (timing barely depends on the data); recall never
was. `benchmark_search.py` keeps the two separate on purpose — do not merge them.

**Part 2 is upgraded, working, and now validated on real scans** — `search_use_QuocNgu_mapping.py`
defaults to embeddings and reuses Part 1's cache. It is also **labelled**: all 59 scans have a
`correct_unicode` in `output/label_sheets/label_template.csv`, so Part 2 has a real accuracy number
(0.4746 embedding / 0.4407 histogram). Read the caveats in BENCHMARK.md §10.6 before quoting it.

## Gotchas already hit (don't rediscover these)

- **`get_image_features` changed shape between transformers versions.** In v5 it returns a
  `BaseModelOutputWithPooling` whose `pooler_output` is the projected embedding; v4 returned the
  tensor. `ChineseClipExtractor._forward` branches on `torch.is_tensor`.
- **5,164 of the 31,208 table rows have no image file.** They are filtered out, and the count is
  printed once — the original script printed one "Not found" line per miss, which buried real
  output. Never assume `len(df) == len(images)`.
- **The original `preprocess` passed `cv2.INTER_LINEAR` as torchvision's `interpolation`.** It
  happened to work because the enum values collide; it is now
  `transforms.InterpolationMode.BILINEAR`.
- **`RADICAL` values carry a residual-stroke suffix** (`'nhất 一 (+3 nét)'`). Strip `(...)` before
  comparing radicals or every character looks like it has a unique radical.
- **Piping a long-running script through `grep` block-buffers its progress output.** Progress looked
  frozen for many minutes during the first full run. Write to a file or drop the pipe.
- **Console encoding on Windows is cp1252 and dies on Sino-Nôm glyphs.** Any script that prints
  characters needs `sys.stdout.reconfigure(encoding='utf-8')`. Both search scripts now do; keep it
  if you touch their `main()`.
- **Write CSVs as `utf-8-sig`.** Without the BOM Excel opens Sino-Nôm output as mojibake. Both
  search scripts pass it explicitly.
- **`RADICAL` is a dictionary convention, not a visual property** — it is absent from the
  character's own IDS decomposition 43.3% of the time. Treat `radical@k` as a proxy with a ~0.73
  ceiling (see `analyze_metric_ceiling.py`), never as an accuracy that should approach 1.0.

## Known directions for improvement (from README.txt)

- Swap the ResNet18 CNN feature extractor for a Transformer-based embedding model (e.g.
  DINOv2+MetaCLIP, as suggested by a collaborator) for better feature quality.
- Consider a similarity approach based on decomposing Sino-Nôm characters into their structural
  components, per the paper in `materials/2309.01083v1.pdf` (implementation reference:
  github.com/FudanVI/FudanOCR/tree/main).
- [Chinese-CLIP (OFA-Sys)](https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16) has also
  been suggested as a candidate feature extractor to try instead of ResNet18.

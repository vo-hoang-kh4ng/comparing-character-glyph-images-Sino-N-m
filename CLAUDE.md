# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A university group project ("Image Comparison") for finding visually-similar Sino-Nôm (Chữ Nôm /
Hán-Nôm) characters by comparing character glyph images. This is not a conventional application —
it's a set of standalone analysis scripts plus their input data (Excel dictionaries and a corpus of
character images). There is no build system, package manifest, or test suite.

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

Person 1's work is functionally complete but **the full-corpus numbers for the two ViT backends are
not yet measured** — see "Current status" below. Persons 2 and 3 have not started; deliberately do
not pre-empt their scope. In particular the proxy metrics in `benchmark_extractors.py` exist only to
rank backends against each other — they are *not* the real evaluation, which is Person 3's task.

## Data files (read-only inputs, do not regenerate by hand)

- `images.zip` — one glyph image per character, named `<UNICODE_CODEPOINT_HEX>.jpg` (e.g.
  `2A0AF.jpg`). Scripts expect this extracted to an `./images/` folder alongside them.
- `final_characteristics-v2.xlsx` — master character table. Key columns used by the scripts:
  `UNICODE` (hex codepoint, matches image filenames) and `CHAR` (the actual glyph). Also carries
  linguistic metadata (`AM_NOM`, `STROKE_NUM`, `RADICAL`, `SHAPE_MORPH`, `STROKE`, `VARIATION`,
  `P_ANCIENT`, `P_MODERN`) not currently consumed by the scripts.
- `QuocNgu_SinoNom_Dic.xlsx` — mapping table with columns `QuocNgu` (modern Vietnamese romanized
  word) and `SinoNom` (corresponding Sino-Nôm character(s)). Used to restrict similarity search to
  characters sharing a given Quốc Ngữ reading.
- `materials/2309.01083v1.pdf` — reference paper ("Sino-Nom character decomposition") cited as a
  direction for future improvement (see README.txt).

Neither xlsx file has been observed to require anything beyond `pandas.read_excel` (openpyxl
engine); don't add unnecessary preprocessing.

## The scripts

### `feature_extractors.py` — pluggable embedding backends

All backends share one contract: `build_extractor(name).embed_paths(paths) -> (N, D) float32`,
L2-normalized so inner product == cosine similarity (what the Faiss `IndexFlatIP` downstream
assumes). Backends:

| name | model | dim | notes |
|---|---|---|---|
| `resnet18` | ImageNet ResNet18 | 512 | **Original baseline, kept bit-for-bit.** Patching `conv1` to 1 channel *discards* the pretrained first-layer filters and re-initializes them randomly — everything downstream is built on a random edge detector. This is the flaw the other backends fix; keep it faithful so "before" numbers stay honest. |
| `resnet18-gray` | ImageNet ResNet18 | 512 | Same net, but `conv1` is seeded by summing the pretrained RGB filters, so the ImageNet prior survives. Isolates how much of the baseline's weakness is that one line. |
| `chinese-clip` | OFA-Sys/chinese-clip-vit-base-patch16 | 512 | Image tower, projected embedding (`get_image_features`). Pretrained on Chinese image–text pairs, so its prior is closest to Han glyphs. |
| `dinov2` | facebook/dinov2-base | 1536 | CLS token ⊕ mean-pooled patch tokens — CLS carries global shape, patch mean carries stroke texture. |

`transformers` v5 returns an output object from `get_image_features` (the projection lives in
`pooler_output`) while v4 returned the tensor directly; `ChineseClipExtractor` handles both.

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

Embeddings are cached to `output/embeddings_<backend>.npz`, keyed by the UNICODE list they were
built from, and reused across runs — this is what makes `benchmark_extractors.py` cheap. Output
goes to `output/output_top_k_similar_char_<backend>.xlsx`/`.csv` (columns: `Input Character`,
`Top K Similar Characters`, `Similarity Scores`). `./output/` is now created automatically.

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
item. Also emits pairwise top-K overlap between backends, and `contact_sheet_<backend>.png`
(query glyph in a red box + its top-5 neighbours) for eyeballing.

### `search_use_QuocNgu_mapping.py` — same-reading top-K similarity

Given a single Vietnamese word (`input_text`, hardcoded near the top of the file — several examples
are present as commented-out alternatives) and a corresponding test image (`test_image_path`),
finds the top-10 most similar characters *only among characters that share that Quốc Ngữ reading*
(looked up via `QuocNgu_SinoNom_Dic.xlsx`, then mapped to `UNICODE` via `final_characteristics-v2.xlsx`).

Pipeline is deliberately simpler than the other script (per README.txt: comparison set is small, so
no Faiss index is used — plain sort is enough): grayscale → resize 100x100 → Gaussian blur → Canny
edge detection → Otsu binary threshold → contrast stretch → 256-bin histogram → L2 distance between
histograms converted to a similarity score (`1 / (1 + distance)`).

To run for a different word: edit `input_text` and `test_image_path` at the top of the file (there
is no CLI argument parsing), and ensure the referenced image exists under `./test_images/`.

Output: `./output/output_histogram_similarity.xlsx` and `.csv` (columns: `UNICODE`, `SinoNom`,
`Similarity`). Same caveat — `./output/` must exist first.

## Setting up (nothing but code is in git)

Bulk data (`images.zip`, both `.xlsx` files, `materials/`, `phan_cong_cong_viec.xlsx`) is gitignored
— too large / not source. It is distributed via a Google Drive folder linked in README.md.
`./output/`, `./images/`, and `./test_images/` are gitignored too (generated or local-only).

So a fresh clone needs:

```bash
pip install -r requirements.txt
# download from the Drive link in README.md, then:
unzip images.zip -d images          # -> ./images/<UNICODE>.jpg
# final_characteristics-v2.xlsx and QuocNgu_SinoNom_Dic.xlsx go in the repo root
```

`search_all_chars_in_corpus.py` creates `./output/` itself; `search_use_QuocNgu_mapping.py` still
requires it to exist, and also needs `./test_images/<name>.jpg` matching its `test_image_path`.

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

### Running on GPU

Pass `--device cuda`. Everything here is **inference only** — no training — so it is cheap:
ViT-B/16 fp32 weights are ~0.4 GB, and `--batch-size 32` fits comfortably in ~3 GB of VRAM. That
matters because the target server's H100 is mostly occupied by vLLM workers (~3.5 GB free at last
check); if VRAM is tight, keep the batch small and consider
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

A future contrastive/Siamese fine-tune (Person 3's stretch direction) would *not* fit in that
headroom — it needs gradients and optimizer state, so it requires a real GPU allocation.

### Throughput (measured, 16-core CPU, no GPU, batch 64)

| backend | img/s | full 26k pass |
|---|---|---|
| `resnet18` | ~347 | ~75 s |
| `resnet18-gray` | ~403 | ~65 s |
| `chinese-clip` | ~5–9 | ~1 hour |
| `dinov2` | ~7 | ~1 hour |

The ViT backends are painful on CPU — this is exactly why the work is moving to the GPU server.
Use `--limit N` while iterating.

## Current status — pick up here

The immediate next step is to **finish Person 1's benchmark on GPU**:

```bash
python search_all_chars_in_corpus.py --backend resnet18      --device cuda
python search_all_chars_in_corpus.py --backend resnet18-gray --device cuda
python search_all_chars_in_corpus.py --backend chinese-clip  --device cuda --batch-size 32
python search_all_chars_in_corpus.py --backend dinov2        --device cuda --batch-size 32
python benchmark_extractors.py --backends resnet18 resnet18-gray chinese-clip dinov2
```

Then write up the comparison (speed + proxy metrics + contact sheets) as Person 1's deliverable.

**What is already known.** Full corpus (26,044 glyphs), top-K = 20, both ResNet backends complete:

| backend | radical@k ↑ | stroke_mae@k ↓ | ids_jaccard@k ↑ |
|---|---|---|---|
| `resnet18` | 0.2315 | 2.8939 | 0.1031 |
| `resnet18-gray` | 0.2202 | 2.5659 | 0.1020 |

Two things to carry forward:

1. **The `conv1` random-init hypothesis did not pan out.** Fixing it (`resnet18-gray`) improved
   stroke MAE but slightly *hurt* radical@k and left IDS overlap flat — it is not the main reason
   the baseline is weak. Report this honestly rather than quietly dropping the variant; a negative
   result is still a result, and it means the gain (if any) has to come from the ViT backends.
2. **Chinese-CLIP looked clearly better on a 200-character trial** (radical@k 0.398 vs 0.278,
   ids_jaccard 0.102 vs 0.067 against `resnet18`). Do **not** quote those numbers as results — a
   200-row `--limit` slice is a contiguous, unrepresentative block of the table. They are only a
   reason to expect the full run to be favourable. The full-corpus ViT run was started on CPU and
   deliberately killed once the GPU server became the plan, so no ViT cache exists yet.

Any `output/embeddings_*.npz` from a previous machine can be reused (the benchmark reads them by
filename), but recomputing on GPU is fast enough that copying is rarely worth it.

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
  characters needs `sys.stdout.reconfigure(encoding='utf-8')`. CSV output uses `utf-8-sig` so Excel
  opens it correctly.

## Known directions for improvement (from README.txt)

- Swap the ResNet18 CNN feature extractor for a Transformer-based embedding model (e.g.
  DINOv2+MetaCLIP, as suggested by a collaborator) for better feature quality.
- Consider a similarity approach based on decomposing Sino-Nôm characters into their structural
  components, per the paper in `materials/2309.01083v1.pdf` (implementation reference:
  github.com/FudanVI/FudanOCR/tree/main).
- [Chinese-CLIP (OFA-Sys)](https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16) has also
  been suggested as a candidate feature extractor to try instead of ResNet18.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A small research codebase for finding visually-similar Sino-Nôm (Chữ Nôm / Hán-Nôm) characters by
comparing character glyph images. This is not a conventional application — it's two standalone
analysis scripts plus their input data (Excel dictionaries and a corpus of character images). There
is no build system, package manifest, or test suite.

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

## The two scripts

### `search_all_chars_in_corpus.py` — corpus-wide top-K similarity

For every character in `final_characteristics-v2.xlsx`, finds the top-K (default 20) most visually
similar other characters across the *entire* image corpus.

Pipeline: load each `./images/<UNICODE>.jpg` as grayscale → resize to 100x100 → normalize →
extract a feature embedding using a pretrained **ResNet18** (first conv layer patched to accept
1-channel input; classification head removed, so output is the 512-d pooled feature) → L2-normalize
embeddings → build a **Faiss** `IndexFlatIP` (inner product ≈ cosine similarity since vectors are
normalized) → for each character, query its own top-(K+1) nearest neighbors and drop the self-match.

Image loading/feature extraction is parallelized with `ThreadPoolExecutor`. Determinism is enforced
via `set_deterministic()` (seeds random/numpy/torch, disables cudnn benchmarking) — preserve this
if you touch the model or data pipeline, since results are expected to be reproducible.

Output: `./output/output_top_k_similar_char.xlsx` and `.csv` (columns: `Input Character`,
`Top 20 Similar Characters`). **The `./output/` directory is not created automatically — create it
before running, or the script will fail on write.**

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

## Running the scripts

`requirements.txt` lists the third-party packages, inferred from imports:

```
pandas openpyxl numpy torch torchvision pillow faiss-cpu opencv-python scikit-learn
```

Install with `pip install -r requirements.txt`. (`pandas`, `numpy`, `openpyxl`, `pillow` are
already available in this environment; `torch`, `torchvision`, `faiss`, `cv2`/opencv-python, and
`sklearn` are not.)

Bulk data (`images.zip`, both `.xlsx` files, `materials/`) is gitignored — not tracked in version
control due to size. It's distributed via a separate Google Drive folder (see README.md for the
link and expected local layout). `./output/` and `./test_images/` are also gitignored since they're
generated/local-only.

Before running either script:
1. Extract `images.zip` to `./images/` in the repo root (script paths are relative, e.g.
   `./images/<UNICODE>.jpg`).
2. Create `./output/` (scripts write here and do not create it themselves).
3. For `search_use_QuocNgu_mapping.py`, also provide `./test_images/<name>.jpg` matching whatever
   `test_image_path` is set to.

```
python search_all_chars_in_corpus.py
python search_use_QuocNgu_mapping.py
```

## Known directions for improvement (from README.txt)

- Swap the ResNet18 CNN feature extractor for a Transformer-based embedding model (e.g.
  DINOv2+MetaCLIP, as suggested by a collaborator) for better feature quality.
- Consider a similarity approach based on decomposing Sino-Nôm characters into their structural
  components, per the paper in `materials/2309.01083v1.pdf` (implementation reference:
  github.com/FudanVI/FudanOCR/tree/main).
- [Chinese-CLIP (OFA-Sys)](https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16) has also
  been suggested as a candidate feature extractor to try instead of ResNet18.

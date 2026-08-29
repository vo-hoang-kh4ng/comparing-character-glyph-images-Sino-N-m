"""Đánh giá trên ảnh query THẬT (`test_images/`) — nhãn lấy từ chính tên file.

Tên file trong `test_images/` được gõ Telex, ví dụ `cofn.jpg` = "còn", `nguwowsi.jpg` = "người".
Giải mã ngược ra chữ Quốc Ngữ rồi tra `QuocNgu_SinoNom_Dic.xlsx` sẽ được **tập ký tự Sino-Nôm
đúng** cho ảnh đó. Đây là nhãn thật, không phải proxy như `radical@k` — nên là phép đo có giá trị
nhất trong đồ án.

Hai phép đo:

Phần 1 (`--part 1`)
    Tìm ảnh query trên **toàn bộ 26k corpus**, rồi hỏi: ký tự trả về có đọc đúng là từ đó không?
    ``hit@k`` = tỉ lệ ảnh mà trong top-k có ít nhất một ký tự thuộc tập đúng. Baseline ngẫu nhiên
    là |tập đúng| / 26.044, cỡ 1e-4 — nên đây là bài kiểm tra khó và trung thực.

Phần 2 (`--part 2`)
    Chỉ xếp hạng trong nhóm cùng âm Quốc Ngữ. Không có nhãn "đúng ký tự nào", nên dùng **pseudo-label**:
    khi Phần 1 trả về top-1 toàn corpus mà ký tự đó *cũng* nằm trong nhóm ứng viên, hai tín hiệu độc
    lập cùng chỉ vào một chữ — coi đó là đáp án. Rồi đo `histogram` có xếp đúng chữ đó lên đầu không.
    (`embedding` đúng 100% theo định nghĩa vì cùng hàm chấm điểm với Phần 1 — con số cần nhìn là của
    histogram.)

Ví dụ
-----
    python evaluate_test_images.py --backend chinese-clip-large --dtype fp16 --device cuda
    python evaluate_test_images.py --part 2
"""

import argparse
import os
import sys
import unicodedata

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from feature_extractors import BACKENDS, DTYPES, build_extractor
from search_all_chars_in_corpus import CHAR_TABLE, cache_tag, load_corpus, set_deterministic
from search_use_QuocNgu_mapping import QUOCNGU_TABLE, extract_histogram

#: Nguyên âm có dấu phụ -> cách gõ Telex.
TELEX_BASE = {"̂": {"a": "aa", "e": "ee", "o": "oo"},
              "̆": {"a": "aw"},
              "̛": {"o": "ow", "u": "uw"}}
#: Thanh điệu -> chữ cái Telex.
TELEX_TONE = {"̀": "f", "́": "s", "̉": "r", "̃": "x", "̣": "j"}


def telex_variants(word):
    """Mọi cách gõ Telex hợp lệ của một từ — dấu thanh đặt ngay sau nguyên âm hoặc ở cuối.

    'còn' -> {'cofn', 'confn'... } thực tế {'cofn', 'conf'}; cả hai đều gặp trong test_images/.
    """
    letters, tone, tone_at = [], "", None
    for ch in unicodedata.normalize("NFD", word.lower()):
        if ch in TELEX_TONE:
            tone, tone_at = TELEX_TONE[ch], len("".join(letters))
        elif ch in TELEX_BASE:
            letters[-1] = TELEX_BASE[ch].get(letters[-1], letters[-1])
        else:
            letters.append("dd" if ch == "đ" else ch)
    base = "".join(letters)
    if not tone:
        return {base}
    return {base[:i] + tone + base[i:] for i in range(tone_at, len(base) + 1)}


#: Tên file gõ sai Telex trong bộ scan gốc, sửa về đúng từ. Giữ ở đây thay vì đổi tên file để
#: ``test_images/`` còn khớp byte-đối-byte với zip gốc.
FILENAME_FIXES = {
    "dau": "dâu",            # thiếu dấu mũ: "bể dâu" (Kiều câu 3), khác với ddau.jpg = "đau"
    "nguwowsi": "người",     # gõ 's' (sắc) thay vì 'f' (huyền)
}


def decode_filenames(test_dir, quocngu):
    """Tên file -> từ Quốc Ngữ. Trả về (khớp được, không khớp được)."""
    lookup = {}
    for word in {str(w) for w in quocngu["QuocNgu"].dropna()}:
        for variant in telex_variants(word):
            lookup.setdefault(variant, set()).add(word)

    matched, unmatched = [], []
    for name in sorted(os.listdir(test_dir)):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        # 'phong_1.jpg' là ảnh thứ hai của cùng một từ.
        head, _, tail = stem.rpartition("_")
        key = head if head and tail.isdigit() else stem
        if key in FILENAME_FIXES:
            matched.append((os.path.join(test_dir, name), stem, FILENAME_FIXES[key]))
            continue
        words = lookup.get(key)
        if not words:
            unmatched.append(stem)
        elif len(words) > 1:
            unmatched.append(f"{stem} (nhập nhằng: {sorted(words)})")
        else:
            # KHÔNG dùng words.pop(): set này nằm trong `lookup`, pop sẽ làm rỗng nó và
            # ảnh thứ hai của cùng một từ (phong_1.jpg) rơi vào `unmatched`.
            matched.append((os.path.join(test_dir, name), stem, next(iter(words))))
    return matched, unmatched


def ground_truth(quocngu, chars, word, have_image):
    """UNICODE của mọi ký tự đọc là `word` và có ảnh trong corpus."""
    sino = set(quocngu[quocngu["QuocNgu"] == word]["SinoNom"])
    hits = chars[chars["CHAR"].isin(sino)]["UNICODE"].astype(str)
    return {u for u in hits if u in have_image}


def normalize_glyph(path, out_path, box=0.80, size=224):
    """Cắt sát nét mực, đệm vào khung vuông, chữ chiếm `box` bề rộng khung.

    Ảnh corpus là bản render font sạch 70x70 vuông; ảnh trong ``test_images/`` là scan, không vuông
    (96x128, 88x96...) và nét dày hơn — resize thẳng về 224x224 làm **méo tỉ lệ chữ**. Hàm này đưa
    hai bên về cùng khung hình.

    Đã đo: hiệu quả nằm trong khoảng nhiễu ở n=56 (xem BENCHMARK.md mục 10), nên **không bật mặc
    định**. Giữ lại vì nó là cách sửa đúng cho phần lệch *hình học*, và sẽ cần khi có tập test lớn hơn.
    """
    array = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    ink = array < int(array.max()) - 40
    if not ink.any():
        ink = array < 250
    if not ink.any():
        Image.fromarray(array).resize((size, size)).save(out_path)
        return out_path
    ys, xs = np.where(ink)
    crop = array[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    height, width = crop.shape
    scale = int(size * box) / max(height, width)
    new_h, new_w = max(1, round(height * scale)), max(1, round(width * scale))
    canvas = Image.new("L", (size, size), 255)
    canvas.paste(Image.fromarray(crop).resize((new_w, new_h), Image.LANCZOS),
                 ((size - new_w) // 2, (size - new_h) // 2))
    canvas.save(out_path)
    return out_path


def normalized_copies(paths, cache_dir, keys):
    os.makedirs(cache_dir, exist_ok=True)
    out = []
    for path, key in zip(paths, keys):
        target = os.path.join(cache_dir, f"{key}.png")
        if not os.path.exists(target):
            normalize_glyph(path, target)
        out.append(target)
    return out


#: Thang tin cậy cho cột ``confidence`` trong ``label_template.csv``. Đây là **đánh giá của người
#: đọc ảnh, KHÔNG suy ra từ điểm số nào của mô hình** — cố tình như vậy: Phần 2 dùng chính cột này
#: để lọc ra tập nhãn chắc chắn rồi chấm lại mô hình trên đó, nên nếu confidence lấy từ điểm
#: similarity thì phép kiểm đó thành vòng tròn (lấy mô hình chấm mô hình).
#:
#:   high — nét quyết định đọc được; mọi ứng viên còn lại khác ở bộ thủ hoặc ở số nét thấy rõ.
#:   med  — chọn được một ứng viên, nhưng còn ≥1 ứng viên là biến thể gần trùng chỉ khác nhau ở
#:          chi tiết đã bị mờ/mất trong bản scan (ví dụ 𣘛 vs 橷, cùng 木+兜).
#:   low  — nét quyết định không đọc được; lựa chọn dựa vào văn cảnh (Truyện Kiều) nhiều hơn hình.
#:
#: Quy ước khi hình và văn cảnh mâu thuẫn: **hình thắng**, và hạ confidence xuống med.
CONFIDENCE_LEVELS = ("high", "med", "low")


def label_sheets(labelled, images, chars, out_dir, thumb=96, max_candidates=None):
    """Phiếu gán nhãn: ảnh query (khung đỏ) + mọi ứng viên đánh số, để nhóm chọn đáp án đúng.

    Đây là thứ còn thiếu để chấm điểm Phần 2 — không ai biết ảnh scan đó *là* ký tự nào cho tới khi
    có người đọc. Kèm `label_template.csv` để điền: ``correct_index`` (số thứ tự trong ``candidates``)
    và ``confidence`` (xem ``CONFIDENCE_LEVELS``).

    ``max_candidates=None`` nghĩa là **liệt kê hết**. Trước đây mặc định là 24, và vì `n_candidates`
    vẫn ghi tổng số thật nên phiếu tự mâu thuẫn: `gia` có 50 ứng viên, đáp án đúng nằm ở vị trí 32,
    tức người gán nhãn không thể thấy nó trong phiếu lẫn trong ảnh sheet. Đừng đặt lại giá trị này
    trừ khi chỉ muốn xem nhanh.
    """
    os.makedirs(out_dir, exist_ok=True)
    char_of = dict(zip(chars["UNICODE"].astype(str), chars["CHAR"]))
    template = []

    for path, stem, word, gt in labelled:
        cand = sorted(gt) if max_candidates is None else sorted(gt)[:max_candidates]
        cols = min(len(cand) + 1, 9)
        rows = -(-(len(cand) + 1) // cols)
        sheet = Image.new("RGB", (cols * (thumb + 8) + 8, rows * (thumb + 18) + 8), "white")
        draw = ImageDraw.Draw(sheet)
        cells = [(path, "QUERY")] + [(os.path.join(images, f"{u}.jpg"), f"{i+1}:{u}")
                                     for i, u in enumerate(cand)]
        for i, (src, tag) in enumerate(cells):
            x = 8 + (i % cols) * (thumb + 8)
            y = 8 + (i // cols) * (thumb + 18)
            sheet.paste(Image.open(src).convert("L").resize((thumb, thumb)), (x, y))
            draw.text((x, y + thumb + 2), tag, fill="black")
            if i == 0:
                draw.rectangle([x - 2, y - 2, x + thumb + 1, y + thumb + 1], outline="red", width=2)
        out = os.path.join(out_dir, f"{stem}.png")
        sheet.save(out)
        template.append({
            "file": stem, "word": word, "n_candidates": len(gt),
            # Kèm luôn codepoint: người gán nhãn điền `correct_unicode` bằng cách chép lại, không
            # phải tự tra ngược từ glyph (nhiều biến thể trông giống hệt nhau).
            "candidates": " ".join(f"{i+1}={char_of.get(u, u)}({u})" for i, u in enumerate(cand)),
            "correct_index": "", "correct_unicode": "", "confidence": "",
        })

    csv_path = os.path.join(out_dir, "label_template.csv")
    pd.DataFrame(template).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Đã ghi {len(template)} phiếu vào {out_dir}/ và {csv_path}")
    print("  -> điền cột correct_index rồi chạy lại với --labels để chấm điểm Phần 2 thật sự")
    return csv_path


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", default="both", choices=("1", "2", "both"))
    parser.add_argument("--backend", default="chinese-clip-large", choices=BACKENDS)
    parser.add_argument("--dtype", default="fp16", choices=sorted(DTYPES))
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--test-images", default="./test_images")
    parser.add_argument("--images", default="./images")
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--normalize", choices=("query", "both"), default=None,
                        help="cắt-đệm về khung vuông trước khi nhúng: 'query' chỉ ảnh test "
                             "(nhanh), 'both' cả corpus (phải nhúng lại, ~2,5 phút trên GPU)")
    parser.add_argument("--sheets", action="store_true",
                        help="chỉ xuất phiếu gán nhãn cho nhóm, không chấm điểm")
    parser.add_argument("--sheet-dir", default="./output/label_sheets")
    parser.add_argument("--min-confidence", choices=CONFIDENCE_LEVELS,
                        help="chỉ chấm trên các nhãn có confidence từ mức này trở lên "
                             "(vd --min-confidence high => bỏ qua nhãn med/low). Dùng để kiểm tra "
                             "kết luận có phụ thuộc vào mấy nhãn không chắc hay không.")
    parser.add_argument("--labels", help="CSV do nhóm gán nhãn (từ --sheets), cột "
                                         "correct_index hoặc correct_unicode")
    args = parser.parse_args()

    set_deterministic()
    quocngu = pd.read_excel(QUOCNGU_TABLE)
    chars = pd.read_excel(CHAR_TABLE)

    matched, unmatched = decode_filenames(args.test_images, quocngu)
    print(f"Giải mã tên file: {len(matched)} khớp, {len(unmatched)} bỏ qua")
    if unmatched:
        print(f"  bỏ qua: {', '.join(unmatched)}")

    cache_path = os.path.join(args.output_dir, f"embeddings_{cache_tag(args.backend, args.dtype)}.npz")
    if not os.path.exists(cache_path):
        raise SystemExit(f"Chưa có {cache_path} — chạy search_all_chars_in_corpus.py trước.")
    cached = np.load(cache_path, allow_pickle=True)
    corpus = cached["embeddings"].astype("float32")
    unicodes = np.array([str(u) for u in cached["unicodes"]])
    have_image = set(unicodes)
    row_of = {u: i for i, u in enumerate(unicodes)}
    print(f"Corpus: {len(unicodes)} ký tự từ {cache_path}")

    labelled = [(p, s, w, gt) for p, s, w in matched
                if (gt := ground_truth(quocngu, chars, w, have_image))]
    dropped = [s for p, s, w in matched if not ground_truth(quocngu, chars, w, have_image)]
    if dropped:
        print(f"  không có ứng viên nào có ảnh, bỏ: {', '.join(dropped)}")
    if not labelled:
        raise SystemExit("Không còn ảnh nào để đánh giá.")

    if args.sheets:
        label_sheets(labelled, args.images, chars, args.sheet_dir)
        return

    extractor = build_extractor(args.backend, device=args.device, dtype=args.dtype)
    query_paths = [p for p, _, _, _ in labelled]
    if args.normalize:
        cache_dir = os.path.join(args.output_dir, "normalized")
        query_paths = normalized_copies(
            query_paths, os.path.join(cache_dir, "query"), [s for _, s, _, _ in labelled])
        if args.normalize == "both":
            corpus_src = [os.path.join(args.images, f"{u}.jpg") for u in unicodes]
            corpus = extractor.embed_paths(
                normalized_copies(corpus_src, os.path.join(cache_dir, "corpus"), unicodes),
                batch_size=args.batch_size)
        print(f"Đã chuẩn hoá: {args.normalize} "
              "(chênh lệch nằm trong khoảng nhiễu ở n=56 — xem BENCHMARK.md mục 10.4)")
    queries = extractor.embed_paths(query_paths, batch_size=args.batch_size)

    # --- Phần 1: tìm trên toàn corpus ---------------------------------------------------------
    scores = queries @ corpus.T
    order = np.argsort(-scores, axis=1)[:, :args.top_k]

    ranks, rows, pseudo = [], [], {}
    for i, (path, stem, word, gt) in enumerate(labelled):
        retrieved = unicodes[order[i]]
        found = [r for r, u in enumerate(retrieved) if u in gt]
        rank = found[0] + 1 if found else None
        ranks.append(rank)
        if rank == 1:
            pseudo[stem] = retrieved[0]
        rows.append({
            "file": stem, "word": word, "n_gt": len(gt),
            "rank": rank if rank else ">%d" % args.top_k,
            "top1": chars.loc[chars["UNICODE"].astype(str) == retrieved[0], "CHAR"].iloc[0]
                    if (chars["UNICODE"].astype(str) == retrieved[0]).any() else retrieved[0],
            "top1_score": round(float(scores[i, order[i, 0]]), 4),
        })

    if args.part in ("1", "both"):
        print(f"\n=== Phần 1: tìm trên toàn corpus ({args.backend}/{args.dtype}) ===")
        print(pd.DataFrame(rows).to_string(index=False))
        n = len(ranks)
        print(f"\nn = {n} ảnh query thật")
        for k in (1, 5, 10, 20):
            if k > args.top_k:
                continue
            hit = sum(1 for r in ranks if r and r <= k)
            print(f"  hit@{k:<3d} = {hit}/{n} = {hit/n:.4f}")
        mrr = np.mean([1 / r if r else 0.0 for r in ranks])
        base = np.mean([len(gt) / len(unicodes) for _, _, _, gt in labelled])
        print(f"  MRR      = {mrr:.4f}")
        print(f"  baseline ngẫu nhiên hit@1 = {base:.6f}  ->  hơn {(sum(1 for r in ranks if r == 1)/n)/base:.0f}×")

    # --- Phần 2: chỉ trong nhóm cùng âm, embedding vs histogram --------------------------------
    if args.part in ("2", "both"):
        truth = dict(pseudo)
        source = f"pseudo-label ({len(pseudo)} ảnh Phần 1 trả top-1 đúng nhóm âm)"
        if args.labels:
            table = pd.read_csv(args.labels)
            cand_of = {stem: sorted(gt) for _, stem, _, gt in labelled}
            truth = {}
            rank_of = {lv: i for i, lv in enumerate(CONFIDENCE_LEVELS)}  # high=0 < med=1 < low=2
            floor = rank_of.get(args.min_confidence, len(CONFIDENCE_LEVELS))
            dropped = []
            for row in table.itertuples():
                cand = cand_of.get(row.file)
                if not cand:
                    continue
                conf = str(getattr(row, "confidence", "") or "").strip().lower()
                if conf and conf not in rank_of:
                    print(f"  ! confidence lạ ở dòng {row.file!r}: {conf!r} "
                          f"(hợp lệ: {', '.join(CONFIDENCE_LEVELS)}) — bỏ qua bộ lọc cho dòng này")
                if args.min_confidence and conf in rank_of and rank_of[conf] > floor:
                    dropped.append(row.file)
                    continue
                if isinstance(getattr(row, "correct_unicode", None), str) and row.correct_unicode.strip():
                    truth[row.file] = row.correct_unicode.strip()
                elif not pd.isna(getattr(row, "correct_index", np.nan)):
                    pos = int(row.correct_index) - 1
                    if 0 <= pos < len(cand):
                        truth[row.file] = cand[pos]
            source = f"nhãn người gán từ {args.labels} ({len(truth)} ảnh)"
            if args.min_confidence:
                source += (f", lọc confidence >= {args.min_confidence} "
                           f"(bỏ {len(dropped)} ảnh: {', '.join(sorted(dropped))})")

        print(f"\n=== Phần 2: trong nhóm cùng âm Quốc Ngữ, embedding vs histogram ===")
        print(f"Nguồn nhãn: {source}")
        if not truth:
            raise SystemExit("Không có nhãn nào — chạy --sheets rồi điền, hoặc bỏ --labels.")

        emb_ok = hist_ok = 0
        emb_ranks, hist_ranks, agree = [], [], []
        for i, (path, stem, word, gt) in enumerate(labelled):
            cand = sorted(gt)
            cand_emb = corpus[[row_of[u] for u in cand]]
            emb_rank = [cand[j] for j in np.argsort(-(cand_emb @ queries[i]))]

            query_hist = extract_histogram(path)
            dist = [np.linalg.norm(query_hist - extract_histogram(
                os.path.join(args.images, f"{u}.jpg"))) for u in cand]
            hist_rank = [cand[j] for j in np.argsort(dist)]

            # Chỉ tính trên các ảnh thật sự được chấm: khi lọc --min-confidence, ảnh bị loại
            # không nằm trong mẫu số của mọi con số khác, nên cũng không được nằm ở đây.
            if stem in truth:
                agree.append(emb_rank[0] == hist_rank[0])
                gold = truth[stem]
                emb_ok += emb_rank[0] == gold
                hist_ok += hist_rank[0] == gold
                emb_ranks.append(emb_rank.index(gold) + 1 if gold in emb_rank else None)
                hist_ranks.append(hist_rank.index(gold) + 1 if gold in hist_rank else None)

        n_t = len(emb_ranks)
        note = "  (100% theo định nghĩa nếu dùng pseudo-label)" if not args.labels else ""
        print(f"  embedding top-1 đúng: {emb_ok}/{n_t} = {emb_ok/n_t:.4f}{note}")
        print(f"  histogram top-1 đúng: {hist_ok}/{n_t} = {hist_ok/n_t:.4f}")
        print(f"  MRR embedding = {np.mean([1/r for r in emb_ranks if r]):.4f}  |  "
              f"histogram = {np.mean([1/r for r in hist_ranks if r]):.4f}")
        base = np.mean([1 / len(gt) for _, stem, _, gt in labelled if stem in truth])
        print(f"  baseline chọn ngẫu nhiên trong nhóm = {base:.4f}")
        print(f"  hai phương pháp chọn cùng top-1: {sum(agree)}/{len(agree)} = {np.mean(agree):.4f}")


if __name__ == "__main__":
    main()

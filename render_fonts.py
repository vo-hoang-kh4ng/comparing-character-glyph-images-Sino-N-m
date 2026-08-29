"""Render mỗi ký tự trong corpus qua NHIỀU font để tạo **cặp dương** cho fine-tune.

Vì sao cần
----------
Corpus chỉ có **đúng 1 ảnh cho 1 ký tự** (26.044 ảnh / 26.044 ký tự). Metric learning cần cặp
dương — hai ảnh mà mô hình phải học cách kéo lại gần nhau — nên hiện tại **không tồn tại cặp nào**
để dạy mô hình bỏ qua khác biệt kiểu chữ. Đó đúng là chỗ hỏng đã đo được ở BENCHMARK.md mục 10:
query viết tay/hành thư vs corpus in khải thư, hit@1 chỉ 0,0714.

Mỗi ký tự đã có mã Unicode nên render lại qua nhiều font là cách rẻ nhất để có cặp dương: cùng một
lớp (mã Unicode), khác kiểu chữ. Script này sinh ra tập đó.

Font
----
Không dùng font hệ thống (máy này không có font CJK nào). Font nằm trong `./fonts/`, tải rời — xem
`FONT_GROUPS` và `download_fonts.sh`. Mỗi nhóm là một *kiểu chữ*, có thể gồm nhiều file theo thứ tự
dự phòng (HanaMinA phủ BMP, HanaMinB phủ Ext-B; ghép lại mới đủ 99,9%).

Độ phủ đã đo trên 26.044 ký tự có ảnh (kiểm bằng bảng `cmap`, không đoán):

    hanamin 99,9% | babelstone 88,3% | iming 83,7% | lxgw-kai 71,3%
    noto-serif 59,9% | noto-sans 59,9% | zhuque-fangsong 42,3%
    -> hợp nhất 26.033/26.044 = 100,0%; chỉ 11 ký tự không font nào render được.

**Ký tự không có trong `cmap` bị bỏ qua, không render.** Nếu render bừa, font trả về ô tofu
(.notdef) — một ô vuông rỗng giống hệt nhau ở mọi ký tự, đưa vào huấn luyện sẽ đầu độc cặp dương.

Còn thiếu gì
------------
Render font **chưa đủ**: ảnh render tự truy hồi đúng chữ ở hit@1 = 0,7018, còn scan thật chỉ 0,0714
(BENCHMARK.md mục 13.3). Phần lớn khoảng cách không nằm ở font mà ở nhiễu scan. Dùng `--augment N`
để thêm bản suy giảm qua `scan_augment.degrade` — đã hiệu chỉnh khớp số đo của `test_images/`, kéo
hit@1 xuống 0,1986 (mục 13.5).

Hình học
--------
Ảnh render đi qua cùng hàm chuẩn hoá với `evaluate_test_images.py`: cắt sát nét mực, đệm vào khung
vuông, chữ chiếm `--box` bề rộng. Nhờ vậy ảnh render và ảnh corpus cùng khung, mô hình không học
nhầm "lề rộng hẹp" thành đặc trưng phân biệt.

Ví dụ
-----
    python render_fonts.py --limit 200            # thử nhanh
    python render_fonts.py                        # toàn bộ, 7 font
    python render_fonts.py --augment 2            # thêm 2 bản "suy giảm giống scan" mỗi ảnh
"""

import argparse
import csv
import os
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from fontTools.ttLib import TTFont

from scan_augment import degrade

# tag -> (danh sách file theo thứ tự dự phòng, mô tả kiểu chữ)
FONT_GROUPS = {
    "hanamin": (["fonts/HanaMinA.otf", "fonts/HanaMinB.otf"], "花園明朝 minh thể, phủ Ext-A..Ext-G"),
    "babelstone": (["fonts/BabelStoneHan.ttf"], "BabelStone Han, tống thể, phủ rộng"),
    "iming": (["fonts/I.Ming-8.10.ttf"], "一點明體 minh thể truyền thống"),
    "lxgw-kai": (["fonts/LXGWWenKaiTC-Regular.ttf"], "霞鶩文楷 KHẢI THƯ — kiểu bút lông, quan trọng nhất"),
    "noto-serif": (["fonts/15_NotoSerifTC/SubsetOTF/TC/NotoSerifTC-Regular.otf"], "Noto Serif TC, tống thể hiện đại"),
    "noto-sans": (["fonts/19_NotoSansTC/NotoSansTC-Regular.otf"], "Noto Sans TC, hắc thể"),
    "zhuque-fangsong": (["fonts/ZhuqueFangsong-v0.212/ZhuqueFangsong-Regular.ttf"], "朱雀仿宋"),
}

RENDER_PX = 256          # cỡ chữ khi vẽ, trước khi cắt sát nét
CANVAS_PX = RENDER_PX * 2


@lru_cache(maxsize=None)
def font_cmap(path):
    """Tập codepoint font thật sự có glyph. Dùng để bỏ qua ký tự sẽ ra ô tofu."""
    face = TTFont(path, fontNumber=0, lazy=True)
    cmap = frozenset(face.getBestCmap())
    face.close()
    return cmap


@lru_cache(maxsize=None)
def pil_font(path, px):
    return ImageFont.truetype(path, px)


def pick_file(tag, codepoint):
    """File đầu tiên trong nhóm có glyph cho codepoint này, hoặc None."""
    for path in FONT_GROUPS[tag][0]:
        if codepoint in font_cmap(path):
            return path
    return None


def normalize_glyph(array, box=0.80, size=224):
    """Cắt sát nét mực, đệm vào khung vuông, chữ chiếm `box` bề rộng khung.

    Cùng quy ước với `evaluate_test_images.normalize_glyph` để ảnh render và ảnh corpus cùng khung.
    """
    ink = array < int(array.max()) - 40
    if not ink.any():
        return None
    ys, xs = np.where(ink)
    crop = array[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    height, width = crop.shape
    scale = int(size * box) / max(height, width)
    new_h, new_w = max(1, round(height * scale)), max(1, round(width * scale))
    canvas = Image.new("L", (size, size), 255)
    canvas.paste(Image.fromarray(crop).resize((new_w, new_h), Image.LANCZOS),
                 ((size - new_w) // 2, (size - new_h) // 2))
    return canvas


def render_char(char, path, box, size):
    """Vẽ một ký tự bằng một file font -> ảnh xám đã chuẩn hoá, hoặc None nếu ra ảnh trắng."""
    canvas = Image.new("L", (CANVAS_PX, CANVAS_PX), 255)
    ImageDraw.Draw(canvas).text((CANVAS_PX // 2, CANVAS_PX // 2), char,
                                font=pil_font(path, RENDER_PX), fill=0, anchor="mm")
    return normalize_glyph(np.array(canvas, dtype=np.uint8), box, size)


def work(job):
    """Render 1 ký tự qua mọi font -> danh sách dòng manifest. Chạy trong worker."""
    code, char, tags, out_dir, box, size, n_aug, strength, seed = job
    codepoint = int(code, 16)
    rng = np.random.default_rng(seed)
    rows = []
    for tag in tags:
        path = pick_file(tag, codepoint)
        if path is None:
            continue
        image = render_char(char, path, box, size)
        if image is None:                       # font có glyph nhưng vẽ ra trắng trơn
            continue
        out = os.path.join(out_dir, tag, f"{code}.png")
        image.save(out)
        rows.append((code, char, tag, 0, os.path.relpath(out)))
        for i in range(n_aug):
            aug_out = os.path.join(out_dir, tag, f"{code}_aug{i + 1}.png")
            degrade(image, rng, strength).save(aug_out)
            rows.append((code, char, tag, i + 1, os.path.relpath(aug_out)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", default="final_characteristics-v2.xlsx")
    parser.add_argument("--images", default="images")
    parser.add_argument("--out-dir", default="output/rendered")
    parser.add_argument("--fonts", nargs="*", default=list(FONT_GROUPS),
                        help=f"nhóm font cần render (mặc định tất cả): {', '.join(FONT_GROUPS)}")
    parser.add_argument("--box", type=float, default=0.80, help="chữ chiếm bao nhiêu bề rộng khung")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--augment", type=int, default=0,
                        help="số bản suy giảm giống scan thêm cho mỗi (ký tự, font)")
    parser.add_argument("--strength", type=float, default=1.0,
                        help="mức suy giảm 0..1 (xem scan_augment.py --calibrate)")
    parser.add_argument("--limit", type=int, help="chỉ render N ký tự đầu, để thử nhanh")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    unknown = [t for t in args.fonts if t not in FONT_GROUPS]
    if unknown:
        parser.error(f"nhóm font không có: {unknown}")
    missing = [p for t in args.fonts for p in FONT_GROUPS[t][0] if not os.path.exists(p)]
    if missing:
        parser.error(f"thiếu file font: {missing}\nChạy `bash download_fonts.sh` trước.")

    table = pd.read_excel(args.table)
    table["U"] = (table["UNICODE"].astype(str).str.upper()
                  .str.replace("U+", "", regex=False).str.strip())
    have = {os.path.splitext(f)[0].upper() for f in os.listdir(args.images)}
    table = table[table["U"].isin(have)][["U", "CHAR"]].dropna()
    if args.limit:
        table = table.head(args.limit)
    print(f"{len(table)} ký tự có ảnh | {len(args.fonts)} nhóm font | {args.augment} biến thể/ảnh")

    for tag in args.fonts:
        os.makedirs(os.path.join(args.out_dir, tag), exist_ok=True)

    jobs = [(row.U, row.CHAR, args.fonts, args.out_dir, args.box, args.size,
             args.augment, args.strength, abs(hash(row.U)) % (2 ** 32))
            for row in table.itertuples()]

    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    per_font = dict.fromkeys(args.fonts, 0)
    per_char = []
    with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["unicode", "char", "view", "aug", "path"])
        # ảnh corpus gốc cũng là một "view" của lớp — ghi vào manifest để dùng chung nhãn
        for row in table.itertuples():
            writer.writerow([row.U, row.CHAR, "corpus", 0,
                             os.path.join(args.images, f"{row.U}.jpg")])
        with Pool(args.workers) as pool:
            for done, rows in enumerate(pool.imap_unordered(work, jobs, chunksize=64), 1):
                writer.writerows(rows)
                for _, _, tag, aug, _ in rows:
                    if aug == 0:
                        per_font[tag] += 1
                per_char.append(sum(1 for r in rows if r[3] == 0))
                if done % 2000 == 0:
                    print(f"  {done}/{len(jobs)}", flush=True)

    print(f"\nmanifest: {manifest_path}")
    print(f"{'font':18s} {'render':>7s} {'%':>7s}  kiểu chữ")
    for tag in args.fonts:
        print(f"{tag:18s} {per_font[tag]:7d} {per_font[tag] / len(table):7.1%}  {FONT_GROUPS[tag][1]}")

    counts = np.array(per_char)
    print(f"\nký tự có ≥1 render (=> ≥2 view, đủ tạo cặp dương): "
          f"{(counts >= 1).sum()}/{len(counts)} = {(counts >= 1).mean():.1%}")
    print(f"ký tự không font nào render được: {(counts == 0).sum()}")
    print(f"trung bình {counts.mean():.2f} kiểu chữ/ký tự | tổng ảnh render: "
          f"{int(counts.sum() * (1 + args.augment))}")


if __name__ == "__main__":
    main()

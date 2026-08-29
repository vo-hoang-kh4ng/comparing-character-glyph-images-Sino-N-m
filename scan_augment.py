"""Làm ảnh render "xuống cấp" cho giống scan thật — hiệu chỉnh theo số đo, không theo cảm tính.

Vì sao cần
----------
`render_fonts.py` sinh được cặp dương phủ 100% corpus, nhưng BENCHMARK.md mục 13.3 đo ra tập đó
**quá dễ**: ảnh render tự truy hồi đúng chữ ở hit@1 = 0,7018, còn scan thật chỉ 0,0714. Chênh gần
9 lần. Nghĩa là phần lớn khoảng cách tới thực tế **không nằm ở font**.

Đo trực tiếp 59 ảnh trong `test_images/` so với ảnh render cho thấy khoảng cách nằm ở đâu:

    chỉ số      scan thật   render    tỉ lệ    ý nghĩa
    edge          15,6      102,3     0,15x    scan MỜ hơn nhiều — khoảng cách lớn nhất
    contrast      91,6      232,8     0,39x    mực xám nhạt, không đen tuyền
    thick        0,051      0,028     1,82x    nét scan dày gần gấp đôi
    fill         0,349      0,206     1,69x    (hệ quả của nét dày)
    noise         10,1        2,7     3,77x    hạt nhiễu giấy/cảm biến
    bg            254,0      255,0    ~1x      nền giấy hơi ngà

`degrade()` mô phỏng đúng sáu trục đó, theo thứ tự vật lý của một trang scan: tay viết → mực loang
trên giấy → mất phân giải khi chụp → nhoè quang học → giấy ngà + tương phản kém → nhiễu cảm biến.

**Không đụng tới cấu trúc bộ thủ.** Mọi phép biến đổi ở đây đều là nhiễu ảnh; ký tự vẫn là ký tự đó,
nên nhãn (mã Unicode) vẫn đúng.

Ví dụ
-----
    python scan_augment.py --calibrate     # so số đo scan / render / render đã suy giảm
    python scan_augment.py --preview       # xuất bảng ảnh để nhìn bằng mắt
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from scipy import ndimage


def _elastic(array, rng, alpha, sigma=6.0):
    """Méo đàn hồi — mô phỏng nét cong do tay người viết và giấy vênh."""
    shape = array.shape
    dx = ndimage.gaussian_filter(rng.uniform(-1, 1, shape), sigma, mode="constant") * alpha
    dy = ndimage.gaussian_filter(rng.uniform(-1, 1, shape), sigma, mode="constant") * alpha
    grid_y, grid_x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    return ndimage.map_coordinates(array, [grid_y + dy, grid_x + dx], order=1,
                                   mode="constant", cval=255.0)


def degrade(image, rng, strength=1.0):
    """Ảnh render sạch -> ảnh giống scan. `strength` 0..1 nội suy mức suy giảm.

    Các khoảng tham số dưới đây **đã hiệu chỉnh** để số đo đầu ra khớp `test_images/` — xem
    `--calibrate`. Sửa số thì phải chạy lại calibrate, đừng chỉnh mò.
    """
    size = image.size[0]
    array = np.array(image.convert("L"), dtype=np.float32)

    # 1. tay viết + giấy vênh
    array = _elastic(array, rng, alpha=rng.uniform(3.0, 11.0) * strength)

    # 2. mực loang trên giấy: nét dày lên ~1,8x
    image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
    for _ in range(rng.integers(1, 3)):
        image = image.filter(ImageFilter.MinFilter(3))

    # 3. mất phân giải khi chụp — cũng là thứ làm hai nét sát nhau DÍNH vào nhau
    small = max(24, int(size * rng.uniform(0.16, 0.30) / max(strength, 1e-3) ** 0.5))
    small = min(small, size)
    image = image.resize((small, small), Image.BILINEAR).resize((size, size), Image.BICUBIC)

    # 4. nhoè quang học
    image = image.filter(ImageFilter.GaussianBlur(rng.uniform(1.0, 2.4) * strength))
    array = np.array(image, dtype=np.float32)

    # 5. giấy ngà + tương phản kém: mực thành xám, không đen tuyền
    alpha = 1.0 - (1.0 - rng.uniform(0.30, 0.55)) * strength
    paper = rng.uniform(246, 254)
    array = paper - (255.0 - array) * alpha
    # vệt sáng tối không đều trên trang giấy (tần số thấp)
    field = ndimage.gaussian_filter(rng.normal(0, 1, array.shape), size / 6.0)
    field *= rng.uniform(4.0, 12.0) * strength / max(field.std(), 1e-6)
    array += field

    # 6. nhiễu cảm biến
    array += rng.normal(0, rng.uniform(5.0, 12.0) * strength, array.shape)

    # 7. scan không vuông (test_images: 88x96, 96x128...) rồi ép lại về khung vuông
    if rng.random() < 0.7 * strength:
        ratio = rng.uniform(0.80, 1.25)
        temp = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
        temp = temp.resize((max(8, int(size * ratio)), size), Image.BILINEAR)
        array = np.array(temp.resize((size, size), Image.BILINEAR), dtype=np.float32)

    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


# --- đo đạc ---------------------------------------------------------------------------------

def measure(source):
    """Sáu chỉ số dùng để so tập ảnh này với scan thật. Nhận đường dẫn hoặc ảnh PIL."""
    array = np.array((Image.open(source) if isinstance(source, str) else source).convert("L"),
                     dtype=np.float32)
    background = np.percentile(array, 95)
    ink = array < background - 40
    if not ink.any():
        return None
    distance = ndimage.distance_transform_edt(ink)
    gy, gx = np.gradient(array)
    gradient = np.sqrt(gx ** 2 + gy ** 2)
    strong = gradient > gradient.max() * 0.2
    return dict(bg=background,
                fill=ink.mean(),
                thick=2 * distance[ink].mean() / max(array.shape),
                edge=gradient[strong].mean() if strong.any() else 0.0,
                noise=array[~ink].std(),
                contrast=background - array[ink].mean())


COLUMNS = ["bg", "fill", "thick", "edge", "noise", "contrast"]


def summarize(rows):
    return pd.DataFrame([r for r in rows if r]).mean()


def calibrate(args):
    rng = np.random.default_rng(0)
    scans = summarize(measure(p) for p in sorted(glob.glob(f"{args.test_images}/*")))

    manifest = pd.read_csv(args.manifest, dtype=str)
    manifest = manifest[(manifest["aug"] == "0") & (manifest["view"] != "corpus")]
    paths = list(manifest.sample(args.n, random_state=0)["path"])
    clean = summarize(measure(p) for p in paths)
    dirty = summarize(measure(degrade(Image.open(p), rng, args.strength)) for p in paths)

    print(f"n scan = {len(glob.glob(f'{args.test_images}/*'))} | n render = {len(paths)} "
          f"| strength = {args.strength}\n")
    print(f"{'chỉ số':10s} {'scan thật':>11s} {'render':>11s} {'đã suy giảm':>12s} "
          f"{'lệch trước':>11s} {'lệch sau':>10s}")
    for column in COLUMNS:
        before = clean[column] / scans[column]
        after = dirty[column] / scans[column]
        print(f"{column:10s} {scans[column]:11.3f} {clean[column]:11.3f} {dirty[column]:12.3f} "
              f"{before:10.2f}x {after:9.2f}x")
    error_before = np.mean([abs(np.log(clean[c] / scans[c])) for c in COLUMNS])
    error_after = np.mean([abs(np.log(dirty[c] / scans[c])) for c in COLUMNS])
    print(f"\nsai lệch log trung bình: trước {error_before:.3f} -> sau {error_after:.3f} "
          f"({'tốt hơn' if error_after < error_before else 'TỆ HƠN'} "
          f"{error_before / max(error_after, 1e-9):.1f}x)")


def preview(args):
    rng = np.random.default_rng(7)
    manifest = pd.read_csv(args.manifest, dtype=str)
    manifest = manifest[(manifest["aug"] == "0") & (manifest["view"] != "corpus")]
    rows = manifest.sample(6, random_state=11)
    scans = sorted(glob.glob(f"{args.test_images}/*"))[:4]

    cell, columns = 112, 6
    sheet = Image.new("RGB", (cell * columns, cell * (len(rows) + 1)), "white")
    for index, path in enumerate(scans):
        sheet.paste(Image.open(path).convert("RGB").resize((cell - 8, cell - 8)),
                    (index * cell + 4, 4))
    for row, item in enumerate(rows.itertuples(), start=1):
        image = Image.open(item.path).convert("L")
        sheet.paste(image.convert("RGB").resize((cell - 8, cell - 8)), (4, row * cell + 4))
        for column in range(1, columns):
            sheet.paste(degrade(image, rng, args.strength).convert("RGB").resize((cell - 8, cell - 8)),
                        (column * cell + 4, row * cell + 4))
    sheet.save(args.out)
    print(f"hàng 1 = scan thật; cột 1 = render gốc; còn lại = đã suy giảm -> {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default="output/rendered/manifest.csv")
    parser.add_argument("--test-images", default="test_images")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--out", default="output/rendered/_scan_preview.png")
    args = parser.parse_args()
    if args.preview:
        preview(args)
    if args.calibrate or not args.preview:
        calibrate(args)


if __name__ == "__main__":
    main()

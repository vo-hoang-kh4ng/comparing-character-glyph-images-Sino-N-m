"""Đóng gói bài nộp: mã nguồn + dữ liệu + mô hình + báo cáo, kèm MANIFEST có checksum.

Yêu cầu của thầy gồm bốn phần: (1) dataset, (2) mã nguồn do nhóm viết, (3) mô hình đã huấn luyện,
(4) liên kết tới mô hình *bên ngoài* thay vì nộp kèm chúng, và (5) báo cáo .doc/.pdf tái lập được.
Script này dựng đúng cấu trúc đó và tự kiểm những chỗ dễ sai.

Ba quyết định đã cân nhắc, ghi lại để người chấm không phải đoán
----------------------------------------------------------------
1. **Mô hình pretrain (Chinese-CLIP, DINOv2, ResNet18) KHÔNG kèm theo.** Chúng là mô hình bên ngoài,
   tổng cộng vài GB, và tải tự động từ Hugging Face khi chạy lần đầu. Chỉ ghi định danh chính xác
   trong ``model/MODEL.md``.
2. **Corpus 26.044 ảnh và hai file .xlsx không kèm mặc định** (103 MB) --- chúng là dữ liệu cho
   trước của môn học, không phải do nhóm tạo. Dùng ``--with-corpus`` nếu người chấm muốn gói tự đủ.
3. **131.604 ảnh render KHÔNG kèm theo** (1,1 GB). Chúng sinh lại được bằng đúng một lệnh và hoàn
   toàn tất định, nên nộp kèm chỉ làm gói phình lên. Lệnh đó nằm trong ``data/DATA.md``.

Ngược lại, ``label_template.csv`` **luôn** được kèm: đó là nhãn người gán tay, chạy lại code không
sinh lại được, và mọi con số Phần 2 phụ thuộc vào nó.

Dùng
----
    .venv/bin/python package_submission.py                 # gói gọn, mô hình ghi bằng liên kết
    .venv/bin/python package_submission.py --with-model    # nhúng luôn best.pt (329 MB)
    .venv/bin/python package_submission.py --with-corpus --zip
"""

import argparse
import hashlib
import os
import shutil
import zipfile

#: Mã nguồn do nhóm viết. Thứ tự ở đây là thứ tự đọc hợp lý, không phải thứ tự bảng chữ cái.
SOURCE_FILES = [
    ("feature_extractors.py", "Sáu backend trích đặc trưng sau một giao diện chung"),
    ("search_all_chars_in_corpus.py", "Phần 1: nhúng toàn corpus, cache, tìm top-k bằng Faiss"),
    ("search_use_QuocNgu_mapping.py", "Phần 2: xếp hạng trong nhóm cùng âm Quốc Ngữ"),
    ("benchmark_extractors.py", "So sánh backend theo ba chỉ số proxy"),
    ("analyze_metric_ceiling.py", "Trần lý thuyết và trần oracle của radical@k"),
    ("benchmark_search.py", "So tìm kiếm chính xác với chỉ mục xấp xỉ"),
    ("evaluate_test_images.py", "Đánh giá trên scan thật; sinh phiếu gán nhãn"),
    ("render_fonts.py", "Sinh cặp dương bằng render đa font"),
    ("scan_augment.py", "Mô hình suy giảm ảnh, hiệu chỉnh theo số đo"),
    ("finetune_glyph.py", "Fine-tune SupCon; nhánh ArcFace để tái lập lần sụp đổ"),
    ("rerank.py", "Tầng xếp hạng lại bằng siêu dữ liệu ngôn ngữ học"),
    ("evaluate_rerank.py", "Ba suite chấm tầng xếp hạng lại, tách vòng tròn khỏi trung thực"),
    ("test_rerank.py", "Unit test cho rerank.py"),
    ("test_evaluate_rerank.py", "Unit test cho evaluate_rerank.py"),
    ("conftest.py", "Khai báo marker pytest"),
    ("build_report.py", "Sinh báo cáo .docx/.pdf"),
    ("package_submission.py", "Chính script này"),
    ("download_fonts.sh", "Tải 7 font Hán-Nôm tự do"),
    ("requirements.txt", "Phụ thuộc Python"),
]

#: CLAUDE.md cố ý KHÔNG nộp: đó là ghi chú làm việc nội bộ, viết bằng tiếng Anh cho công cụ, không
#: phải tài liệu dành cho người chấm. BENCHMARK.md thì có — nó là bản dài của chính báo cáo.
DOCS = ["README.md", "BENCHMARK.md"]
CKPT = "output/finetune/best.pt"
CKPT_FP16 = "output/finetune/best_fp16.pt"
LABELS = "output/label_sheets/label_template.csv"
TRAIN_LOG = "output/finetune/train.log"

#: Điền vào sau khi tải checkpoint lên. Để trống thì MODEL.md sẽ ghi rõ là còn thiếu, chứ không
#: lặng lẽ sinh ra một liên kết chết.
MODEL_URL = "https://huggingface.co/vohoangkh4ng/chinese-clip-ft-sinonom"


def sha256(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def human(num_bytes):
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024 or unit == "GB":
            return f"{num_bytes:.0f} {unit}" if unit == "B" else f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024


def copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)


DATA_MD = """# Dữ liệu

## Kèm trong gói này

| Đường dẫn | Mô tả |
|---|---|
| `test_images/` | 59 ảnh scan thật, chữ viết tay. **Đây là tập đánh giá.** Tên file gõ Telex (`cofn.jpg` = "còn") nên nhãn giải mã ngược ra được từ chính tên file. |
| `label_template.csv` | Nhãn Phần 2 **do người gán tay**, kèm cột `confidence` (53 high / 4 med / 2 low). Chạy lại code KHÔNG sinh lại được file này, và mọi con số Phần 2 phụ thuộc vào nó. |
| `train.log` | Nhật ký đầy đủ của lần fine-tune, gồm cả lần ArcFace bị sụp đổ (mục 6.3 của báo cáo). |

## Không kèm, và cách lấy

**Corpus 26.044 ảnh glyph + hai bảng thuộc tính** (~106 MB) --- dữ liệu cho trước của môn học, tải
theo liên kết Drive ghi trong `README.md`. Giải nén `images.zip` thành `./images`, đặt
`final_characteristics-v2.xlsx` và `QuocNgu_SinoNom_Dic.xlsx` vào thư mục gốc.

**131.604 ảnh render đa font** (~1,1 GB) --- không nộp kèm vì sinh lại được và hoàn toàn tất định:

```bash
bash download_fonts.sh          # tải 7 font Hán-Nôm tự do vào ./fonts
python render_fonts.py          # ~20 phút trên 16 nhân CPU -> output/rendered/
```

Không cần `--augment`: `finetune_glyph.py` suy giảm ảnh **trực tuyến** lúc nạp dữ liệu, nên tập
huấn luyện thật sự không bao giờ nằm trên đĩa.

## Chạy test

Từ thư mục `src/`:

```bash
python -m pytest -q
```

Đủ dữ liệu thì **55 test pass**. Nếu chưa đặt `final_characteristics-v2.xlsx` vào thư mục làm việc
thì sẽ thấy **3 fail + 4 skip** — đó là các test cần bảng thuộc tính thật, không phải code hỏng.
Đặt file đó vào rồi chạy lại là đủ 55.

## Ghi chú về rò rỉ dữ liệu

`test_images/` **không** đi vào huấn luyện. Điều này được bảo đảm bằng một `assert` chạy trước mỗi
lần huấn luyện (`_assert_test_images_unseen` trong `finetune_glyph.py`), không phải bằng quy ước.
Ngoài ra, 2.604 mã Unicode bị gỡ toàn bộ khỏi tập huấn luyện để đo khả năng tổng quát hoá.
"""


def model_md(url, ckpt_included, ckpt_info):
    link = (f"Tải checkpoint tại: <{url}>" if url else
            "> **CHƯA ĐIỀN LIÊN KẾT.** Tải `best.pt` lên Hugging Face Hub hoặc Google Drive, rồi\n"
            "> đặt URL vào biến `MODEL_URL` trong `package_submission.py` và đóng gói lại.")
    where = ("Checkpoint nằm ngay trong thư mục này (`best.pt`)." if ckpt_included else
             "Checkpoint **không** kèm trong gói vì dung lượng lớn.\n\n" + link)
    return f"""# Mô hình

## Mô hình do nhóm huấn luyện

{where}

{ckpt_info}

**Vì sao là bản fp16.** Đường ống suy luận gọi `model.to(fp16)` ngay sau khi nạp trọng số, nên lưu
sẵn fp16 chỉ là làm trước đúng phép ép kiểu vẫn luôn xảy ra --- không phải xấp xỉ. Đã kiểm bằng cách
dựng lại toàn bộ cache 26.044 vector từ cả hai bản: **lệch tuyệt đối tối đa = 0,0**, cosine trung
bình = 1,0. Bản fp32 chỉ cần nếu định huấn luyện tiếp.

Kiến trúc: tháp ảnh Chinese-CLIP ViT-B/16, giữ nguyên `visual_projection` (512 chiều). Vì hình dạng
trọng số không đổi so với bản pretrain, checkpoint nạp thẳng vào đường ống có sẵn:

```bash
python search_all_chars_in_corpus.py --backend chinese-clip-ft --dtype fp16 --device cuda
python evaluate_test_images.py --backend chinese-clip-ft --dtype fp16 --device cuda \\
       --labels data/label_template.csv
```

Đường dẫn checkpoint mặc định là `output/finetune/best.pt`; đổi bằng biến môi trường
`GLYPH_FT_CKPT`. Nếu thiếu file, chương trình **báo lỗi và dừng** chứ không lặng lẽ quay về trọng số
zero-shot --- nếu quay về thì bảng kết quả sẽ ghi số zero-shot dưới cái tên đã fine-tune.

## Mô hình bên ngoài (KHÔNG nộp kèm)

Tải tự động từ Hugging Face khi chạy lần đầu:

| Định danh | Dùng ở đâu |
|---|---|
| `OFA-Sys/chinese-clip-vit-base-patch16` | backend `chinese-clip`, và là điểm khởi đầu của bản fine-tune |
| `OFA-Sys/chinese-clip-vit-large-patch14` | backend `chinese-clip-large` --- mốc so chính |
| `OFA-Sys/chinese-clip-vit-huge-patch14` | backend `chinese-clip-huge` |
| `facebook/dinov2-base` | backend `dinov2` |
| torchvision `ResNet18_Weights.IMAGENET1K_V1` | backend `resnet18` --- baseline của bản gốc |

Font Hán-Nôm cũng là tài nguyên ngoài, tải bằng `download_fonts.sh`; tất cả đều là font tự do và
nguồn ghi ngay trong script đó.
"""


def export_fp16(src, dst):
    """Ghi bản fp16 của checkpoint. Nửa dung lượng, embedding giống hệt.

    Không phải phép xấp xỉ: ``ChineseClipExtractor`` gọi ``model.to(fp16)`` ngay sau khi nạp trọng
    số, nên lưu sẵn fp16 chỉ là làm trước đúng phép ép kiểu mà đường ống vẫn làm. Đã kiểm lại bằng
    cách dựng lại toàn bộ cache 26.044 vector từ cả hai bản: lệch tuyệt đối tối đa = 0,0.
    """
    import torch
    ckpt = torch.load(src, map_location="cpu")
    ckpt["model"] = {k: (v.half() if v.is_floating_point() else v)
                     for k, v in ckpt["model"].items()}
    ckpt["dtype"] = "fp16"
    torch.save(ckpt, dst)
    return dst


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="submission")
    parser.add_argument("--with-model", action="store_true",
                        help="nhúng luôn checkpoint (bản fp16, ~165 MB) thay vì chỉ ghi liên kết")
    parser.add_argument("--fp32", action="store_true",
                        help="nhúng bản fp32 gốc (~329 MB). Chỉ cần nếu định huấn luyện tiếp; "
                             "để suy luận thì fp16 cho kết quả giống HỆT (xem MODEL.md)")
    parser.add_argument("--with-corpus", action="store_true",
                        help="nhúng luôn images/ và hai bảng .xlsx (~106 MB)")
    parser.add_argument("--zip", action="store_true")
    args = parser.parse_args()

    # Kiểm trước, dựng sau. Thiếu file mà vẫn dựng thì ra một gói trông đầy đủ nhưng hỏng.
    missing = [f for f, _ in SOURCE_FILES if not os.path.exists(f)]
    for path in (LABELS, "report/BaoCao.docx", "report/BaoCao.pdf"):
        if not os.path.exists(path):
            missing.append(path)
    if missing:
        raise SystemExit("Thiếu: " + ", ".join(missing) +
                         "\nChạy build_report.py trước nếu thiếu báo cáo.")

    out = args.out
    if os.path.exists(out):
        shutil.rmtree(out)
    manifest = []

    # --- mã nguồn ---
    for name, _ in SOURCE_FILES:
        copy(name, os.path.join(out, "src", name))
    for name in DOCS:
        if os.path.exists(name):
            copy(name, os.path.join(out, name))

    # --- dữ liệu ---
    copy("test_images", os.path.join(out, "data", "test_images"))
    copy(LABELS, os.path.join(out, "data", "label_template.csv"))
    if os.path.exists(TRAIN_LOG):
        copy(TRAIN_LOG, os.path.join(out, "data", "train.log"))
    with open(os.path.join(out, "data", "DATA.md"), "w", encoding="utf-8") as handle:
        handle.write(DATA_MD)
    if args.with_corpus:
        for name in ("images", "final_characteristics-v2.xlsx", "QuocNgu_SinoNom_Dic.xlsx"):
            if os.path.exists(name):
                copy(name, os.path.join(out, "data", name))

    # --- mô hình ---
    # Mặc định nộp bản fp16. Đường ống suy luận vốn cast model sang fp16 ngay khi nạp, nên bản fp16
    # cho embedding **giống hệt từng bit** với bản fp32 (đã kiểm: lệch tuyệt đối tối đa = 0,0 trên
    # cả 26.044 vector). Nộp fp32 chỉ có nghĩa nếu ai đó định huấn luyện tiếp.
    if os.path.exists(CKPT) and not os.path.exists(CKPT_FP16):
        export_fp16(CKPT, CKPT_FP16)
    source = CKPT if args.fp32 else CKPT_FP16
    info = "Chưa tìm thấy checkpoint — chạy `finetune_glyph.py` trước."
    if os.path.exists(source):
        size, digest = os.path.getsize(source), sha256(source)
        info = (f"- File: `{os.path.basename(source)}` "
                f"({'fp32, dùng khi muốn huấn luyện tiếp' if args.fp32 else 'fp16, dùng để suy luận'})\n"
                f"- Dung lượng: {human(size)}\n"
                f"- SHA-256: `{digest}`\n"
                f"- Kết quả trên 59 ảnh scan thật: `hit@1` = 0,5932, `hit@10` = 0,7966, "
                f"MRR = 0,6691\n")
        if args.with_model:
            name = os.path.basename(source)
            copy(source, os.path.join(out, "model", name))
            manifest.append((f"model/{name}", size, digest))
    os.makedirs(os.path.join(out, "model"), exist_ok=True)
    with open(os.path.join(out, "model", "MODEL.md"), "w", encoding="utf-8") as handle:
        handle.write(model_md(MODEL_URL, args.with_model and os.path.exists(CKPT), info))

    # --- báo cáo ---
    for name in ("BaoCao.docx", "BaoCao.pdf"):
        copy(os.path.join("report", name), os.path.join(out, "report", name))

    # --- manifest ---
    total = 0
    for root, _, files in os.walk(out):
        for name in sorted(files):
            path = os.path.join(root, name)
            total += os.path.getsize(path)
    lines = ["# Nội dung gói nộp", "",
             f"Tổng dung lượng: **{human(total)}**", "",
             "| Đường dẫn | Dung lượng | Nội dung |", "|---|---|---|"]
    for name, desc in SOURCE_FILES:
        path = os.path.join(out, "src", name)
        lines.append(f"| `src/{name}` | {human(os.path.getsize(path))} | {desc} |")
    lines += ["", "## Dữ liệu, mô hình, báo cáo", "",
              "| Đường dẫn | Nội dung |", "|---|---|",
              "| `data/test_images/` | 59 ảnh scan thật — tập đánh giá |",
              "| `data/label_template.csv` | Nhãn người gán (không sinh lại được) |",
              "| `data/train.log` | Nhật ký fine-tune, gồm cả lần huấn luyện thất bại |",
              "| `data/DATA.md` | Nguồn từng tập, và cách lấy phần không kèm |",
              "| `model/MODEL.md` | Checkpoint của nhóm + định danh mô hình bên ngoài |",
              "| `report/BaoCao.pdf` `.docx` | Báo cáo |",
              "| `BENCHMARK.md` | Bản dài: mọi bảng, mọi lần chạy, mọi kết quả âm |", ""]
    if manifest:
        lines += ["## Checksum", "", "| File | Dung lượng | SHA-256 |", "|---|---|---|"]
        lines += [f"| `{n}` | {human(s)} | `{d}` |" for n, s, d in manifest]
        lines.append("")
    lines += ["## Bắt đầu từ đâu", "",
              "1. `report/BaoCao.pdf` — báo cáo, Phụ lục A có toàn bộ lệnh tái lập.",
              "2. `data/DATA.md` — cách lấy phần dữ liệu không kèm trong gói.",
              "3. `model/MODEL.md` — checkpoint và các mô hình pretrain bên ngoài.", ""]
    with open(os.path.join(out, "MANIFEST.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    print(f"Đã dựng {out}/ — {human(total)}")
    for entry in sorted(os.listdir(out)):
        path = os.path.join(out, entry)
        if os.path.isdir(path):
            n = sum(len(f) for _, _, f in os.walk(path))
            print(f"  {entry}/  ({n} file)")
        else:
            print(f"  {entry}")
    if not MODEL_URL and not args.with_model:
        print("\n! MODEL_URL còn trống: model/MODEL.md sẽ ghi rõ là thiếu liên kết.\n"
              "  Tải best.pt lên Hugging Face hoặc Drive, điền MODEL_URL, rồi chạy lại.")

    if args.zip:
        archive = out + ".zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for root, _, files in os.walk(out):
                for name in files:
                    path = os.path.join(root, name)
                    zf.write(path, os.path.relpath(path, os.path.dirname(out)))
        print(f"Đã nén {archive} — {human(os.path.getsize(archive))}")


if __name__ == "__main__":
    main()

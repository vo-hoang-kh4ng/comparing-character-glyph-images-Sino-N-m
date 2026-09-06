# Đồ án NLP: Tìm kiếm ảnh ký tự Hán-Nôm tương đồng

Repo: <https://github.com/vo-hoang-kh4ng/comparing-character-glyph-images-Sino-N-m>
Mô hình đã huấn luyện: <https://huggingface.co/vohoangkh4ng/chinese-clip-ft-sinonom>
Báo cáo: [`report/paper.pdf`](report/paper.pdf) — 9 trang, định dạng bài hội nghị.

Cho một ảnh glyph Hán-Nôm, tìm những ký tự trong corpus 26.044 ảnh có hình dạng gần nó nhất. Hai
tình huống sử dụng: **Phần 1** tìm trên toàn corpus khi chưa biết chữ đó là gì; **Phần 2** xếp hạng
trong nhóm ký tự cùng âm Quốc Ngữ khi đã biết chữ đó đọc là gì.

Kết quả chính, đo trên 59 ảnh **scan viết tay thật** (không một pixel nào đi vào huấn luyện):

| | ResNet18 (bản gốc) | Chinese-CLIP zero-shot | **Fine-tune (bài này)** |
|---|---:|---:|---:|
| Phần 1 — `hit@1` trên 26.044 ký tự | — | 0,0847 | **0,5932** |
| Phần 2 — đúng top-1 trong nhóm | 0,4407 | 0,4746 | **0,8136** |
| `radical@k` toàn corpus (chỉ số proxy) | 0,2316 | 0,5361 | — |

---

# Tái lập (reproduce)

Phần này là hướng dẫn đầy đủ để dựng lại mọi con số trong báo cáo, từ một bản clone sạch. Đọc hết
mục 0 trước khi chạy lệnh nào.

## 0. Cần chuẩn bị gì

**Dữ liệu đầu vào của đề bài không nằm trong repo** (ảnh corpus 103 MB, và `*.xlsx` bị `.gitignore`
loại vì là dữ liệu được phát chứ không phải mã nguồn nhóm viết). Cần đặt ba thứ sau vào **thư mục
gốc repo**:

| Cần có | Đặt ở đâu | Bắt buộc cho |
|---|---|---|
| `images/` (giải nén từ `images.zip`, 26.044 file `<UNICODE>.jpg`) | `./images/` | mọi thứ |
| `final_characteristics-v2.xlsx` | `./` | mọi thứ |
| `QuocNgu_SinoNom_Dic.xlsx` | `./` | Phần 2 |
| `test_images/` (giải nén từ `test_images-*.zip`, 59 ảnh scan) | `./test_images/` | mục 3, 5 — **các con số chính** |

Nếu thầy/cô chấm bài: đây đúng là bộ dữ liệu đã phát cho lớp, đặt vào như bảng trên là đủ. Bản sao
dự phòng ở Google Drive:
<https://drive.google.com/drive/folders/12kfJjiM866taJS_-QhOS30TmRi1000Bt>

Nhãn Phần 2 (`output/label_sheets/label_template.csv`) **có** trong repo — đó là nhãn nhóm gán tay,
chạy lại code không sinh lại được, và mọi con số Phần 2 phụ thuộc vào nó.

Phần cứng: mục 1–3 chạy được trên CPU. Mục 4 (fine-tune) cần **GPU ≥ 6 GB** — đo thực tế đỉnh
3,7 GB. Không có GPU thì bỏ qua mục 4 và tải trọng số từ Hugging Face ở mục 5.

## 1. Cài đặt

```bash
git clone https://github.com/vo-hoang-kh4ng/comparing-character-glyph-images-Sino-N-m.git
cd comparing-character-glyph-images-Sino-N-m
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q          # kỳ vọng: 55 passed
```

Mọi lệnh bên dưới viết `.venv/bin/python`; **chạy từ trong thư mục gốc repo**, vì mọi script phân
giải `./images`, `./output`, `./test_images`, `./fonts` theo thư mục làm việc hiện tại.

Lần chạy đầu, backend HuggingFace tự tải trọng số (~600 MB Chinese-CLIP, ~350 MB DINOv2) vào
`~/.cache/huggingface`. Máy không có mạng thì nạp sẵn cache đó hoặc đặt `HF_HOME`.

<details>
<summary>Nếu máy đã cài sẵn torch và không được phép đụng vào (máy server dùng chung)</summary>

Dựng venv kế thừa site-packages của hệ thống để khỏi kéo torch bản khác về:

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install "torchvision==0.21.0" faiss-cpu opencv-python-headless
```

`torchvision` phải khớp bản torch (0.21.0 cho torch 2.6.0), nếu không pip sẽ nâng cấp torch theo.
Dùng `opencv-python-headless` khi server không có `libGL.so.1`.
</details>

## 2. Chỉ số proxy trên toàn corpus → Bảng IV của báo cáo

```bash
PY=.venv/bin/python
for b in resnet18 resnet18-gray chinese-clip dinov2; do
  $PY search_all_chars_in_corpus.py --backend $b --device cuda --batch-size 32
done
$PY search_all_chars_in_corpus.py --backend chinese-clip-large --device cuda --batch-size 32 --dtype fp16
$PY search_all_chars_in_corpus.py --backend chinese-clip-huge  --device cuda --batch-size 32 --dtype fp16
$PY benchmark_extractors.py            # -> output/benchmark_summary.xlsx
$PY analyze_metric_ceiling.py          # trần của chính chỉ số radical@k
$PY benchmark_search.py                # Faiss chính xác so với ANN
```

Bỏ `--device cuda` nếu chạy CPU (chậm hơn nhiều với các backend ViT; thêm `--limit 500` để thử
nhanh). Embedding được cache vào `output/embeddings_<backend>.npz` và dùng lại ở mọi bước sau.

**Kỳ vọng** — `radical@k` (k=20): `resnet18` 0,2316 · `resnet18-gray` 0,2202 · `chinese-clip` 0,4843
· `chinese-clip-large` **0,5361** · `chinese-clip-huge` 0,5147 · `dinov2` 0,3081.

## 3. Đánh giá trên scan thật, cấu hình zero-shot → Bảng V, VI

```bash
$PY evaluate_test_images.py --backend chinese-clip-large --dtype fp16 --device cuda
$PY evaluate_test_images.py --part 2 --backend chinese-clip-large --dtype fp16 --device cuda \
      --labels output/label_sheets/label_template.csv
```

**Kỳ vọng** — Phần 1: `hit@1 = 5/59 = 0,0847`, `hit@20 = 0,2034`, `MRR = 0,1143`.
Phần 2: embedding 28/59 = 0,4746 · histogram 26/59 = 0,4407.

## 4. Sinh dữ liệu huấn luyện và fine-tune → mục IV, VII của báo cáo

```bash
bash download_fonts.sh                  # 7 font Hán-Nôm tự do -> ./fonts (~200 MB)
$PY render_fonts.py                     # 131.604 ảnh -> output/rendered/ (~25 phút)
$PY scan_augment.py --calibrate         # bảng hiệu chỉnh suy giảm; kỳ vọng sai lệch log 0,880 -> 0,103
$PY finetune_glyph.py --smoke           # 200 lớp, kiểm đường ống trước, ~1 phút
$PY finetune_glyph.py --epochs 6 --device cuda    # ~25 phút -> output/finetune/best.pt
```

`finetune_glyph.py` gọi `scan_augment.degrade` **trực tuyến** trong DataLoader, nên **không cần**
`render_fonts.py --augment`. Dòng đánh giá in ra mỗi 600 bước có cột `cos=` — đó là cosine trung
bình giữa các cặp vector ngẫu nhiên; nó phải đi từ ~0,89 về ~0,00. **Nếu `cos` tiến về 1,0 thì biểu
diễn đang sụp đổ**, dừng lại chứ đừng chờ hết epoch.

Muốn tái lập lần thất bại đã ghi trong báo cáo (mục VII-C): `--loss arcface`.

## 5. Chấm lại bằng mô hình đã fine-tune → Hình 1, Bảng V, VI

Bỏ qua mục 4 được — tải thẳng trọng số đã công bố:

```bash
mkdir -p output/finetune
curl -L -o output/finetune/best.pt \
  https://huggingface.co/vohoangkh4ng/chinese-clip-ft-sinonom/resolve/main/best_fp16.pt
```

Rồi chạy đúng đường ống cũ, chỉ đổi tên backend:

```bash
$PY search_all_chars_in_corpus.py --backend chinese-clip-ft --dtype fp16 --device cuda
$PY evaluate_test_images.py --backend chinese-clip-ft --dtype fp16 --device cuda
$PY evaluate_test_images.py --part 2 --backend chinese-clip-ft --dtype fp16 --device cuda \
      --labels output/label_sheets/label_template.csv
$PY evaluate_test_images.py --part 2 --backend chinese-clip-ft --dtype fp16 --device cuda \
      --labels output/label_sheets/label_template.csv --min-confidence high
```

**Kỳ vọng** — Phần 1: `hit@1 = 35/59 = 0,5932`. Phần 2: 48/59 = **0,8136** (MRR 0,8983); lọc
`--min-confidence high`: 45/53 = 0,8491.

Bản fp16 cho embedding **giống hệt** bản fp32: đường suy luận gọi `model.to(fp16)` ngay sau khi nạp,
đã kiểm trên cả 26.044 vector, sai khác tuyệt đối lớn nhất 0,0.

## 6. Tầng xếp hạng lại → mục VIII

```bash
S="--suite render --sample 600"
$PY evaluate_rerank.py $S --no-rerank      # tầng 1 đơn thuần
$PY evaluate_rerank.py $S --oracle-meta    # trần: giả sử biết đúng siêu dữ liệu của truy vấn
$PY evaluate_rerank.py $S                  # xếp hạng lại thật
$PY evaluate_rerank.py $S --normalize none
```

Đây là **kết quả âm** và được báo cáo như vậy: oracle cho thấy còn +17,5 điểm dư địa, nhưng bước
suy ra bộ thủ từ ảnh truy vấn chỉ đạt 11,86% nên tầng này chưa dùng được.

## 7. Dựng lại báo cáo và gói nộp

```bash
$PY make_figures.py                        # 5 hình -> report/figs/*.pdf
tectonic -X compile report/paper.tex       # -> report/paper.pdf (9 trang)
$PY build_report.py                        # -> report/BaoCao.docx + .pdf (bản Doc)
$PY package_submission.py --zip            # -> submission/ + submission.zip
```

`paper.tex` **bắt buộc** biên dịch bằng XeLaTeX hoặc LuaLaTeX (không dùng pdflatex): bài có chữ Nôm
ngoài mặt phẳng cơ bản Unicode nên phải qua `fontspec`. Cách nhẹ nhất là
[tectonic](https://tectonic-typesetting.github.io/) — một file nhị phân, không cần sudo, tự tải gói
TeX khi cần:

```bash
curl -fsSL https://drop-sh.fullyjustified.net | sh    # -> ./tectonic
```

Nếu chỉ muốn đọc báo cáo thì không cần cài gì: [`report/paper.pdf`](report/paper.pdf) đã có sẵn
trong repo.

## Nếu số chạy ra khác

- **Chênh ở chữ số thứ 4 trở đi** là bình thường (thứ tự thu gọn số thực trên GPU khác nhau).
  Chênh lớn hơn thì có gì đó thật sự khác.
- **`hit@1` bằng 0 ở mục 5** — gần như chắc chắn là checkpoint không nạp. `chinese-clip-ft` cố ý
  **báo lỗi to** khi thiếu `output/finetune/best.pt` thay vì âm thầm quay về zero-shot; nếu thấy con
  số zero-shot dưới cái tên fine-tune thì đó là bug, hãy báo.
- **Phần 2 lệch** — kiểm `output/label_sheets/label_template.csv` còn nguyên. Xoá file đó là mất
  nhãn vĩnh viễn, chạy lại code không sinh lại được.
- **Ô vuông rỗng thay cho chữ Nôm trong PDF** — thiếu font; chạy `bash download_fonts.sh`.
  `build_report.py` tự chép font vào `~/.fonts`.
- **`pytest` báo skip** — các test đọc `final_characteristics-v2.xlsx` sẽ skip khi thiếu bảng đó.
  Thiếu dữ liệu là *skip*, không phải *fail*.

---

# Các thành phần

- **Phần 1 — Tìm top-k tương đồng character, so sánh với toàn bộ corpus**
  Dùng `search_all_chars_in_corpus.py` + `images.zip` (ảnh từng character) +
  `final_characteristics-v2.xlsx`.
  Ý tưởng: trích xuất đặc trưng ảnh của từng ký tự, sau đó dùng
  [Faiss](https://github.com/facebookresearch/faiss) để tìm top-k vector tương đồng nhanh trên
  toàn bộ corpus.
  Feature extractor có thể thay thế được (`feature_extractors.py`) — chọn qua `--backend`:

  | backend | model | dim |
  |---|---|---|
  | `resnet18` | ResNet18 ImageNet (baseline gốc, giữ nguyên để so sánh) | 512 |
  | `resnet18-gray` | ResNet18 nhưng `conv1` khởi tạo từ trọng số pretrained thay vì random | 512 |
  | `chinese-clip` | [OFA-Sys/chinese-clip-vit-base-patch16](https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16) | 512 |
  | **`chinese-clip-large`** | [OFA-Sys/chinese-clip-vit-large-patch14](https://huggingface.co/OFA-Sys/chinese-clip-vit-large-patch14) — **mặc định nên dùng** | 768 |
  | `chinese-clip-huge` | [OFA-Sys/chinese-clip-vit-huge-patch14](https://huggingface.co/OFA-Sys/chinese-clip-vit-huge-patch14) | 1024 |
  | `dinov2` | [facebook/dinov2-base](https://huggingface.co/facebook/dinov2-base) | 1536 |
  | **`chinese-clip-ft`** | ViT-B/16 + `output/finetune/best.pt` — **tốt nhất trên scan thật**, xem mục 5 phần Tái lập | 512 |

- **So sánh các backend** — `benchmark_extractors.py`
  Đo tốc độ + 3 chỉ số proxy chất lượng (`radical@k`, `stroke_mae@k`, `ids_jaccard@k`) suy ra từ
  các cột `RADICAL` / `STROKE_NUM` / `SHAPE_MORPH` mà pipeline chính không dùng tới, cộng thêm
  contact sheet ảnh (`output/contact_sheet_<backend>.png`) để đối chiếu bằng mắt.

  **Kết quả đã đo trên full corpus: [`BENCHMARK.md`](BENCHMARK.md).** Tóm tắt —
  `chinese-clip-large` thắng baseline ResNet18 **2,31×** ở `radical@k` (0,5361 vs 0,2316) và
  **2,05×** ở `ids_jaccard@k`. Scale tiếp lên `chinese-clip-huge` thì *kém đi*, nên ViT-L là điểm
  ngọt. `dinov2` thắng ở `stroke_mae@k` nên bổ sung cho họ CLIP.

- **So sánh phương án search** — `benchmark_search.py`
  Đối chiếu Faiss `IndexFlatIP` (chính xác tuyệt đối) với HNSW / IVFFlat / IVFPQ về recall@20 và độ
  trễ, ở nhiều cỡ corpus. Kết luận: **giữ `IndexFlatIP`** — ở 26k, search chỉ chiếm 6% thời gian;
  ANN chỉ đáng dùng khi corpus vượt ~300k. Chi tiết ở mục 7 của `BENCHMARK.md`.

- **`radical@k` dừng ở 0,5 — bão hoà hay là trần của chính nhãn?** — `analyze_metric_ceiling.py`
  Trần lý thuyết của `radical@20` là **0,9862** (chỉ 3% query bị chạm trần) nên **không phải** do
  top-k thiếu. Nhưng `RADICAL` không phải nhãn thị giác thuần tuý: bộ thủ chỉ thật sự xuất hiện
  trong hình chữ ở **56,7%** số ký tự, và một oracle *biết trước* phân rã IDS cũng chỉ đạt **0,7338**.
  So với trần đúng đó, `chinese-clip-large` (0,5361) đang ở **73% quãng đường** và **24,4× mức ngẫu
  nhiên**. Chi tiết + hướng đi tiếp theo ở mục 9 của `BENCHMARK.md`.

- **Sinh cặp dương để fine-tune** — `render_fonts.py`

  Corpus chỉ có 1 ảnh/ký tự nên không có cặp "cùng chữ, khác kiểu" nào để dạy mô hình bất biến kiểu
  chữ. Script render lại toàn bộ 26.044 ký tự qua 7 kiểu chữ (minh, tống, khải, hắc, phỏng tống),
  phủ hợp nhất **100,0%**, ra 131.604 ảnh gán nhãn sẵn bằng mã Unicode. Font tải qua
  `download_fonts.sh`.

- **Làm ảnh render giống scan thật** — `scan_augment.py`

  Render sạch vẫn quá dễ so với scan thật (hit@1 0,70 vs 0,07). Đo 6 chỉ số trên 59 ảnh scan cho
  thấy khoảng cách nằm ở **độ nét (6,55×), tương phản (2,54×), độ dày nét và nhiễu** — không phải ở
  font. `degrade()` mô phỏng đúng sáu trục đó, hiệu chỉnh theo số đo (`--calibrate`: sai lệch log
  0,880 → 0,103). Kéo hit@1 về 0,1986, tức từ *dễ hơn 9,4×* xuống *dễ hơn 2,8×*. Bật bằng
  `render_fonts.py --augment N`. Phần 2,8× còn lại là khác biệt **cấu trúc** (bút lông, hành thư),
  phải có dữ liệu thư pháp thật mới lấp được.

- **Đánh giá trên ảnh query THẬT** — `evaluate_test_images.py`
  Tên file trong `test_images/` gõ Telex (`cofn.jpg` = "còn") nên **giải mã ra được nhãn thật** —
  đây là phép đo duy nhất không phải proxy. Kết quả: tìm trên toàn corpus đạt **hit@1 = 0,0847**
  (179× ngẫu nhiên, nhưng tuyệt đối là thấp). Nguyên nhân là **lệch phân phối**: query là chữ viết
  tay/hành thư, corpus là khải thư in bằng font. Chi tiết ở mục 10 của `BENCHMARK.md`.
  Cả 59 ảnh đã có nhãn người gán, nên Phần 2 cũng có accuracy thật: **0,4746** (embedding) so với
  **0,4407** (histogram) — chênh đúng 2 ảnh, xem cảnh báo ở mục 10.6 về việc không được đọc khoảng
  cách này là kết luận.

- **Báo cáo nộp** — `report/paper.pdf`, **9 trang, định dạng bài hội nghị (IEEEtran) hai cột, 5
  hình**, kèm ba phụ lục: toàn bộ lệnh tái lập, môi trường và chi phí tính toán, cấu trúc gói nộp.
  Nguồn là `report/paper.tex`; biên dịch bằng `tectonic -X compile report/paper.tex` (bắt buộc
  XeLaTeX/LuaLaTeX vì có chữ Nôm ngoài BMP). Hình do `make_figures.py` sinh từ số liệu thật vào
  `report/figs/` — sửa script rồi chạy lại, đừng sửa tay file PDF.
  Bản Doc cùng nội dung: `build_report.py` → `report/BaoCao.docx` + `.pdf` (9 trang).
  `report/report.tex` là bản kỹ thuật dài **đã bị thay thế**, giữ lại làm lịch sử.

- **Mô hình đã huấn luyện** — [vohoangkh4ng/chinese-clip-ft-sinonom](https://huggingface.co/vohoangkh4ng/chinese-clip-ft-sinonom)
  trên Hugging Face. Bản fp16, 172 MB; model card ghi đủ kết quả kèm khoảng tin cậy, code dùng
  thử, 6 hạn chế, và lần ArcFace bị collapse.

- **Đóng gói bài nộp** — `package_submission.py` → `submission/` gồm mã nguồn, tập đánh giá + nhãn
  người gán, checkpoint (bản fp16, 165 MB, embedding **giống hệt từng bit** bản fp32), báo cáo, và
  `MANIFEST.md` có checksum. Mô hình pretrain bên ngoài chỉ ghi định danh, không nộp kèm.

- **Fine-tune** — `finetune_glyph.py` ← **kết quả lớn nhất của đồ án**
  Học bất biến kiểu chữ bằng SupCon trên 141.981 ảnh render đa font, nhãn = mã Unicode. Trên chính
  59 scan thật đó (không một pixel nào vào huấn luyện):

  | | zero-shot `large` | **finetuned** |
  |---|---:|---:|
  | Phần 1, hit@1 trên 26.044 ký tự | 0,0847 | **0,5932** |
  | Phần 2, đúng top-1 trong nhóm | 0,4746 | **0,8136** |

  Hai khoảng tin cậy 95% **rời hẳn nhau** — đây là lần đầu n=59 đủ để kết luận, vì hiệu ứng đủ lớn
  chứ không phải vì mẫu to ra. Lớp Unicode *chưa từng thấy* lúc huấn luyện đạt đúng bằng lớp đã thấy
  (0,9939 vs 0,9954), nên không phải thuộc lòng. Mục 14 của `BENCHMARK.md` ghi cả lần ArcFace bị
  **collapse** trước đó và vì sao.

- **Phần 2 — Tìm top-k tương đồng character, so sánh trong nhóm cùng nghĩa "Quốc Ngữ"**
  Dùng `search_use_QuocNgu_mapping.py` + `images.zip` + `final_characteristics-v2.xlsx` +
  `QuocNgu_SinoNom_Dic.xlsx`.
  Ý tưởng giống Phần 1 nhưng chỉ so sánh giữa các ký tự cùng nghĩa Quốc Ngữ (phạm vi nhỏ nên dùng
  sort thông thường, không cần Faiss).

Chi tiết cài đặt của từng script xem `CLAUDE.md`; toàn bộ phân tích, mọi lần chạy và mọi kết
quả âm ở [`BENCHMARK.md`](BENCHMARK.md).

## Tài liệu tham khảo

- Ý tưởng thay thế feature extractor: DINOv2 + MetaCLIP (Vision Transformer) thay cho ResNet18,
  gợi ý từ anh Khoa NCS — kỳ vọng trích xuất đặc trưng ảnh tốt hơn CNN cơ bản.
- Ý tưởng tìm similar bằng cách decompose các thành phần của Sino-Nôm Script:
  - Paper: https://arxiv.org/pdf/2309.01083
  - Source code: https://github.com/FudanVI/FudanOCR/tree/main
- Model tham khảo khác: [Chinese-CLIP (OFA-Sys)](https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16)

## Hướng cải tiến

Theo thứ tự chi phí tăng dần (lập luận đầy đủ ở mục 9.5 của `BENCHMARK.md`):

1. ~~Gán nhãn `test_images/`~~ — **xong**, 59/59 ảnh, xem mục 10.6 của `BENCHMARK.md`.
2. **Scan thêm ảnh thật.** Đây mới là nút thắt hiện tại: n = 59 cho khoảng tin cậy 95% rộng ±13,3
   điểm, nên mọi cải tiến dưới ~15 điểm đều không chứng minh được. Cần n ≈ 150 để xuống ±8 điểm.
3. **Đánh giá bằng người** trên ~200 cặp — các chỉ số còn lại đều là nhãn yếu.
4. **Ensemble `chinese-clip-large` + `dinov2`** — hai model chỉ trùng 12,9% top-k và mạnh ở hai chỉ
   số khác nhau, dùng lại cache có sẵn nên gần như miễn phí.
5. ~~Finetune metric learning~~ — **xong**, xem mục 14 của `BENCHMARK.md`. Đòn bẩy tiếp theo là
   **hard negative mining**: loss rơi về 0,0068 chỉ sau 400 bước vì 24 lớp lấy ngẫu nhiên từ 23.440
   thì phân biệt quá dễ, nên phải nhồi vào cùng batch những chữ *trông giống nhau*.

~~Thử các embedding model dòng Transformer~~ — đã làm, xem `BENCHMARK.md`.

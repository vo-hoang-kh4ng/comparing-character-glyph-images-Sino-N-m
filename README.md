# Đồ án NLP: Image Comparison (Sino-Nôm Character Similarity)

Đồ án tìm kiếm ký tự Hán-Nôm (Sino-Nôm) tương đồng về mặt hình ảnh (glyph), phục vụ cho các bài
toán OCR/đối chiếu chữ Nôm cổ.

## Nội dung chính

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

Chi tiết pipeline của từng script xem thêm trong `README.txt` và `CLAUDE.md`.

## Dữ liệu

Dữ liệu (ảnh corpus, file mapping) không được track trong git do dung lượng lớn — tải tại Google
Drive:

https://drive.google.com/drive/folders/12kfJjiM866taJS_-QhOS30TmRi1000Bt

Bao gồm:
- Bộ ảnh `.zip` để đối sánh tương đồng (giải nén vào thư mục `./images/`)
- `test_images-*.zip` — ảnh query thật để đánh giá (giải nén vào `./test_images/`)
- File mapping hỗ trợ (`final_characteristics-v2.xlsx`, `QuocNgu_SinoNom_Dic.xlsx`, đặt ở thư mục
  gốc repo)
- File code `.py` tham khảo

## Cài đặt

```bash
pip install -r requirements.txt
```

Sau khi tải dữ liệu từ Drive:
1. Giải nén `images.zip` vào `./images/`
2. Đặt `final_characteristics-v2.xlsx` và `QuocNgu_SinoNom_Dic.xlsx` ở thư mục gốc
3. Tạo thư mục `./output/` (cần cho `search_use_QuocNgu_mapping.py`; script Phần 1 tự tạo)

Lần chạy đầu, backend HuggingFace sẽ tự tải weights (~600 MB cho Chinese-CLIP, ~350 MB cho
DINOv2) và cache vào `~/.cache/huggingface`.

## Chạy thử

```bash
# Phần 1 - chọn backend qua --backend
python search_all_chars_in_corpus.py --backend chinese-clip
python search_all_chars_in_corpus.py --backend resnet18       # baseline gốc
python search_all_chars_in_corpus.py --backend dinov2 --limit 500   # chạy thử nhanh

# So sánh các backend (chạy Phần 1 cho từng backend trước, embeddings được cache lại)
python benchmark_extractors.py            # mặc định chạy cả sáu backend

# So sánh phương án search, và phân tích trần của chỉ số
python benchmark_search.py
python analyze_metric_ceiling.py

# Đánh giá trên ảnh query thật (cần giải nén test_images-*.zip vào ./test_images/)
python evaluate_test_images.py --device cuda
python evaluate_test_images.py --sheets      # phiếu gán nhãn: query + TOÀN BỘ ứng viên
python evaluate_test_images.py --part 2 --labels output/label_sheets/label_template.csv
python evaluate_test_images.py --part 2 --labels output/label_sheets/label_template.csv \
       --min-confidence high                 # chỉ chấm trên 53 nhãn chắc chắn

bash download_fonts.sh          # tải font CJK vào ./fonts
python render_fonts.py          # 131.604 ảnh render đa font -> output/rendered/
python scan_augment.py --calibrate           # bảng hiệu chỉnh suy giảm
python render_fonts.py --augment 2 --strength 1.0   # thêm bản giống scan; KHÔNG cần cho fine-tune,
                                                   # finetune_glyph.py suy giảm online lúc nạp

# Fine-tune (mục 14 BENCHMARK.md) - ~30 phút, 3,7 GB VRAM
python finetune_glyph.py --smoke                   # 200 lớp, kiểm đường ống trước
python finetune_glyph.py --epochs 6 --device cuda  # -> output/finetune/best.pt

# Chấm lại bằng chính đường ống cũ, backend mới nạp thẳng checkpoint
python search_all_chars_in_corpus.py --backend chinese-clip-ft --dtype fp16 --device cuda
python evaluate_test_images.py --backend chinese-clip-ft --dtype fp16 --device cuda \
       --labels output/label_sheets/label_template.csv

# Báo cáo và gói nộp
python make_figures.py                             # -> report/figs/fig1..fig5.pdf
tectonic -X compile report/paper.tex               # -> report/paper.pdf (báo cáo chính, 9 trang)
python build_report.py                             # -> report/BaoCao.docx + BaoCao.pdf (bản Doc)
python package_submission.py --zip                 # -> submission/ và submission.zip

# Phần 2 - mặc định dùng embedding, tái dùng cache của Phần 1
python search_use_QuocNgu_mapping.py --word ta --image ./images/54B1.jpg --method both
```

Lưu ý tốc độ (CPU 16 nhân, không GPU, batch 64): ResNet18 ~350-400 ảnh/s (chạy full 26k ảnh
~1 phút), còn 2 backend ViT chỉ ~5-10 ảnh/s (~1 tiếng cho full corpus). Dùng `--limit` khi thử
nghiệm, và chạy full ở chế độ nền.

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

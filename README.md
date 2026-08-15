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
  | `dinov2` | [facebook/dinov2-base](https://huggingface.co/facebook/dinov2-base) | 1536 |

- **So sánh các backend** — `benchmark_extractors.py`
  Đo tốc độ + 3 chỉ số proxy chất lượng (`radical@k`, `stroke_mae@k`, `ids_jaccard@k`) suy ra từ
  các cột `RADICAL` / `STROKE_NUM` / `SHAPE_MORPH` mà pipeline chính không dùng tới, cộng thêm
  contact sheet ảnh (`output/contact_sheet_<backend>.png`) để đối chiếu bằng mắt.

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
python benchmark_extractors.py --backends resnet18 resnet18-gray chinese-clip dinov2

# Phần 2: chỉnh input_text + test_image_path trong file trước khi chạy,
# và cần có ảnh test trong ./test_images/
python search_use_QuocNgu_mapping.py
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

- Thử các embedding model dòng Transformer để cải thiện chất lượng đặc trưng ảnh so với ResNet18.
- Áp dụng hướng decompose thành phần chữ Sino-Nôm theo paper tham khảo ở trên.

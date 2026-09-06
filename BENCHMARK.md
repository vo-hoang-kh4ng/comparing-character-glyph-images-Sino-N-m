# So sánh các Feature Extractor cho bài toán tìm ký tự Sino-Nôm tương đồng

Báo cáo phần việc: **thay ResNet18 bằng feature extractor họ Transformer và đo hiệu quả so với
baseline.**

Toàn bộ số liệu trong báo cáo này tái lập được bằng:

```bash
# thêm --dtype fp16 cho chinese-clip-large và chinese-clip-huge
.venv/bin/python search_all_chars_in_corpus.py --backend <backend> --device cuda --batch-size 32
.venv/bin/python benchmark_extractors.py   # mặc định chạy cả sáu backend
.venv/bin/python benchmark_search.py       # nghiên cứu exact vs ANN ở mục 7
.venv/bin/python analyze_metric_ceiling.py # trần của radical@k ở mục 9

# mục 10 — đánh giá trên ảnh query thật, nhãn lấy từ tên file Telex
.venv/bin/python evaluate_test_images.py --device cuda
.venv/bin/python evaluate_test_images.py --device cuda --normalize query   # biến thể B ở 10.4
.venv/bin/python evaluate_test_images.py --device cuda --normalize both    # biến thể C ở 10.4
.venv/bin/python evaluate_test_images.py --sheets   # xuất phiếu cho nhóm gán nhãn Phần 2

bash download_fonts.sh                    # tải font CJK vào ./fonts (~200MB, không commit)
.venv/bin/python render_fonts.py           # sinh cặp dương đa font ở mục 13
.venv/bin/python scan_augment.py --calibrate  # bảng hiệu chỉnh ở mục 13.5
.venv/bin/python render_fonts.py --augment 2 --strength 1.0  # thêm bản giống scan

# Phần 2 (mục 8) — so sánh trực tiếp hai phương pháp trên cùng một query
.venv/bin/python search_use_QuocNgu_mapping.py --word ta --image ./images/54B1.jpg --method both

# mục 15 — xếp hạng lại bằng siêu dữ liệu; nhãn là mã Unicode nên không vòng tròn
S="--suite render --sample 600 --backend chinese-clip-large --dtype fp16"
.venv/bin/python evaluate_rerank.py $S --no-rerank      # tầng 1 đơn thuần
.venv/bin/python evaluate_rerank.py $S --oracle-meta    # trần (chẩn đoán, không phải kết quả)
.venv/bin/python evaluate_rerank.py $S                  # xếp hạng lại thật
.venv/bin/python evaluate_rerank.py $S --normalize none # bản trước B2
.venv/bin/python -m pytest -q                           # 55 test cho rerank/evaluate_rerank
```

---

## 1. Mục tiêu

Baseline của đồ án dùng ResNet18 (CNN, pretrain ImageNet) để trích đặc trưng ảnh glyph, rồi tìm
top-K lân cận bằng Faiss. Câu hỏi cần trả lời:

1. ResNet18 baseline yếu tới mức nào, và **nguyên nhân nằm ở đâu**?
2. Feature extractor họ Transformer có thật sự tốt hơn không, và **hơn bao nhiêu**?

## 2. Thiết lập thí nghiệm

| Hạng mục | Giá trị |
|---|---|
| Corpus | 26.044 ảnh glyph (trên tổng 31.208 dòng bảng; 5.164 dòng không có ảnh) |
| top-K | 20 |
| Chỉ mục | Faiss `IndexFlatIP` trên vector đã chuẩn hoá L2 (⇒ inner product = cosine) |
| Phần cứng | NVIDIA H100 80GB (dùng chung với 4 engine vLLM), 16 nhân CPU |
| Môi trường | Python 3.10.12, torch 2.6.0+cu124, transformers 4.57.6 |
| Tính tất định | `set_deterministic(seed=42)` — seed random/numpy/torch, tắt cudnn benchmark |

Sáu backend được so sánh:

| backend | mô hình | chiều | dtype | vai trò |
|---|---|---|---|---|
| `resnet18` | ResNet18 ImageNet | 512 | fp32 | baseline gốc, giữ nguyên từng dòng |
| `resnet18-gray` | ResNet18, `conv1` khởi tạo từ trọng số pretrained | 512 | fp32 | biến thể kiểm chứng giả thuyết |
| `chinese-clip` | OFA-Sys/chinese-clip-vit-base-patch16 | 512 | fp32 | ViT-B/16, pretrain trên cặp ảnh–văn bản tiếng Trung |
| `chinese-clip-large` | OFA-Sys/chinese-clip-vit-large-patch14 | 768 | fp16 | ViT-L/14, cùng họ — cô lập ảnh hưởng của kích thước |
| `chinese-clip-huge` | OFA-Sys/chinese-clip-vit-huge-patch14 | 1024 | fp16 | ViT-H/14, kiểm tra scale còn lợi tới đâu |
| `dinov2` | facebook/dinov2-base | 1536 | fp32 | ViT tự giám sát, CLS ⊕ mean patch tokens |

Ba backend `chinese-clip*` cùng một họ, chỉ khác kích thước và giữ nguyên dữ liệu pretrain — nên
chuỗi base → large → huge cô lập được đúng ảnh hưởng của việc scale mô hình.

### Hai thủ thuật bộ nhớ để model lớn vừa VRAM

GPU chỉ còn ~3,05 GB trống (4 engine vLLM chiếm phần còn lại), trong khi ViT-H/14 fp32 cần ~3,8 GB.
Hai thay đổi giúp nó vừa:

1. **Bỏ tháp text trước khi đưa lên GPU.** `get_image_features` chỉ dùng `vision_model` +
   `visual_projection`, nhưng `ChineseCLIPModel` nạp thêm cả một RoBERTa hoàn chỉnh — riêng với
   ViT-H là ~1,3 GB nạp lên rồi không bao giờ dùng.
2. **fp16.** Giảm một nửa trọng số.

**fp16 được kiểm chứng chứ không phải giả định.** So trên chính `chinese-clip` base (đã có số fp32):
cosine(fp32, fp16) ≥ 0,99999 và **trùng 99,31%** danh sách top-20. Nên số của các model fp16 vẫn so
sánh công bằng được với các model fp32.

### Về `resnet18-gray`

Baseline thay `conv1` của ResNet18 bằng một conv 1 kênh mới. Thao tác này **vứt bỏ toàn bộ bộ lọc
tầng đầu đã pretrain và thay bằng trọng số ngẫu nhiên** — nghĩa là mọi tầng phía sau đang xử lý đầu
ra của một bộ dò biên ngẫu nhiên. `resnet18-gray` sửa đúng một dòng đó: cộng 3 kênh RGB của bộ lọc
pretrain thành 1 kênh, giữ lại prior ImageNet. Đây là biến thể để kiểm chứng giả thuyết "baseline
yếu vì lỗi `conv1`".

## 3. Phương pháp đánh giá

Pipeline tìm kiếm **chỉ dùng ảnh**. Vì vậy có thể lấy các cột metadata mà pipeline không hề đụng tới
trong `final_characteristics-v2.xlsx` làm nhãn yếu (weak label) để chấm điểm:

| chỉ số | định nghĩa | nguồn |
|---|---|---|
| `radical@k` ↑ | tỉ lệ lân cận top-K cùng bộ thủ với query | `RADICAL` |
| `stroke_mae@k` ↓ | trung bình \|chênh lệch số nét\| | `STROKE_NUM` |
| `ids_jaccard@k` ↑ | Jaccard giữa các thành phần IDS | `SHAPE_MORPH` |

> **Giới hạn quan trọng.** Đây **không phải** đánh giá của con người. Chúng là các chỉ số thay thế
> (proxy) rẻ tiền, tương quan với "trông giống nhau", đủ để **xếp hạng các backend với nhau**. Đánh
> giá thật (Precision@k với nhãn do người gán) là một phần việc riêng, chưa thực hiện.

## 4. Kết quả

### 4.1 Chất lượng (full corpus 26.044 glyph, top-K = 20)

| backend | chiều | radical@k ↑ | stroke_mae@k ↓ | ids_jaccard@k ↑ |
|---|---|---|---|---|
| `resnet18` (baseline) | 512 | 0,2316 | 2,8942 | 0,1031 |
| `resnet18-gray` | 512 | 0,2202 | 2,5658 | 0,1020 |
| `chinese-clip` | 512 | 0,4843 | 2,6296 | 0,1997 |
| **`chinese-clip-large`** | 768 | **0,5361** | 2,6992 | 0,2113 |
| `chinese-clip-huge` | 1024 | 0,5147 | 2,6547 | **0,2144** |
| `dinov2` | 1536 | 0,3081 | **2,3397** | 0,1251 |

So với baseline, `chinese-clip-large` đạt **radical@k gấp 2,31 lần** và **ids_jaccard gấp 2,05 lần**.

### 4.2 Tốc độ

| backend | CPU 16 nhân | H100 | full 26k trên GPU |
|---|---|---|---|
| `resnet18` | ~347 img/s | 3.950 img/s | 6,6 s |
| `resnet18-gray` | ~403 img/s | 3.937 img/s | 6,6 s |
| `chinese-clip` | ~5–9 img/s | 358 img/s | 73 s |
| `chinese-clip-large` | chưa đo | 354 img/s | 74 s |
| `chinese-clip-huge` | chưa đo | 283 img/s | 92 s |
| `dinov2` | ~7 img/s | 306 img/s | 85 s |

Đáng chú ý: `chinese-clip-large` gấp đôi tham số so với base nhưng **gần như không tốn thêm thời
gian** (354 vs 358 img/s) — fp16 bù lại đúng phần chi phí tăng thêm.

Chuyển sang GPU giúp backend ViT nhanh gấp **~45 lần** (từ ~1 tiếng xuống ~1 phút mỗi lượt), đủ để
chạy full corpus thoải mái thay vì phải dùng `--limit`.

> Số tốc độ chỉ mang tính tham khảo (±10%): H100 dùng chung với 4 engine vLLM, tải thay đổi theo
> thời gian. Ngược lại, các chỉ số chất lượng là tất định và lặp lại chính xác.

### 4.3 Mức đồng thuận giữa các backend

Số lân cận trùng nhau trung bình trên top-20:

| cặp backend | trùng / 20 | tỉ lệ |
|---|---|---|
| `chinese-clip` ↔ `dinov2` | 2,57 | 12,9% |
| `resnet18-gray` ↔ `chinese-clip` | 2,31 | 11,6% |
| `resnet18` ↔ `chinese-clip` | 2,12 | 10,6% |
| `resnet18-gray` ↔ `dinov2` | 1,94 | 9,7% |
| `resnet18` ↔ `resnet18-gray` | 1,81 | 9,0% |
| `resnet18` ↔ `dinov2` | 1,52 | 7,6% |

## 5. Phân tích

### 5.1 Chinese-CLIP thắng rõ rệt, và biên độ lớn

`chinese-clip-large` hơn baseline **2,31 lần** ở radical@k và **2,05 lần** ở ids_jaccard. Đây là kết
quả phù hợp với kỳ vọng: mô hình được pretrain trên dữ liệu ảnh–văn bản tiếng Trung nên prior của nó
gần với chữ Hán hơn hẳn prior ImageNet (vốn học từ ảnh chó, mèo, xe cộ).

Đáng chú ý: bản chạy thử 200 ký tự trước đó cho radical@k = 0,398, tức **dự đoán còn dè dặt hơn**
thực tế full corpus (0,4843 với base). Điều này củng cố việc không nên kết luận từ lát cắt
`--limit` — nhưng ở đây sai lệch nghiêng về hướng an toàn.

### 5.2 Scale có lợi, nhưng bão hoà rồi đảo chiều

Đây là phát hiện đáng giá nhất từ việc thêm hai model lớn:

| bước scale | radical@k | thay đổi |
|---|---|---|
| base → large | 0,4843 → 0,5361 | **+10,7%** |
| large → huge | 0,5361 → 0,5147 | **−4,0%** |

Model to hơn **không** đồng nghĩa tốt hơn. `chinese-clip-huge` nhiều tham số hơn ViT-L đáng kể, tốn
thêm 33% số chiều và 20% thời gian, nhưng lại *kém hơn* ở radical@k. Nó chỉ nhỉnh hơn ở ids_jaccard
(0,2144 vs 0,2113) — mức chênh quá nhỏ để bù chi phí.

Hệ quả thực tế: **`chinese-clip-large` là điểm ngọt**, và nếu ai đề xuất thử tiếp ViT-G hay các
checkpoint lớn hơn nữa thì đây là bằng chứng cho thấy nhiều khả năng không đáng.

Vì sao bão hoà? Giả thuyết hợp lý: các model CLIP lớn được tối ưu cho ảnh tự nhiên đa dạng, trong
khi glyph là ảnh nhị phân, một kênh, cấu trúc rất hẹp. Khi mô hình đủ lớn để bắt được cấu trúc nét,
phần dung lượng tăng thêm dùng cho những đặc trưng bài toán này không cần. Đây mới là giả thuyết,
chưa kiểm chứng.

### 5.3 DINOv2 bổ sung cho họ CLIP, không đơn thuần là kém hơn

Không có một người thắng tuyệt đối:

- Mọi backend `chinese-clip*` thắng ở `radical@k` và `ids_jaccard@k` — tức các tín hiệu **cấu
  trúc/ngữ nghĩa** của chữ.
- `dinov2` thắng ở `stroke_mae@k` (2,3397, tốt nhất trong sáu backend) — tức bám sát **mật độ nét**
  hơn.

Kết hợp với mục 4.3 (hai mô hình chỉ trùng 12,9% kết quả), đây là bằng chứng cho thấy chúng nhìn
chữ theo hai cách khác nhau. Hướng ensemble hoặc rerank là bước tiếp theo hợp lý.

### 5.4 Giả thuyết lỗi `conv1` **không** đúng

`resnet18-gray` sửa đúng lỗi khởi tạo ngẫu nhiên, kết quả:

- `stroke_mae@k` cải thiện: 2,8942 → 2,5658 (tốt hơn 11,3%)
- `radical@k` **tệ đi**: 0,2316 → 0,2202
- `ids_jaccard@k` gần như không đổi: 0,1031 → 0,1020

Vậy lỗi `conv1` **không phải** nguyên nhân chính khiến baseline yếu. Đây là một **kết quả âm**, và
nó có giá trị: nó loại bỏ một giả thuyết rẻ tiền, cho thấy khoảng cách chất lượng đến từ bản thân
kiến trúc/prior của mô hình chứ không phải một dòng code hỏng. Chính điều này biện minh cho việc
chuyển hẳn sang backend ViT.

### 5.5 Các backend đồng thuận rất thấp

Mức trùng lặp top-20 chỉ 7,6–12,9%. Ngay cả cặp gần nhau nhất cũng khác nhau ~87% kết quả trả về.
Hai hệ quả:

1. Các chỉ số proxy đang thực sự **phân biệt được** các backend, chứ không phải đo cùng một thứ.
2. Đánh giá bằng người sẽ mang tính **quyết định**, không phải chỉ để xác nhận — vì các mô hình
   không hội tụ về một đáp án chung nào cả.

### 5.6 Kiểm chứng định tính

Contact sheet (`output/contact_sheet_<backend>.png`) khớp với số liệu. Với query 旴 (U+65F4):

| backend | top-5 lân cận |
|---|---|
| `resnet18` | 貼 护 仲 肚 亜 — không có quan hệ cấu trúc rõ ràng |
| `chinese-clip-large` | 盱 盰 昕 旰 盽 — đều chung cấu trúc 日/目 + 于/干 |

Với query bộ 犭 (𤟂), `resnet18` trả 猜 指 𢯡 褃 棈 (lẫn lộn 犭/扌/礻/米) còn `chinese-clip` trả
猎 狺 猜 獍 獖 (gần như toàn bộ 犭).

Ngược lại, với query bộ 月 (𦝊) thì `dinov2` trả 腿 脙 𦟂 胣 膔 (gần như toàn bộ 月) trong khi
`chinese-clip` lệch sang bộ 辶 — minh hoạ trực quan cho tính bổ sung ở mục 5.3.

> Lưu ý đọc contact sheet: từng query lẻ **không** phản ánh thứ hạng tổng thể. Vẫn có query mà
> `chinese-clip` base cho kết quả gọn hơn `chinese-clip-large` dù xét trung bình toàn corpus thì
> large tốt hơn rõ rệt. Chỉ số tổng hợp ở mục 4.1 mới là căn cứ; contact sheet dùng để phát hiện
> lỗi thô, không dùng để xếp hạng.

## 6. Ghi chú kỹ thuật: `use_fast=True` đã thử và bị loại

Backend HF cảnh báo đang dùng "slow image processor". Đã thử bật `use_fast=True`:

| | preprocess đo riêng | full corpus (chinese-clip) | A/B xen kẽ, 4k ảnh |
|---|---|---|---|
| `use_fast=False` | 548 img/s | **358 img/s** | **595,9 ± 8,9 img/s** |
| `use_fast=True` | 1.700 img/s | 311 img/s | 521,8 ± 144,6 img/s |

Nhanh gấp 3,1 lần khi đo riêng, nhưng **chậm hơn khi chạy thật** trên cả hai backend, và độ lệch
chuẩn cao gấp 16 lần (dao động 320–650 img/s giữa các lượt). Nguyên nhân: đường fast dùng op torch
CPU đa luồng, tranh chấp CPU với các engine vLLM chạy chung máy; đường slow (PIL) đơn luồng nên ổn
định. Đã hoàn tác, và đặt `use_fast=False` tường minh để tắt luôn cảnh báo.

Về chất lượng, hai đường cho embedding **gần như y hệt** (cosine ≥ 0,99996 trên toàn bộ 26.044
vector), nên cảnh báo "minor differences in outputs" của HF không đáng lo ở bài toán này.

## 7. Có nên thay Faiss `IndexFlatIP` bằng chỉ mục xấp xỉ (ANN)?

Gợi ý từ mentor: ngoài Faiss, thử tìm cách search nhanh hơn. `IndexFlatIP` là **brute-force chính
xác** — O(N) mỗi query — nên câu hỏi hoàn toàn hợp lý. Script `benchmark_search.py` trả lời bằng đo
đạc thay vì suy đoán.

**Phương pháp tách hai loại số** để không nguỵ tạo:

- **Recall** chỉ đo trên embedding thật (`chinese-clip-large`), vì recall phụ thuộc phân bố vector.
- **Độ trễ** đo trên dữ liệu thật tới 26k, rồi dùng **vector ngẫu nhiên** cho các cỡ chưa có
  (100k–1M) — thời gian gần như không phụ thuộc dữ liệu thật hay giả. Các dòng này ghi rõ là tổng
  hợp và **không** được dùng để kết luận về recall.

### 7.1 Recall trên dữ liệu thật (N = 26.044, top-K = 20)

| chỉ mục | tham số | recall@20 | build | ms/query | nhanh gấp |
|---|---|---|---|---|---|
| **Flat (exact)** | — | **1,0000** | 0,05 s | 0,195 | 1× |
| HNSW32 | efSearch=32 | 0,9823 | 0,51 s | 0,012 | 16,1× |
| HNSW32 | efSearch=64 | 0,9912 | 0,51 s | 0,010 | 20,4× |
| HNSW32 | efSearch=256 | 0,9967 | 0,51 s | 0,089 | 2,2× |
| IVFFlat | nprobe=32 | 0,9646 | 1,90 s | 0,019 | 10,3× |
| IVFFlat | nprobe=64 | 0,9843 | 1,90 s | 0,044 | 4,5× |
| IVFPQ | nprobe=64 | **0,6047** | 6,16 s | 0,012 | 16,9× |

HNSW là lựa chọn ANN tốt nhất: giữ 99,1% recall mà nhanh gấp 20 lần. IVFPQ nén quá mạnh cho bài
toán này — mất 40% recall, không dùng được.

### 7.2 Nhưng ở cỡ corpus hiện tại, tăng tốc đó vô nghĩa

Exact search hiện đã tốn **0,195 ms/query**. Đổi 1% recall để tiết kiệm 0,19 mili giây là một giao
dịch tồi — nhất là khi recall chính là thứ đang cần cải thiện (radical@k mới 0,54).

Quét toàn corpus (mỗi ký tự query một lần, tức N query × N vector):

| N | ms/query | search toàn bộ | embed @354 img/s | **search chiếm** |
|---|---|---|---|---|
| 26.044 (hiện tại) | 0,195 | 5 s | 74 s | **6%** |
| 100.000 | 0,633 | 63 s | 282 s | **18%** |
| 500.000 | 5,484 | 2.742 s | 1.412 s | **66%** |
| 1.000.000 | 7,532 | 7.532 s | 2.825 s | **73%** |

### 7.3 Kết luận: mentor đúng về nguyên lý, nhưng chưa tới lúc

Điểm mấu chốt là **embed tăng tuyến tính O(N) còn quét toàn corpus tăng O(N²)**. Nên tỉ trọng của
search chỉ có tăng:

- **Ở 26k, search chiếm 6%** — tối ưu nó là tối ưu nhầm chỗ. Kể cả tăng tốc vô hạn cũng chỉ rút
  được 5 giây trên tổng 79 giây.
- **Điểm hoà vốn nằm quanh N ≈ 300.000**, nơi search vượt embed để trở thành chi phí chính.
- **Từ 500k trở đi thì ANN là bắt buộc.** Ở 1M, exact mất **2,1 giờ** mỗi lượt quét, trong khi HNSW
  (efSearch=64) chỉ mất ~154 s build + ~129 s search ≈ **4,7 phút** — nhanh gấp **27 lần**.

**Khuyến nghị: giữ nguyên `IndexFlatIP`.** Corpus 26k còn cách ngưỡng hơn một bậc độ lớn, và tính
chính xác tuyệt đối đáng giá hơn ở giai đoạn còn đang so sánh chất lượng các backend — nếu search
cũng xấp xỉ thì không còn phân biệt được lỗi do model hay do chỉ mục.

Chuyển sang HNSW khi corpus vượt ~300k (ví dụ mở rộng sang nhiều nguồn glyph, hoặc index từng ứng
viên OCR theo trang). Lúc đó dùng `efSearch=64` làm điểm khởi đầu.

> **Lưu ý về giới hạn.** Recall chỉ được đo ở N = 26.044. Recall của HNSW thường **giảm** khi N
> tăng, nên con số 99,1% ở trên không được phép suy diễn cho mốc 1M — cần đo lại khi thật sự tới đó.

## 8. Phần 2: nâng từ histogram lên embedding

`search_use_QuocNgu_mapping.py` xếp hạng các ký tự **cùng một âm Quốc Ngữ** — phạm vi chỉ vài chục
ứng viên nên không cần Faiss. Bản gốc chấm điểm bằng histogram cạnh Canny; nay bổ sung phương án
embedding dùng chung `feature_extractors.py` với Phần 1, chọn qua `--method`.

Ứng viên là tập con của corpus, nên script **đọc thẳng embedding từ cache
`output/embeddings_<backend>.npz`** của Phần 1 — chỉ còn đúng một lượt forward cho ảnh query.

### 8.1 Vì sao histogram yếu

Chuỗi Canny → Otsu → `calcHist` 256 bin thực chất chỉ đếm **số điểm ảnh biên**, không mã hoá vị trí
hay hình dạng nét. Hai chữ khác hẳn nhau nhưng có lượng mực tương đương sẽ cho histogram gần như
giống nhau.

Đây không phải suy đoán. Lấy chính ảnh của 咱 (U+54B1) làm query, trong 25 ứng viên cùng âm "ta":

| phương pháp | hạng 1 | điểm |
|---|---|---|
| embedding (`chinese-clip`) | **咱** (đúng chính nó) | 1,0000 |
| histogram | **些** (chữ khác) | 1,0000 |

Histogram chấm **điểm tuyệt đối 1,0000 cho một chữ hoàn toàn khác** — nó không phân biệt nổi 咱 với
些.

### 8.2 Phép thử tự truy hồi

Dùng chính ảnh của từng ứng viên làm query. Đáp án đúng bắt buộc là chính nó — không cần nhãn người,
và một phương pháp không vượt được phép thử này thì hỏng về nguyên tắc. Trên 7 từ Quốc Ngữ, 64 query:

| phương pháp | tự truy hồi đúng @1 |
|---|---|
| embedding | **64/64 (100%)** |
| histogram | 62/64 (96,9%) |

Hai ca histogram sai đều do **hoà điểm** với chữ khác, đúng cơ chế ở mục 8.1.

### 8.3 Khả năng phân biệt

Với cùng query 咱 trên 25 ứng viên:

| phương pháp | biên độ điểm | độ lệch chuẩn | số ứng viên ≥ 0,99 |
|---|---|---|---|
| embedding | **0,1502** | 0,0322 | 1/25 |
| histogram | 0,0636 | 0,0151 | 2/25 |

Histogram nén toàn bộ ứng viên vào một dải hẹp gấp **2,4 lần**, nên thứ hạng của nó phần lớn là
nhiễu. Hai phương pháp chỉ trùng **6/10** kết quả top-10.

### 8.4 Trạng thái

Chạy được và đã kiểm chứng bằng glyph lấy từ corpus. **Chưa chạy trên ảnh test thật** vì
`test_images/` không có trong bộ dữ liệu Drive — cần ảnh chụp/crop chữ Nôm thật. Lưu ý số ở trên đo
trên ảnh sạch dựng từ font; với ảnh scan cổ có nhiễu, khoảng cách giữa hai phương pháp nhiều khả
năng còn **rộng hơn**, vì histogram đếm điểm biên nên rất nhạy với nhiễu.

## 9. Phản hồi của thầy: `radical@k` có thật sự "bão hoà ở 0,5"?

Thầy nêu ba giả thuyết cho việc `radical@k` dừng quanh 0,5. Mục này kiểm chứng từng cái bằng số đo
thay vì suy đoán. Script: `analyze_metric_ceiling.py`.

### 9.1 Giả thuyết 1 — "giới hạn tập top-k bị thiếu": **bị bác bỏ**

Nếu corpus không đủ ký tự cùng bộ thủ để lấp đầy 20 chỗ, `radical@20` không thể đạt 1,0. Tính trần
đó trực tiếp: với query có bộ thủ *r* thuộc nhóm *c* phần tử, số bạn cùng bộ khả dụng là *c* − 1,
nên trần của riêng query đó là `min(c-1, k)/k`.

| k | trần lý thuyết của `radical@k` |
|---|---|
| 1 | 1,0000 |
| 5 | 0,9992 |
| 10 | 0,9957 |
| **20** | **0,9862** |
| 50 | 0,9536 |

Chỉ **888/26.035 query (3%)** bị trần chạm dưới 1,0. Trần là 0,9862 chứ không phải 0,5 — **kích
thước tập top-k không phải nguyên nhân.**

Bằng chứng thứ hai, mạnh hơn: nghẽn nằm ngay ở láng giềng *gần nhất*, không phải ở đuôi danh sách.

| backend | @1 | @2 | @3 | @5 | @10 | @20 | @50 | @100 |
|---|---|---|---|---|---|---|---|---|
| `chinese-clip` | 0,5825 | 0,5628 | 0,5508 | 0,5347 | 0,5121 | 0,4843 | 0,4398 | 0,3989 |
| **`chinese-clip-large`** | **0,6331** | 0,6124 | 0,5994 | 0,5833 | 0,5606 | **0,5361** | 0,4965 | 0,4565 |
| `chinese-clip-huge` | 0,6081 | 0,5888 | 0,5744 | 0,5588 | 0,5392 | 0,5147 | 0,4745 | 0,4350 |
| `dinov2` | 0,4674 | 0,4417 | 0,4238 | 0,3963 | 0,3549 | 0,3081 | 0,2461 | 0,2019 |
| `resnet18` | 0,3851 | 0,3532 | 0,3349 | 0,3072 | 0,2697 | 0,2316 | 0,1818 | 0,1477 |
| `resnet18-gray` | 0,3836 | 0,3519 | 0,3290 | 0,2998 | 0,2598 | 0,2202 | 0,1716 | 0,1401 |

(Nhân tiện: `resnet18-gray` bám sát `resnet18` ở k=1 (0,3836 vs 0,3851) rồi tụt dần — củng cố kết
quả âm ở mục 5.3, sửa `conv1` không đổi được gì về mặt bộ thủ.)

`radical@1 = 0,6331`: ngay cả ứng viên số 1 cũng chỉ cùng bộ thủ 63% số lần. Đường cong xuống rất
thoải (0,63 → 0,54 khi k đi từ 1 lên 20), tức đây **không** phải hiện tượng "loãng dần vì lấy quá
nhiều". Có nới k xuống 1 thì con số cũng chỉ lên 0,63.

### 9.2 Trần *thực tế* là 0,73 chứ không phải 0,99 — nhãn mới là chỗ nghẽn

Trần 0,9862 giả định tồn tại một bộ rank hoàn hảo. Nhưng `RADICAL` **không phải là nhãn thị giác
thuần tuý**, nên không mô hình ảnh nào chạm tới được nó:

- Bộ thủ chỉ **xuất hiện thật trong phân rã IDS của chính chữ đó ở 56,7%** số ký tự. Với 43,3% còn
  lại, bộ thủ là quy ước tra từ điển, **không nhìn thấy được từ hình chữ.**
- Cặp cùng bộ thủ có IDS-Jaccard trung bình 0,2571 (khác bộ thủ: 0,0009). Tín hiệu có thật nhưng
  yếu — cùng bộ thủ **không** kéo theo trông giống nhau.

Để định lượng trần thực tế, dựng một **oracle**: xếp hạng bằng Jaccard trên thành phần IDS *thật*
(lấy thẳng từ `SHAPE_MORPH`) — tức một bộ rank "biết trước cấu trúc chữ", tốt hơn bất kỳ mô hình thị
giác nào có thể suy ra từ ảnh.

| bộ xếp hạng | `radical@20` |
|---|---|
| ngẫu nhiên | 0,0220 |
| `resnet18` (baseline) | 0,2316 |
| `chinese-clip-large` | **0,5361** |
| **oracle IDS-Jaccard** (biết trước cấu trúc) | **0,7338** |
| trần lý thuyết | 0,9862 |

**`chinese-clip-large` đang đạt 73% của oracle** (0,5361 / 0,7338) và **24,4× mức ngẫu nhiên**.
Khoảng trống còn lại tới oracle là **+37% tương đối**, không phải "gấp đôi". Nói cách khác: 0,5
trông như bão hoà chỉ vì đang so với 1,0; so với trần đúng của bài toán thì nó nằm ở khoảng ba phần
tư quãng đường.

### 9.3 Giả thuyết 2 — "Chinese-CLIP không chuyên nét/bộ thủ": **đúng, và có bằng chứng gián tiếp**

Không đo trực tiếp được, nhưng dữ liệu ủng hộ:

- Chinese-CLIP được huấn luyện contrastive ảnh–văn bản trên ảnh web tiếng Trung; đặc trưng của nó ở
  tầng *ngữ nghĩa cảnh vật*, không phải tầng *nét bút*. Ở đây tháp text đã bị xoá (mục 2), chỉ còn
  tháp ảnh — nó chưa từng được dạy rằng "cùng bộ thủ" là quan hệ đáng quan tâm.
- Chữ ký rõ nhất là **scale bị bão hoà rồi quay đầu**: base → large +10,7%, large → huge **−4,0%**
  (mục 5.2). Thêm dung lượng cho một mục tiêu huấn luyện *sai* thì không giúp gì — nếu chỉ thiếu
  năng lực biểu diễn, huge đã phải tốt hơn large.
- `dinov2` (self-supervised thuần ảnh, không dùng văn bản) lại **thắng ở `stroke_mae@k`** (2,3397 —
  tốt nhất) trong khi thua xa ở `radical@k`. Hai họ model bắt hai loại tín hiệu khác nhau, và chỉ
  trùng 12,9% top-k → chưa model nào bắt được "cấu trúc chữ".

### 9.4 Giả thuyết 3 — finetune: đây mới là đòn bẩy thật, kèm một cái bẫy

Finetune là hướng đúng, và dữ liệu giám sát **đã có sẵn trong repo**: `final_characteristics-v2.xlsx`
cho `RADICAL` (214 lớp), `STROKE_NUM`, và `SHAPE_MORPH` (3.598 thành phần IDS) cho 26.044 ảnh — đủ
để huấn luyện metric learning trên chính corpus này.

**Bẫy cần tránh:** nếu finetune trên nhãn `RADICAL` rồi lại báo cáo `radical@k`, chỉ số sẽ tăng vọt
mà **không chứng minh được gì** — đó là đo lại tập huấn luyện. Bắt buộc phải:

1. Chia **held-out theo bộ thủ** (một số bộ thủ chỉ xuất hiện ở tập test), không chia ngẫu nhiên
   theo ký tự.
2. Huấn luyện trên một nhãn (vd. thành phần IDS) và **đánh giá trên nhãn khác** (bộ thủ, số nét).
3. Giữ nguyên `resnet18` và `chinese-clip-large` zero-shot làm mốc so sánh trong cùng bảng.

### 9.5 Ba việc nên làm, xếp theo chi phí

| # | việc | chi phí | kỳ vọng |
|---|---|---|---|
| 1 | **Đánh giá bằng người** trên ~200 cặp | 1 buổi, không cần GPU | Biết `radical@k` lệch bao nhiêu so với "trông giống". Đây là việc mục 10.7 đã nêu, và giờ là việc **chặn** mọi kết luận khác. |
| 2 | **Ensemble `chinese-clip-large` + `dinov2`** (rerank hoặc cộng điểm) | vài giờ, dùng lại cache | Hai model chỉ trùng 12,9% và mạnh ở hai chỉ số khác nhau — khả năng cộng hưởng cao, chưa cần huấn luyện gì. |
| 3 | **Finetune metric learning** trên thành phần IDS | vài ngày + GPU | Đòn bẩy lớn nhất, nhưng chỉ nên làm sau khi (1) xác nhận proxy đáng tin. Tham chiếu: paper decompose Sino-Nôm trong README. |

### 9.6 Tóm tắt để trả lời thầy

> Em đã kiểm tra cả ba giả thuyết. **Giả thuyết top-k không đúng**: trần lý thuyết của `radical@20`
> là 0,9862 (chỉ 3% query bị chạm trần), và `radical@1` cũng chỉ đạt 0,6331 — nghẽn nằm ngay ở láng
> giềng gần nhất chứ không ở đuôi danh sách.
>
> Nhưng bọn em phát hiện một điều khác: **`RADICAL` không phải nhãn thị giác thuần tuý** — bộ thủ chỉ
> thật sự xuất hiện trong hình chữ ở 56,7% số ký tự. Một oracle *biết trước* phân rã IDS cũng chỉ đạt
> `radical@20` = 0,7338. So với trần đúng đó thì `chinese-clip-large` (0,5361) đang ở **73% quãng
> đường** và **24,4× mức ngẫu nhiên** — không phải bão hoà, mà là còn khoảng +37% tương đối.
>
> **Giả thuyết pretrain sai domain thì khớp với dữ liệu**: scale từ large lên huge làm kết quả *giảm*
> 4,0%, đúng chữ ký của "thêm dung lượng cho mục tiêu huấn luyện sai". Nên finetune đúng là đòn bẩy
> thật. Bọn em định làm theo thứ tự: (1) đánh giá bằng người ~200 cặp để xác nhận proxy đáng tin,
> (2) ensemble large + dinov2 (hai model chỉ trùng 12,9% top-k), (3) finetune metric learning trên
> thành phần IDS với held-out chia theo bộ thủ để tránh đo lại tập huấn luyện.

## 10. Đánh giá trên ảnh query THẬT — phép đo duy nhất không phải proxy

Tất cả các mục trước đều dùng **nhãn yếu** và query là chính ảnh corpus. Mục này khác hẳn: query là
59 ảnh scan trong `test_images/`, và **nhãn là thật**. Script: `evaluate_test_images.py`.

### 10.1 Nhãn đến từ đâu

Tên file được gõ Telex: `cofn.jpg` = "còn", `truwowsc.jpg` = "trước", `nhuwxng.jpg` = "những".
Giải mã ngược rồi tra `QuocNgu_SinoNom_Dic.xlsx` cho ra **tập ký tự Sino-Nôm đúng** của ảnh đó.

Telex cho phép đặt dấu thanh ngay sau nguyên âm (`cofn`) lẫn ở cuối từ (`conf`), nên bộ giải mã sinh
mọi vị trí đặt dấu rồi khớp. Kết quả hiện tại: **59/59 khớp, không có trường hợp nhập nhằng.**

Trước đây con số này là 56/59. Ba ảnh bị loại vì ba lý do khác nhau, và chỉ một trong ba là lỗi dữ liệu:

- `phong_1` — **lỗi code**. `decode_filenames` gọi `words.pop()` trên chính cái `set` đang nằm trong
  `lookup`, nên sau khi khớp `phong.jpg` thì `lookup["phong"]` rỗng và ảnh thứ hai của *bất kỳ* từ
  lặp nào cũng bị âm thầm bỏ. Đã sửa thành `next(iter(words))`.
- `dau` — gõ thiếu dấu mũ. Văn bản là "bể **dâu**" (Kiều câu 3), phân biệt với `ddau.jpg` = "đau".
- `nguwowsi` — gõ `s` (sắc) thay vì `f` (huyền); từ đúng là "người".

Hai trường hợp sau là lỗi gõ tên file, sửa bằng bảng `FILENAME_FIXES` trong `evaluate_test_images.py`
thay vì đổi tên file, để `test_images/` còn khớp byte-đối-byte với zip gốc.

### 10.2 Phần 1 — tìm trên toàn bộ 26.044 ký tự

Câu hỏi: ký tự trả về có *thật sự đọc là từ đó* không?

| chỉ số | `chinese-clip-large` fp16 |
|---|---|
| hit@1 | 5/59 = **0,0847** |
| hit@5 | 9/59 = 0,1525 |
| hit@10 | 11/59 = 0,1864 |
| hit@20 | 12/59 = 0,2034 |
| MRR | 0,1143 |
| baseline ngẫu nhiên hit@1 | 0,000474 |

**Hơn ngẫu nhiên 179×, nhưng tuyệt đối thì thấp.** Đây là con số trung thực đầu tiên của đồ án, và
nó thấp hơn hẳn những gì các chỉ số proxy gợi ý. Cần nói rõ trong báo cáo thay vì chỉ khoe
`radical@k`.

### 10.3 Vì sao thấp: lệch phân phối, không phải model kém

Dấu hiệu đầu tiên là **hub effect** — top-1 lặp lại bất thường: 攅 xuất hiện 6 lần, 劕/湸/旦/刟 mỗi
chữ 3 lần trên 56 query khác nhau. Query nằm ngoài phân phối nên tất cả bị hút về vài vector corpus.

Đo trực tiếp mức lệch:

| | `test_images/` (scan) | `images/` (corpus) |
|---|---|---|
| tỉ lệ điểm ảnh có mực | **0,353** | 0,184 |
| chiều cao hộp bao / khung | 0,919 | 0,800 |
| chiều rộng hộp bao / khung | 0,927 | 0,743 |
| giá trị nền | 249 (xám, nhiễu scan) | 255 (trắng tuyệt đối) |
| kích thước | 88×96, 96×128, 124×112… **không vuông** | **70×70, luôn vuông** |

Nhưng nguyên nhân *lớn nhất* chỉ nhìn bằng mắt mới thấy: **query là chữ viết tay / hành thư, corpus
là khải thư in bằng font sạch.**

- `moojt.jpg` ("một") là 沒 viết tay nét cong mềm; 9 ứng viên trong corpus đều là khải thư ngay ngắn.
- `cofn.jpg` ("còn") là 群 viết tay; đáp án đúng có trong corpus, và Phần 2 xếp nó **hạng 2/5**.

Nghĩa là thông tin cần thiết *có* trong ảnh — mô hình phải bắc cầu giữa hai kiểu thư pháp, việc mà
Chinese-CLIP chưa bao giờ được dạy. Đây chính là giả thuyết "pretrain sai domain" ở mục 9.3, nhưng
lần này quan sát được trực tiếp thay vì suy ra từ đường cong scale.

### 10.4 Thử chuẩn hoá hình học: **không đủ**

`--normalize` cắt sát nét mực, đệm vào khung vuông, đưa chữ về đúng tỉ lệ chiếm khung như corpus:

| biến thể | hit@1 | hit@5 | hit@20 | MRR | số top-1 khác nhau |
|---|---|---|---|---|---|
| A. gốc | 0,0847 | 0,1525 | 0,2034 | 0,1143 | 41/59 |
| B. chuẩn hoá **query** | **0,1017** | 0,1525 | **0,2542** | **0,1286** | **48/59** |
| C. chuẩn hoá **cả hai phía** | 0,0847 | **0,1864** | 0,2203 | 0,1203 | 39/59 |

> **Đừng đọc bảng này như một chiến thắng.** Ở n = 59, chênh lệch hit@1 giữa A và B là **5 ảnh so
> với 6 ảnh** — đúng một ảnh, nằm gọn trong khoảng nhiễu; khoảng tin cậy 95% của hai tỉ lệ chồng lên
> nhau gần như hoàn toàn. (Ở n = 56 khoảng cách này là 4 so với 6 ảnh — thêm ba ảnh đã làm nó co lại,
> minh hoạ đúng vì sao không được đọc bảng này theo hướng có lợi.) Điều duy nhất kết luận được là
> **hub effect giảm rõ** (41 → 48 top-1 khác nhau ở biến thể B), tức chuẩn hoá có sửa được phần lệch
> *hình học*.

Phần còn lại — khoảng cách thư pháp — chuẩn hoá không chạm tới được. Vì vậy `--normalize` **không
bật mặc định**; nó được giữ lại vì là cách sửa đúng cho một nửa vấn đề và sẽ cần khi có tập test lớn
hơn để đo cho ra kết luận.

### 10.5 Phần 2 chạy được trên scan thật — đây mới là chỗ dùng được

Phần 2 chỉ xếp hạng trong nhóm cùng âm (trung vị ~7 ứng viên thay vì 26.044), nên bài toán dễ hơn
nhiều bậc và **kết quả dùng được ngay**. Ví dụ `--word còn --image test_images/cofn.jpg`:

| | top-1 | top-2 | top-3 |
|---|---|---|---|
| embedding | 存 0,8036 | **群 0,7994** ← đúng | 羣 0,7883 |
| histogram | 噲 0,9987 | 羣 0,9984 | 群 0,9954 |

Chú ý điểm số histogram: 0,9987 / 0,9984 / 0,9954 — **cách nhau 0,003 trên toàn nhóm**, gần như
không phân biệt được gì. Embedding trải từ 0,80 xuống 0,72. Trên toàn bộ 59 ảnh, hai phương pháp chỉ
chọn cùng top-1 ở **23/59 (39,0%)** — thấp hơn hẳn mức 6/10 đo trên ảnh corpus sạch ở mục 8, đúng
như dự đoán rằng scan nhiễu sẽ nới rộng khoảng cách.

Trước đây mục này chỉ có pseudo-label (4 ảnh mà Phần 1 trả top-1 đã đúng nhóm âm): embedding 4/4,
histogram 2/4. Cỡ mẫu đó quá nhỏ. **Nhãn người gán cho cả 59 ảnh giờ đã có — xem mục 10.6.**

### 10.6 Nhãn cho Phần 2 — đã gán, và đây là accuracy thật

Chấm điểm Phần 2 cho đúng cần biết ảnh scan đó **là ký tự cụ thể nào** — không suy ra được từ tên
file, vì một âm Quốc Ngữ ứng với nhiều chữ (trung vị 7, nhiều nhất 97 với "tư").

**Nguồn văn bản.** 59 ảnh trong `test_images/` là mười mấy câu mở đầu **Truyện Kiều**
("Trăm năm trong cõi người ta / Chữ tài chữ mệnh khéo là ghét nhau / ..."). Biết được điều này
quan trọng: ngoài so hình còn đối chiếu được với chính tả Nôm đã chuẩn hoá của văn bản.

**Quy trình gán nhãn.** Với mỗi ảnh, dựng phiếu (`--sheets`) gồm query khoanh đỏ cạnh **toàn bộ**
ứng viên đánh số, rồi:

1. So hình là chính — nhận diện bộ thủ và cấu trúc trái/phải, trên/dưới;
2. Đối chiếu với văn bản Kiều làm tiên nghiệm;
3. Khi (1) và (2) mâu thuẫn thì **theo (1)**, vì đang gán nhãn cho ảnh này chứ không cho một ấn bản.
   Có xảy ra thật: "qua" trong scan là **戈** chứ không phải 過 chuẩn; "năm" là **𢆥** chứ không
   phải 年; "một" là **没** chứ không phải 𠬠.
4. Chỉ khi hình không tách nổi mới lấy dạng chuẩn của Kiều, và đánh dấu độ tin thấp.

**Cột `confidence` nghĩa là gì, và tính từ đâu.** Không tính từ đâu cả — đây là **đánh giá của
người đọc ảnh, không suy ra từ bất kỳ điểm số nào của mô hình**, và điều đó là cố ý. Ngay dưới đây
có một phép kiểm dùng chính cột này để lọc ra tập nhãn chắc chắn rồi chấm lại mô hình trên đó; nếu
`confidence` lấy từ điểm similarity thì phép kiểm ấy thành vòng tròn — lấy mô hình chấm mô hình.
Thang (định nghĩa trong `CONFIDENCE_LEVELS`, `evaluate_test_images.py`):

| mức | tiêu chí | n |
|---|---|---|
| `high` | nét quyết định đọc được; mọi ứng viên còn lại khác ở **bộ thủ** hoặc ở **số nét thấy rõ** | 53 |
| `med` | chọn được một ứng viên, nhưng còn ≥1 ứng viên là **biến thể gần trùng** chỉ khác ở chi tiết đã mờ/mất trong scan (vd 𣘛 vs 橷, cùng 木+兜) | 4 |
| `low` | nét quyết định **không** đọc được; chọn dựa vào văn cảnh nhiều hơn hình | 2 |

Theo quy ước ở bước 3, khi hình và văn cảnh mâu thuẫn thì hình thắng **và** hạ `confidence` xuống
`med`. Vì thế `med`/`low` không phải "nhãn kém" mà là **nhãn có ứng viên cạnh tranh chưa loại được**
— đúng thứ cần đánh dấu để soát lại.

**Phiếu từng cắt mất đáp án (đã sửa).** `label_sheets()` trước đây mặc định `max_candidates=24`
trong khi cột `n_candidates` vẫn ghi tổng số thật, nên phiếu tự mâu thuẫn: `gia` ghi 50 ứng viên
nhưng chỉ in 24, mà đáp án đúng nằm ở vị trí **32**. Bốn ảnh dính lỗi này — `gia` (32/50),
`lujc` (35/40), `phong` (30/35), `tuw` (27/97) — nghĩa là **người gán nhãn không thể nhìn thấy đáp
án đúng** trong phiếu lẫn trong ảnh sheet. Nhãn hiện tại không sai (chúng được gán từ danh sách đầy
đủ, và hàm chấm điểm luôn dùng `sorted(gt)` không cắt: đã kiểm lại cả 59 dòng, `correct_index` khớp
`correct_unicode` 59/59), nhưng phiếu thì không dùng lại được. Nay `max_candidates=None` — in hết —
và mỗi ứng viên kèm codepoint (`32=笳(7B33)`) để không phải tra ngược từ glyph. Phiếu `tuw` giờ có
đủ 97 ô.

Ba ảnh thu hồi được ở mục 10.1 gán thêm sau: `nguwowsi` = 𠊛 (`2029B`, `high`), `phong_1` = 豐
(`8C50`, `high`), `dau` = 𣘛 (`2361B`, `med` — 𣘛 và 橷 6A77 là hai biến thể 木+兜 gần như trùng
hình, scan lại mờ). Đáng chú ý: `phong_1` đọc là "phong" nhưng là **豐**, trong khi `phong.jpg` là
**風** — hai chữ khác hẳn cùng một âm, đúng loại ca mà Phần 2 sinh ra để xử lý.

**Kết quả (n = 59, nhóm cùng âm, `chinese-clip-large`/fp16):**

| | top-1 đúng | tỉ lệ | KTC 95% | MRR |
|---|---:|---:|---|---:|
| embedding | 28/59 | **0,4746** | [0,343; 0,609] | 0,6566 |
| histogram | 26/59 | 0,4407 | [0,312; 0,576] | 0,6070 |
| chọn ngẫu nhiên trong nhóm | — | 0,1987 | — | — |

Cả hai hơn baseline ngẫu nhiên rõ rệt (2,4× và 2,2×) — Phần 2 **thật sự dùng được**, khác hẳn Phần 1
(hit@1 = 0,0847 ở mục 10.2). Cả ba ảnh mới đều bị **cả hai** phương pháp trả sai, nên chúng chỉ làm
tăng mẫu số.

**Nhưng khoảng cách embedding–histogram thì không kết luận được.** Hai khoảng tin cậy chồng nhau gần
hết: khoảng cách là **2 ảnh trên 59**. Phép thử độ nhạy (`--min-confidence high`, 53 ảnh) cho
0,5094 vs 0,4717 — vẫn đúng 2 ảnh, MRR 0,6808 vs 0,6402, tỉ lệ chọn cùng top-1 22/53 = 0,4151. Với
KTC 95% rộng ±13 điểm ở mỗi phía, chênh 2 ảnh **không phải bằng chứng** về bất cứ điều gì.

Kết luận trung thực: mục 8 nói "embedding thay được histogram" là **chưa được số liệu này chứng minh**.
Nó chứng minh được điều khác và cũng có giá trị — thu hẹp về nhóm cùng âm giúp cả hai vượt xa ngẫu
nhiên, tức nút thắt là ở tra cứu toàn corpus chứ không phải ở hàm chấm điểm.

#### 10.6.1 Soát lại 11 dòng `med`/`low`

Soát từng dòng một: phóng ảnh gốc lên 460–520px (ảnh scan chỉ 94×104 tới 178×162 px), cắt riêng
**nửa chứa nét quyết định** rồi đặt cạnh các ứng viên cạnh tranh, và đối chiếu với văn bản Kiều.

| ảnh | trước | sau | conf | nét quyết định |
|---|---|---|---|---|
| `chuwx` | 𡨸 | 𡨸 | med→**high** | nửa phải là 字: thấy rõ nét ngang của 子 xuyên hai bên, 守 (寸) chỉ có một chấm |
| `coix` | 揆 | **𡎝** | med | **sửa nhãn** — bộ trái là 土 (nét đáy nằm ngang, sổ dừng ở đó) chứ không phải 扌 (móc thõng xuống). Trùng luôn với chính tả Kiều "trong cõi người ta" |
| `dau` | 𣘛 | 𣘛 | med | 𣘛 vs 橷 chỉ khác ruột của 兜 — chỗ đó mất nét |
| `gia` | 笳 | 笳 | med→**low** | ảnh là **2 tầng**: đỉnh nhỏ + 加. Loại được 嘉 (3 tầng, có 口 ở giữa) nhưng **không tách nổi ⺮ (笳) với ⺾ (茄)**. Xem ghi chú dưới bảng |
| `giowr` | 𢷣 | 𢷣 | med→**high** | bộ trái 扌 rõ; các ứng viên còn lại là 口/足 |
| `lujc` | 錄 | 錄 | med→**high** | bộ trái đáy là **một nét ngang** (金), không phải ba chấm của 糸 (綠). Khớp nghĩa "cổ lục" = 古錄 |
| `moojt` | 没 | 没 | med | 没 vs 沒 khác nhau ở góc trên phải, mờ |
| `sawsc` | 嗇 | 嗇 | med→**high** | thấy hai nét chéo ở đỉnh (dạng 來), phân biệt được với 啬. Khớp "bỉ sắc tư phong" = 彼嗇斯豐 |
| `trari` | 𣦰 | 𣦰 | low | ruột của 厂 không đọc ra: 𣦰 vs 𣦆 vs 𣥱 vẫn ngang nhau. **Chưa giải quyết được** |
| `troong` | 篭 | 篭 | med→**high** | ảnh là cấu trúc **trên–dưới** (⺮ trên 龍); mọi ứng viên 目-bên-trái (𥊛, 矓) là trái–phải, loại hết |
| `xanh` | 撑 | 撑 | med | 撑 vs 撐 là hai biến thể gần trùng |

Kết quả: **53 `high`, 4 `med`, 2 `low`** (trước là 48/10/1).

**Ca `gia` đáng chú ý.** Văn cảnh rất mạnh: câu "Rằng năm **Gia Tĩnh** triều Minh" — niên hiệu
嘉靖, và ba chữ `tixnh`/`trieefu`/`minh` trong tập test xác nhận đúng câu đó. Nhưng ảnh **không phải
嘉**: 嘉 có ba tầng (士 / 口 / 加), ảnh chỉ có hai (một bộ thủ nhỏ trên 加). Nên đây là chỗ chữ trong
bản scan lệch khỏi tự dạng chuẩn — đúng loại ca mà quy ước "hình thắng" sinh ra để xử lý — nhưng
giữa 笳 và 茄 thì nét đỉnh đã mờ mất, không quyết được. Để `low` và ghi lại cả hai khả năng.
Tra ngoài (hvdic.thivien.net) còn cho thấy 笳 **không** nằm trong danh sách chữ Nôm của "gia" ở từ
điển đó, dù `QuocNgu_SinoNom_Dic.xlsx` của đề bài có liệt — một lý do nữa để không chốt dòng này.

**Nhãn sai đã sửa đổi kết quả.** `coix` từ 揆 sang 𡎝 làm embedding **mất** một ảnh và histogram
**được** một ảnh: 29/25 thành 28/26. Khoảng cách giữa hai phương pháp co từ 4 ảnh xuống **2 ảnh**.
Nói cách khác, một lỗi gán nhãn duy nhất đã nghiêng kết quả về phía embedding — bằng chứng trực tiếp
cho việc bảng trên không chịu nổi sức nặng của một kết luận.

**Hạn chế còn lại.** Nhãn vẫn do so hình mà ra, không phải do chuyên gia Hán-Nôm đọc. Sáu dòng
`med`/`low` còn lại (`coix`, `dau`, `gia`, `moojt`, `trari`, `xanh`) nên được người biết chữ Nôm
soát; riêng `trari` và `gia` thì cần **ảnh trang gốc** chứ soi chữ rời không ra. Lưu ý `test_images/`
chỉ có chữ rời, không kèm số câu, nên "đối chiếu văn cảnh" ở đây là suy từ tập từ vựng của mấy câu
đầu Kiều chứ không phải tra được đúng vị trí.

**59 là trần cứng của tập test này.** `test_images/` chỉ có 59 ảnh scan và không có nguồn scan nào
khác trong kho. Ở n = 59 quanh p ≈ 0,5, khoảng tin cậy 95% rộng **±13,3 điểm**; muốn xuống ±8 điểm
cần **n ≈ 150**, xuống ±6 điểm cần **n ≈ 300**. Nghĩa là mọi cải tiến dưới ~15 điểm — gần như chắc
chắn bao gồm cả finetune — **không thể chứng minh trên tập này**. Muốn đo được thì phải có thêm ảnh
scan thật, không phải thêm phương pháp.

Tái lập:

    .venv/bin/python evaluate_test_images.py --part 2 --labels output/label_sheets/label_template.csv
    .venv/bin/python evaluate_test_images.py --part 2 --labels output/label_sheets/label_template.csv \
        --min-confidence high      # phép kiểm độ nhạy ở trên, 53 ảnh

### 10.7 Ảnh hưởng tới kết luận ở mục 9

Mục 9 kết luận `radical@k` = 0,5361 tương đương 73% trần thực tế. Mục này cho thấy con số đó **đo
trên đúng phân phối mà model được thuận lợi nhất** (query là ảnh corpus sạch). Trên query thật, hiệu
năng sụt mạnh. Hai mục cùng chỉ về một hướng: **finetune là đòn bẩy thật**, và giờ đã rõ nên finetune
cái gì — không chỉ học thành phần IDS, mà học **bất biến với kiểu thư pháp**, bằng cách ghép cặp
scan viết tay ↔ bản in cùng một ký tự.

## 11. Hạn chế

- Ba chỉ số đều là **nhãn yếu**, không phải phán đoán của con người. Chúng đo "cùng bộ thủ / cùng số
  nét / cùng thành phần IDS", không đo trực tiếp "trông giống nhau".
- `radical@k` bỏ qua các ký tự thiếu `RADICAL`; `stroke_mae@k` bỏ qua ký tự thiếu `STROKE_NUM`. Mẫu
  hợp lệ của mỗi chỉ số vì thế hơi khác nhau.
- Số đo tốc độ nhiễu do dùng chung GPU (xem 4.2).
- 5.164/31.208 dòng bảng không có ảnh nên nằm ngoài toàn bộ đánh giá.
- Recall của ANN (mục 7) chỉ đo ở N = 26.044; các mốc 100k–1M chỉ đo độ trễ trên vector tổng hợp.
- Giải thích cho hiện tượng bão hoà khi scale model (mục 5.2) mới là giả thuyết; mục 9.3 gom được
  bằng chứng gián tiếp ủng hộ nó nhưng vẫn chưa phải kiểm chứng trực tiếp.
- **n = 59 là quá nhỏ, và đó là toàn bộ dữ liệu scan hiện có.** KTC 95% quanh p ≈ 0,5 rộng ±13,3
  điểm; muốn ±8 điểm cần n ≈ 150. Mọi so sánh ở mục 10.4 (chuẩn hoá hình học) đều nằm trong khoảng
  nhiễu. **Không có nguồn scan nào khác trong kho** — muốn tăng n phải scan thêm ảnh thật.
- Nhãn Phần 2 (mục 10.6) do **so hình** mà ra, không phải do chuyên gia Hán-Nôm đọc. Sau vòng soát
  lại ở mục 10.6.1 còn 6/59 dòng `med`/`low`, và vòng soát đó đã **sửa một nhãn sai** làm khoảng
  cách embedding–histogram co từ 4 ảnh xuống 2 ảnh. Chênh 2 ảnh trên 59 **không đủ để nói embedding
  thắng histogram**.
- Ảnh trong `test_images/` là chữ viết tay/hành thư, còn corpus là khải thư in — mọi số của mục 10
  đo trên đúng một cặp phân phối này, chưa chắc khái quát sang loại scan khác.
- Oracle ở mục 9.2 dựa trên `SHAPE_MORPH`, vốn chỉ có ở 96,9% ký tự và bản thân cũng là nhãn do
  người biên soạn — nó là *ước lượng* trần thực tế, không phải trần chính xác.
- Kết quả xếp hạng lại (mục 15) đo trên **render sạch**, nơi tầng 1 đã đạt 0,7467 nên còn rất ít
  chỗ cải thiện, và **chưa chạy với tầng 1 fine-tune** — tức chưa chạy ở cấu hình duy nhất mà nó có
  cơ hội thắng. Chi tiết ở 15.5.

## 12. Kết luận

1. **Dùng `chinese-clip-large` làm mặc định.** Hơn baseline 2,31× ở radical@k và 2,05× ở
   ids_jaccard, mà gần như không tốn thêm thời gian so với bản base (74 s vs 73 s cho full corpus,
   có cache nên chỉ trả một lần).
2. **Đừng scale tiếp.** `chinese-clip-huge` to hơn nhưng *kém hơn* ở radical@k. Đường cong đã quay
   đầu — đây là bằng chứng để không tốn công thử ViT-G.
3. **Giữ lại `dinov2`** — nó thắng ở số nét và chỉ trùng 12,9% kết quả với họ CLIP, nên là ứng viên
   tốt cho ensemble/rerank.
4. **Báo cáo kết quả âm của `resnet18-gray` một cách trung thực.** Nó bác bỏ giả thuyết `conv1` và
   chính là lý do chuyển sang ViT.
5. **Không thay Faiss `IndexFlatIP`.** Xem mục 7 — search chỉ chiếm 6% thời gian ở cỡ corpus hiện
   tại; điểm hoà vốn của ANN nằm quanh N ≈ 300.000.
6. **Phần 2 nên dùng `--method embedding`.** Với model zero-shot, lý do phải là *chất lượng điểm
   số* chứ không phải accuracy: histogram chấm điểm tuyệt đối cho chữ sai và chỉ đạt 62/64 ở phép
   thử tự truy hồi (mục 8), trong khi trên scan thật hai bên chỉ chênh **2 ảnh trên 59** (0,4746 vs
   0,4407, mục 10.6). **Sau khi finetune thì khác hẳn**: 0,8136 vs 0,4407, chênh 22 ảnh, hai khoảng
   tin cậy rời nhau (mục 14.2) — giờ accuracy đã đủ để tự đứng làm lý do.
7. **Trên ảnh query thật, hiệu năng thấp hơn nhiều so với các chỉ số proxy gợi ý** — hit@1 = 0,0847
   trên toàn corpus (179× ngẫu nhiên, nhưng tuyệt đối là thấp). Nguyên nhân chính là **khoảng cách
   thư pháp**: query là scan viết tay, corpus là khải thư in bằng font. Phải nói rõ điều này trong
   báo cáo (mục 10).
8. **Phần 2 mới là chỗ dùng được ngay.** Thu hẹp từ 26.044 xuống trung vị 7 ứng viên khiến bài toán
   dễ hơn hẳn; trên scan thật embedding và histogram chỉ đồng ý 39,0% — khoảng cách rộng ra đúng như
   dự đoán ở mục 8.
9. **`radical@k` = 0,5 không phải bão hoà.** Trần lý thuyết là 0,9862 nhưng trần *thực tế* — đo
   bằng oracle biết trước phân rã IDS — chỉ là 0,7338, vì `RADICAL` không phải nhãn thị giác thuần
   tuý. `chinese-clip-large` đang ở 73% quãng đường và 24,4× mức ngẫu nhiên (mục 9).
10. **Finetune đúng là đòn bẩy lớn nhất, và đã đo được.** `chinese-clip-ft` đạt hit@1 = 0,5932 trên
   scan thật, so với 0,0847 của `chinese-clip-large` zero-shot — hai CI rời hẳn nhau, và lớp chưa
   từng thấy lúc huấn luyện đạt đúng bằng lớp đã thấy (0,9939 vs 0,9954) nên không phải thuộc lòng.
   Chi tiết ở mục 14, kể cả lần ArcFace bị collapse trước đó.
11. **Xếp hạng lại bằng siêu dữ liệu: chưa dùng được, nhưng không phải vì ý tưởng sai.** Cấu hình
   oracle hơn tầng 1 tới +17,5 điểm, nên trần còn cao; cấu hình thật lại thấp hơn tầng 1 14/600 ảnh
   vì bước suy ra bộ thủ của query chỉ đúng 11,86% với tầng 1 zero-shot. Con số đó lên 0,6610 khi
   tầng 1 là mô hình fine-tune, mà cấu hình ấy **chưa được chạy** — đây là việc còn dở, không phải
   kết luận đã chốt (mục 15).
12. **Bước tiếp theo cần đánh giá bằng người.** Mức đồng thuận giữa các backend quá thấp để kết luận
   chỉ dựa trên proxy — và mục 9 cho thấy chính proxy cũng có trần riêng. Sau đó mới tới ensemble
   `chinese-clip-large` + `dinov2`, rồi finetune metric learning trên thành phần IDS.

---

## 13. Sinh cặp dương bằng render đa font (`render_fonts.py`)

### 13.1 Vì sao

Corpus có **đúng 1 ảnh cho 1 ký tự** (26.044 ảnh / 26.044 ký tự). Metric learning cần *cặp dương* —
hai ảnh mà mô hình phải học kéo lại gần nhau — nên hiện tại **không tồn tại cặp nào** để dạy mô hình
bỏ qua khác biệt kiểu chữ. Mỗi ký tự đã có mã Unicode, nên render lại qua nhiều font là cách rẻ nhất
để có cặp dương "cùng chữ, khác kiểu".

### 13.2 Bộ font và độ phủ

Máy dev không có font CJK nào (`fc-list :lang=zh` = 0); font tải rời qua `download_fonts.sh` vào
`fonts/` (đã gitignore, ~200 MB). Độ phủ đo bằng bảng `cmap` của từng font, **không đoán** — ký tự
không có trong `cmap` bị bỏ qua thay vì render ra ô tofu `.notdef` (một ô vuông rỗng giống hệt nhau
ở mọi ký tự, đưa vào huấn luyện sẽ đầu độc cặp dương).

| nhóm font | kiểu chữ | render được | % của 26.044 |
|---|---|---:|---:|
| `hanamin` (HanaMinA+B) | 花園明朝 minh thể, Ext-A..Ext-G | 26.019 | 99,9% |
| `babelstone` | BabelStone Han, tống thể | 23.009 | 88,3% |
| `iming` | 一點明體 minh thể truyền thống | 21.805 | 83,7% |
| `lxgw-kai` | 霞鶩文楷 **khải thư**, chất bút lông | 18.569 | 71,3% |
| `noto-serif` | Noto Serif TC | 15.590 | 59,9% |
| `noto-sans` | Noto Sans TC, hắc thể | 15.591 | 59,9% |
| `zhuque-fangsong` | 朱雀仿宋 | 11.021 | 42,3% |

**Hợp nhất: 26.033/26.044 = 100,0%** — chỉ 11 ký tự không font nào render được. Tức là *mọi* ký tự
đều có ≥1 bản render, cộng ảnh corpus gốc là ≥2 view → đủ tạo cặp dương. Trung bình **5,05 kiểu
chữ/ký tự**, tổng **131.604 ảnh render** + 26.044 ảnh corpus = 157.648 view trên 26.044 lớp.

Ảnh render đi qua cùng hàm chuẩn hoá hình học với `evaluate_test_images.py` (cắt sát nét mực → đệm
khung vuông → chữ chiếm 0,80 bề rộng), nên render và corpus cùng khung; mô hình không học nhầm "lề
rộng hẹp" thành đặc trưng phân biệt.

### 13.3 Tập này có chạm đúng chỗ hỏng không? (đo, không đoán)

Lấy 400 ảnh render mỗi font làm query, tìm trên index corpus 26k bằng `chinese-clip-large`/fp16 —
tức hỏi: *mô hình hiện tại có nhận ra bản render khác font là cùng một chữ không?*

| query | hit@1 | hit@10 |
|---|---:|---:|
| `babelstone` | 0,7800 | 0,9500 |
| `noto-serif` | 0,7450 | 0,9400 |
| `hanamin` | 0,7200 | 0,9350 |
| `iming` | 0,7025 | 0,9400 |
| `zhuque-fangsong` | 0,6800 | 0,9125 |
| `noto-sans` | 0,6500 | 0,8950 |
| **`lxgw-kai` (khải thư)** | **0,6350** | 0,8775 |
| **trung bình (n=2.800)** | **0,7018** | **0,9214** |

Thêm augmentation (đổi độ dày nét + méo đàn hồi, hàm `augment`): hit@1 **0,6732**, hit@10 0,8971.

**Đọc kết quả này cho đúng — đây là phần quan trọng nhất của mục 13:**

1. `lxgw-kai` là font **khó nhất** (0,6350), và nó cũng là font gần chất bút lông nhất. Thứ tự khó
   dễ đi đúng theo hướng "càng ngả về bút lông càng khó" → tập render *có* chứa đúng trục biến thiên
   mình cần.
2. Nhưng độ khó của nó **kém xa** thực tế: render khó nhất là 0,6350, còn **scan thật là 0,0847**
   (mục 10.2). Chênh gần 9 lần.
3. Augmentation hiện tại chỉ kéo 0,7018 → 0,6732 — **quá yếu**, chưa mô phỏng được scan thật.

Kết luận: **render đa font là điều kiện cần, chưa phải điều kiện đủ.** Phần lớn khoảng cách tới
scan thật không nằm ở *font*, mà ở *chất viết tay và nhiễu scan* (mực loang, nét dính, nền giấy 249
thay vì 255, tỉ lệ không vuông — xem mục 10.3). Huấn luyện chỉ trên tập này sẽ dạy được bất biến
kiểu chữ, nhưng chưa chắc dạy được bất biến scan.

### 13.4 Suy giảm ảnh cho giống scan (`scan_augment.py`)

Mục 13.3 kết luận khoảng cách tới scan thật **không nằm ở font**. Vậy nó nằm ở đâu? Đo trực tiếp
59 ảnh `test_images/` so với ảnh render:

| chỉ số | scan thật | render | lệch | nghĩa là |
|---|---:|---:|---:|---|
| `edge` (độ nét) | 15,6 | 102,2 | **6,55×** | scan **mờ** hơn nhiều — khoảng cách lớn nhất |
| `contrast` | 91,6 | 232,8 | **2,54×** | mực xám nhạt, không đen tuyền |
| `thick` (độ dày nét) | 0,051 | 0,028 | 0,55× | nét scan dày gần gấp đôi |
| `fill` (tỉ lệ mực) | 0,349 | 0,204 | 0,59× | hệ quả của nét dày |
| `noise` | 10,1 | 2,7 | 0,26× | hạt nhiễu giấy/cảm biến |
| `bg` (nền) | 254,0 | 255,0 | 1,00× | giấy hơi ngà |

`scan_augment.degrade()` mô phỏng đúng sáu trục đó theo thứ tự vật lý của một trang scan: méo đàn
hồi (tay viết, giấy vênh) → nở nét (mực loang) → hạ phân giải rồi phóng lại (mất nét khi chụp, cũng
là thứ làm hai nét sát nhau **dính** vào nhau) → nhoè Gauss (quang học) → giấy ngà + hạ tương phản
→ nhiễu cảm biến → co giãn phi vuông rồi ép lại về khung vuông. Không phép nào đụng tới cấu trúc bộ
thủ, nên nhãn (mã Unicode) vẫn đúng.

Khoảng tham số **hiệu chỉnh theo số đo**, không chỉnh mò — `python scan_augment.py --calibrate`:

| chỉ số | scan thật | render | đã suy giảm | lệch trước | lệch sau |
|---|---:|---:|---:|---:|---:|
| `bg` | 253,966 | 255,000 | 254,653 | 1,00× | 1,00× |
| `fill` | 0,349 | 0,204 | 0,321 | 0,59× | 0,92× |
| `thick` | 0,051 | 0,028 | 0,041 | 0,55× | 0,81× |
| `edge` | 15,606 | 102,210 | 11,950 | 6,55× | 0,77× |
| `noise` | 10,135 | 2,682 | 10,096 | 0,26× | 1,00× |
| `contrast` | 91,561 | 232,806 | 87,049 | 2,54× | 0,95× |

**Sai lệch log trung bình: 0,880 → 0,103, tốt hơn 8,5×.** Cả sáu chỉ số về trong khoảng 0,77–1,00×
so với scan thật.

### 13.5 Suy giảm có kéo được độ khó về đúng mức không?

Cùng phép đo ở mục 13.3 (400 ảnh/font, `chinese-clip-large`/fp16), thay đổi `--strength`:

| strength | hit@1 | hit@10 |
|---|---:|---:|
| 0,0 (render sạch) | 0,7018 | 0,9214 |
| 0,4 | 0,4179 | 0,6832 |
| 0,7 | 0,2825 | 0,5239 |
| **1,0** | **0,1986** | **0,4161** |
| *scan thật (mục 10.2)* | *0,0847* | *0,1864* |

Augmentation cũ (chỉ dày nét + méo đàn hồi) chỉ đạt 0,6732 — **dễ hơn scan thật 9,4 lần**. Bản mới
ở strength 1,0 đạt 0,1986, **dễ hơn 2,8 lần**. Tức đã lấp được phần lớn khoảng cách.

**Phần còn lại 2,8× là gì, và vì sao augmentation không lấp nốt được:** mọi phép ở `degrade()` đều
là *nhiễu ảnh* — mờ, mực, hạt, co giãn. Nhưng scan thật còn khác ở chỗ **chữ được viết bằng bút lông
theo lối hành thư**: nét nối liền, tỉ lệ bộ phận xê dịch, có nét bị lược. Đó là khác biệt **cấu
trúc**, không phải nhiễu — không có cách nào sinh ra từ một ảnh in bằng xử lý ảnh. Muốn lấp nốt phải
có chữ bút lông thật (MCCD, mục 13.6), hoặc fine-tune trên chính scan Hán-Nôm (NomNaOCR/IHR-NomDB).

Kết luận cho việc huấn luyện: dùng `render_fonts.py --augment N --strength 1.0`. Đó là dữ liệu khó
nhất sinh được miễn phí, và đã ở đúng bậc độ khó của bài toán thật.

### 13.6 Việc tiếp theo, theo thứ tự

1. **Tăng cường augmentation cho giống scan**: mờ, nhiễu, nền giấy, méo mạnh hơn, làm dính nét. Rẻ
   nhất, và mục 13.3 cho thấy bản hiện tại còn quá nhẹ.
2. **MCCD** — 329.715 ảnh ký tự thư pháp đã cắt rời, 7.765 lớp, gán nhãn 10 thể chữ (khải/hành/thảo/
   lệ/triện). Đây là bút lông *thật*, đúng chỗ mà render font không với tới.
3. **NomNaOCR / IHR-NomDB** — phủ 18% ký tự Ext-B (chữ Nôm riêng) mà không bộ Trung Quốc nào có.
4. **CASIA-HWDB** — 3,9 triệu mẫu, 7.356 lớp, 1.020 người viết. Giá trị chính là số người viết, nhưng
   chỉ phủ 26,0% corpus (GB2312) và là bút cứng giản thể hiện đại, sai trục phong cách. Ưu tiên thấp
   hơn MCCD.

## 14. Fine-tune (`finetune_glyph.py`) — kết quả lớn nhất của đồ án

Mục 9.4 kết luận finetune là đòn bẩy thật; mục 13 sinh xong dữ liệu. Đây là kết quả.

### 14.1 Cách làm

Học **bất biến kiểu chữ**: cùng một mã Unicode qua font khác nhau phải ra cùng một vector. Nhãn duy
nhất là **mã Unicode** — cố tình không đụng `RADICAL`/`STROKE_NUM`, vì mục 9.4 đã cảnh báo huấn luyện
trên nhãn nào rồi báo cáo chỉ số của chính nhãn đó là đo lại tập huấn luyện.

| | |
|---|---|
| backbone | Chinese-CLIP ViT-B/16, giữ nguyên `visual_projection` (512 chiều) |
| loss | SupCon (InfoNCE giám sát), nhiệt độ 0,07 |
| lấy mẫu | **P×K: 24 lớp × 2 view mỗi batch** |
| dữ liệu | 141.981 ảnh / 23.440 lớp (đã trừ held-out) |
| suy giảm | `scan_augment.degrade` **online** trong DataLoader, p=0,6, strength 1,0 |
| tối ưu | AdamW, lr 1e-5, warmup 300, cosine, bf16, gradient checkpointing |
| thời gian | ~4 phút/epoch, 6 epoch, đỉnh ở epoch 5 |
| VRAM | **3,7 GB** — vừa chỗ trống cạnh 4 engine vLLM đang chiếm 74/80 GB |

Vì sao ViT-B/16 chứ không phải large: large không vừa VRAM *lúc huấn luyện*. Nên trong bảng dưới,
mốc so công bằng nhất là zero-shot **của chính ViT-B/16** (hit@1 = 0,0169); mốc `large` (0,0847) là
mốc *cũ của đồ án*, và bản finetune vượt cả hai.

### 14.2 Kết quả trên scan thật (`test_images/`, n = 59)

Không một pixel nào của `test_images/` vào huấn luyện — `_assert_test_images_unseen` chặn cứng.

| model | hit@1 | hit@10 | MRR | CI95 của hit@1 |
|---|---:|---:|---:|---|
| `chinese-clip` zero-shot (ViT-B/16) | 0,0169 | 0,0678 | 0,0327 | — |
| `chinese-clip-large` zero-shot (mốc cũ, mục 10.2) | 0,0847 | 0,1864 | — | [0,028; 0,187] |
| **`chinese-clip-ft`** | **0,5932** | **0,7966** | **0,6691** | **[0,457; 0,719]** |

**Hai khoảng tin cậy rời hẳn nhau.** Đây là lần đầu trong đồ án n=59 đủ để kết luận — không phải vì
mẫu to ra, mà vì hiệu ứng đủ lớn (35/59 so với 5/59, hơn ngẫu nhiên 1.250×).

Phần 2 (xếp hạng trong nhóm cùng âm, nhãn người gán ở mục 10.6):

| phương pháp | đúng top-1 | MRR | CI95 |
|---|---:|---:|---|
| histogram | 26/59 = 0,4407 | 0,6070 | [0,312; 0,576] |
| embedding zero-shot (large) | 28/59 = 0,4746 | 0,6566 | [0,343; 0,609] |
| **embedding finetuned** | **48/59 = 0,8136** | **0,8983** | **[0,691; 0,903]** |

Độ nhạy `--min-confidence high` (53 nhãn chắc chắn): 45/53 = 0,8491 vs histogram 0,4717. Kết luận
không phụ thuộc vào 6 nhãn med/low.

**Điều này sửa lại kết luận 6 ở mục 12.** Trước đây khoảng cách embedding–histogram chỉ là *2 ảnh
trên 59*, nên phải khuyến nghị embedding dựa trên chất lượng điểm số chứ không dám viện accuracy.
Giờ khoảng cách là **22 ảnh**, hai CI rời nhau — accuracy đã đủ để tự đứng làm lý do.

### 14.3 Có phải chỉ thuộc lòng 23.440 danh tính không?

Phép đo: query = một bản render font, gallery = ảnh corpus, chạy riêng trên **lớp đã thấy** và **lớp
chưa thấy** (2.604 mã Unicode bị gỡ *toàn bộ view* khỏi huấn luyện). Hai mẫu cùng cỡ, cùng cách chọn
ngẫu nhiên — chọn theo thứ tự mã Unicode sẽ thiên về khối CJK phổ thông nên không so được.

| | lớp đã thấy | lớp chưa thấy |
|---|---:|---:|
| trước huấn luyện | 0,7938 | 0,8041 |
| sau huấn luyện | 0,9954 | **0,9939** |

**Bằng nhau.** Không có khoảng cách thuộc lòng. SupCon không có tâm lớp học được, nên không có gì để
nhớ — thứ nó học là phép so hai ảnh, và phép đó chuyển sang chữ chưa từng gặp nguyên vẹn.

### 14.4 ArcFace đã thử và **sập** — ghi lại để không ai làm lại

Thiết kế đầu là ArcFace trên 23.440 lớp (scale 32, margin 0,30, lr head 1e-3). Sau 1 epoch:

| | trước | sau |
|---|---:|---:|
| render→corpus, lớp đã thấy | 0,7938 | **0,0035** |
| render→corpus, lớp chưa thấy | 0,8041 | **0,0050** |
| scan thật hit@1 | 0,0169 | **0,0000** |
| loss | 20,5 | 19,4 (gần như đứng yên) |

Sụp ở **cả hai** phía nên không phải overfit — là **collapse**, mọi ảnh dồn về một vector.

Nguyên nhân là **hình dạng dữ liệu**, không phải hyperparameter: 23.440 lớp mà mỗi lớp chỉ ~6 ảnh.
Head ArcFace khởi tạo ngẫu nhiên với 23.440 tâm không kịp tổ chức, nên gradient nó đẩy về backbone
gần như là nhiễu. Adam chuẩn hoá theo độ lớn gradient, nên "lr nhỏ 1e-5" vẫn dịch chuyển trọng số đủ
để phá cấu trúc pretrain trong ~3.000 bước; `clip_grad_norm_` không cứu được vì nhiễu bị cắt chuẩn
vẫn là nhiễu.

SupCon không có tham số học được nào trong loss, nên không có head ngẫu nhiên bơm nhiễu vào backbone.
`--loss arcface` vẫn còn trong script để tái lập thất bại này, **không phải để dùng**.

### 14.5 Ba thứ thêm vào để phát hiện sự cố sớm

1. **`--eval-steps 600`** — eval giữa epoch. Collapse lộ ra sau ~1 phút thay vì sau cả epoch.
2. **Cột `cos`** — cosine trung bình giữa 2.000 vector corpus ngẫu nhiên. Collapse thì số này chạy về
   1,0. Không có nó thì "học hỏng" và "collapse" trông giống hệt nhau trên hit@1, mà hai thứ đó sửa
   theo hai cách khác nhau. Diễn biến thực tế: **0,890 → 0,000**, tức không gian giãn ra tới mức gần
   trực giao. Con số 0,890 lúc zero-shot tự nó giải thích vì sao khả năng phân biệt trước đây yếu.
3. **Chữ ký collapse của SupCon** là loss đứng đúng ở ln(P·K−1) = ln 47 = 3,85.

Ngoài ra `supcon_loss` có unit test tại chỗ (dương hoàn hảo → 0, collapse → ln 5, gradient hữu hạn),
viết sau khi dính lỗi `-inf × 0 = nan` làm loss ra nan ngay bước đầu.

### 14.6 Việc tiếp theo

Loss rơi 2,96 → 0,0068 **chỉ sau 400 bước**, và từ epoch 1 tới 6 hit@1 chỉ dao động 0,51–0,59 không
theo xu hướng nào. Tín hiệu huấn luyện đã cạn: 24 lớp lấy ngẫu nhiên từ 23.440 thì phân biệt quá dễ.

Đòn bẩy tiếp theo, theo thứ tự:

1. **Hard negative mining** — nhồi vào cùng batch những chữ *trông giống nhau* (láng giềng gần trong
   chính không gian vector hiện tại, làm mới mỗi epoch) thay vì 24 lớp ngẫu nhiên. Rẻ nhất, và là
   thứ duy nhất giải quyết đúng chỗ loss đang cạn.
2. **Backbone lớn hơn** khi GPU rảnh — ViT-B/16 là ràng buộc VRAM, không phải lựa chọn.
3. **MCCD / NomNaOCR** (mục 13.6) — render font vẫn không sinh ra được nét bút lông thật.

## 15. Xếp hạng lại bằng siêu dữ liệu (`rerank.py`) — kết quả âm, kèm trần của nó

Bảng `final_characteristics-v2.xlsx` còn ba cột mà toàn bộ pipeline ảnh không hề đụng tới:
`RADICAL`, `STROKE_NUM`, `SHAPE_MORPH`. Chúng mô tả đúng những thuộc tính mà mô hình ảnh không nhìn
thấy được. Câu hỏi: dùng chúng **xếp hạng lại** danh sách ứng viên của tầng ảnh thì có tốt hơn không?

Câu trả lời là **chưa** — nhưng lý do cụ thể hơn nhiều so với "không hiệu quả", và chính cái lý do
đó mới là phần dùng được.

### 15.1 Không được chấm bằng ba chỉ số cũ

`radical@k`, `stroke_mae@k`, `ids_jaccard@k` suy ra từ **đúng ba cột** mà tầng xếp hạng lại dùng làm
đặc trưng. Chấm tầng này bằng chúng là đo nó với chính đầu vào của nó: điểm sẽ leo về trần 0,978
(mục 9.1) mà không chứng minh được gì.

Đây không phải cảnh báo suông — nó được cài thành mã. `Reranker.circularity_report()` trả về đúng
những chỉ số bị vô hiệu bởi cấu hình hiện tại, và `evaluate_rerank.py --suite proxy` **cố tình chạy
phép đo vòng tròn đó** rồi dán hậu tố `[CIRCULAR]` vào từng cột, kèm một sheet `Warnings` trong file
Excel xuất ra. Suite đó là một minh hoạ, không bao giờ là một kết quả.

Phép đo trung thực phải dùng nhãn **độc lập** với ba cột kia. Ở đây là **mã Unicode** của chính ký
tự, trên tập ảnh render của mục 13 — `evaluate_rerank.py --suite render`.

### 15.2 Bốn quyết định thiết kế, mỗi cái sửa một lỗi

Ký hiệu B1–B4 dùng thống nhất trong `rerank.py`, `test_rerank.py` và `evaluate_rerank.py`.

| | vấn đề | cách xử lý |
|---|---|---|
| **B1** | chấm bằng chỉ số suy ra từ chính đặc trưng đầu vào | `circularity_report()` + suite `proxy` dán nhãn `[CIRCULAR]` |
| **B2** | trộn điểm thô biến xếp hạng thành sắp xếp từ điển | chuẩn hoá z-score **trong từng danh sách ứng viên** trước khi trộn |
| **B3** | ảnh query chưa biết là chữ gì thì không có siêu dữ liệu | `infer_query_meta()` bỏ phiếu từ láng giềng toàn corpus; truyền `None` giờ **ném lỗi** thay vì âm thầm thành no-op |
| **B4** | `load_corpus()` chỉ trả UNICODE/CHAR/path nên `Reranker(df)` chết vì `KeyError` | `Reranker.from_excel()` + kiểm cột, báo đúng phải dùng gì |

B2 đáng nói riêng vì nó là lỗi định lượng được: với trọng số mặc định, một ứng viên phải thắng về
cosine hơn **0,364** mới bù nổi một lần lệch bộ thủ — mà khoảng cách cosine trong top-20 không bao
giờ rộng đến thế. Trộn thô vì vậy **không phải** một phép trộn: nó là sắp xếp từ điển bộ thủ → số
nét → hình dạng, đẩy mô hình ảnh xuống làm tiêu chí phá hoà.

### 15.3 Kết quả trên tập render (n = 600, nhãn = mã Unicode)

```bash
S="--suite render --sample 600 --backend chinese-clip-large --dtype fp16"
.venv/bin/python evaluate_rerank.py $S --no-rerank        # tầng 1 đơn thuần
.venv/bin/python evaluate_rerank.py $S --oracle-meta      # trần (chẩn đoán, không phải kết quả)
.venv/bin/python evaluate_rerank.py $S                    # xếp hạng lại thật
.venv/bin/python evaluate_rerank.py $S --normalize none   # bản trước B2
```

| cấu hình | hit@1 | KTC 95% | so với tầng 1 |
|---|---:|---|---:|
| tầng 1, không xếp hạng lại | 448/600 = 0,7467 | [0,710; 0,780] | — |
| *xếp hạng lại, oracle* (**chẩn đoán**) | *553/600 = 0,9217* | *[0,897; 0,941]* | *+105 ảnh* |
| **xếp hạng lại, thật** | **434/600 = 0,7233** | **[0,686; 0,758]** | **−14 ảnh** |
| trộn điểm thô (trước B2) | 113/600 = 0,1883 | [0,159; 0,222] | −335 ảnh |

Ba điều rút ra, theo thứ tự quan trọng:

1. **Chuẩn hoá trước khi trộn là bắt buộc, không phải tinh chỉnh.** Bỏ nó mất **55,8 điểm** hit@1.
   Đây là kết quả mạnh nhất của cả mục này, và nó không nói gì về siêu dữ liệu — nó nói về cách trộn
   những tín hiệu khác thang đo.
2. **Siêu dữ liệu có trần cao.** Cấu hình oracle — cố tình đọc trộm siêu dữ liệu thật của query —
   được **+17,5 điểm**. Vấn đề không nằm ở ý tưởng. Nếu oracle mà không hơn tầng 1 thì đã kết luận
   được "siêu dữ liệu không giúp gì" và dừng; nó hơn, nên không dừng được.
3. **Nhưng trần đó chưa với tới.** Cấu hình thật *thấp hơn* tầng 1 đúng 14 ảnh — nằm trong khoảng
   nhiễu (hai KTC chồng nhau gần hết), nên đọc là **"chưa có tác dụng"**, không phải "làm hỏng".

Cấu hình oracle được vẽ rỗng ruột trong Hình 4 của báo cáo và ghi thẳng chữ *chẩn đoán* vì nó gian
lận có chủ ý. Nó là trần, không phải điểm.

### 15.4 Chỗ nghẽn, đã định lượng

Query là ảnh scan chưa biết là chữ gì nên không tra được siêu dữ liệu của nó. Cách thay thế là
`infer_query_meta()`: **bỏ phiếu từ láng giềng toàn corpus** (softmax theo cosine, nhiệt độ 0,05,
top-50), và bộ thủ nào thắng với biên dưới `min_margin=0,15` thì **trả về None** — thà bỏ tín hiệu
còn hơn xếp hạng theo một phỏng đoán.

Đo trực tiếp bước này (`--suite scans`):

| cách suy ra bộ thủ của query | độ chính xác |
|---|---:|
| đoán đều tay (1/214) | 0,0047 |
| luôn đoán bộ phổ biến nhất | 0,0489 |
| **bỏ phiếu, tầng 1 zero-shot** | **0,1186** |
| **bỏ phiếu, tầng 1 fine-tune** | **0,6610** |

Với tầng 1 zero-shot, phép bỏ phiếu **sai 88% số lần** — hơn đoán mò 2,4× nhưng vẫn là nhiễu, trong
khi bộ thủ chiếm trọng số 0,20 của điểm tổng. Tầng xếp hạng lại đang được nuôi bằng nhiễu.

Nguyên nhân có tính cơ chế chứ không phải chuyện tinh chỉnh: phiếu lấy từ láng giềng của tầng 1, mà
tầng 1 zero-shot chỉ đúng 8,47% trên scan thật (mục 10.2). Đa số phiếu là của chữ sai. **Đổi tầng 1
sang mô hình fine-tune thì độ chính xác nhảy 5,6× lên 0,6610.** Mắt xích yếu nằm ở tầng đứng trước,
không ở thiết kế của phép suy luận.

### 15.5 Ba hạn chế của chính phép đo này

Cần nói rõ, vì mục này là một kết quả âm, và một kết quả âm chỉ dùng được khi biết nó đo trong điều
kiện nào.

- **Bảng 15.3 đo trên render SẠCH, không suy giảm.** Lệnh đã chạy (ghi trong `make_figures.py`)
  không có `--augment`, và hit@1 tầng 1 là 0,7467 — khớp mốc render sạch 0,7018 của mục 13.3, chứ
  không phải mốc suy giảm 0,1986. Nghĩa là tầng 1 ở đây **đã rất mạnh sẵn**, còn rất ít chỗ cho tầng
  hai cải thiện. Docstring của `evaluate_rerank.py` khuyên chạy cả bốn cấu hình kèm `--augment`;
  **điều đó chưa được làm**, và đó mới là phép đo có ý nghĩa hơn.
- **Hai bảng ở trên đến từ hai suite khác nhau.** 15.3 là `--suite render` (ảnh render sạch), 15.4
  là `--suite scans` (59 scan thật). Nên chuỗi lập luận "xếp hạng lại thất bại *vì* bỏ phiếu chỉ
  đúng 11,86%" đang nối một thất bại đo trên render với một chẩn đoán đo trên scan. `--suite render`
  **có in** độ chính xác bỏ phiếu của chính nó, nhưng con số đó chưa được ghi lại vào
  `make_figures.py`. Ghi lại là việc rẻ và sẽ khép kín được lập luận.
- **Chưa chạy trên cấu hình tốt nhất.** Toàn bộ bảng 15.3 dùng tầng 1 `chinese-clip-large` zero-shot.
  Với `chinese-clip-ft`, chỗ nghẽn ở 15.4 gần như biến mất (0,1186 → 0,6610), nên câu hỏi "xếp hạng
  lại có giúp không" **vẫn chưa được trả lời ở nơi nó có cơ hội trả lời có**.

### 15.6 Việc tiếp theo, theo thứ tự

1. **Chạy `--suite scans --backend chinese-clip-ft`.** Rẻ nhất, code đã sẵn, và là thí nghiệm duy
   nhất có thể lật kết luận của mục này. `--suite scans` không dính nhiễm dữ liệu như `--suite
   render` (59 scan thật không nằm trong tập huấn luyện), nên đây là chỗ đo hợp lệ duy nhất cho mô
   hình đã fine-tune.
2. **Chạy lại bảng 15.3 kèm `--augment`**, để tầng 1 không còn ở mức dễ 0,75.
3. **Ghi độ chính xác bỏ phiếu của `--suite render` vào `MEASUREMENTS`**, khép kín lập luận ở 15.5.
4. Chỉ sau ba việc trên mới nên đụng tới trọng số. Tinh chỉnh trọng số khi tín hiệu bộ thủ còn sai
   88% là tinh chỉnh nhiễu — và `evaluate_rerank.py` đã tự in cảnh báo đúng điều đó khi độ chính xác
   dưới 0,40.

**Không được tinh chỉnh trọng số theo `radical@k`.** Đó chính là vòng tròn ở 15.1. Chỉ tinh chỉnh
theo `--suite render` hoặc `--suite scans`.

## Phụ lục: file kết quả

| file | nội dung |
|---|---|
| `output/benchmark_summary.xlsx` | 3 sheet: `Metrics`, `Agreement`, `Samples` |
| `output/benchmark_search.xlsx` | so exact vs ANN: `All`, `Flat scaling` (mục 7) |
| `output/contact_sheet_<backend>.png` | query (khung đỏ) + 5 lân cận đầu, 12 ký tự mẫu |
| `output/output_top_k_similar_char_<backend>.xlsx` / `.csv` | top-20 đầy đủ cho toàn corpus |
| `output/embeddings_<backend>.npz` | cache embedding, dùng lại cho các lần chạy sau |
| `output/label_sheets/*.png` + `label_template.csv` | phiếu gán nhãn Phần 2 (mục 10.6) |
| `output/rendered/<font>/<UNICODE>.png` + `manifest.csv` | 131.604 ảnh render đa font, nhãn = mã Unicode (mục 13) |
| `output/finetune/best.pt` + `train.log` | checkpoint fine-tune và log đầy đủ mọi lần eval (mục 14) |
| `output/output_top_k_similar_char_<tag>_reranked_<normalize>.xlsx` / `.csv` | top-20 sau khi xếp hạng lại, kèm cả điểm ảnh gốc (mục 15) |
| `output/rerank_proxy_<tag>_<normalize>.xlsx` | phép đo **vòng tròn có chủ ý**: 2 sheet `Circular` + `Warnings`. Không trích vào báo cáo (mục 15.1) |
| `output/embeddings_render_<tag>_<clean\|aug>_<n>.npz` | cache embedding của ảnh render dùng làm query ở `--suite render` |

"""Fine-tune tháp ảnh Chinese-CLIP thành mô hình *nhận dạng glyph*, bằng cặp dương render đa font.

Vì sao
------
BENCHMARK.md mục 5.2 đo được chữ ký của "pretrain sai domain": nâng từ `large` lên `huge` làm kết
quả **giảm** 4,0%. Thêm dung lượng cho một mục tiêu huấn luyện sai thì không giúp gì. Mục 9.4 kết
luận finetune mới là đòn bẩy thật, và mục 13 đã sinh xong dữ liệu: 131.604 ảnh render qua 7 font,
nhãn = mã Unicode, phủ 100% corpus.

Bài toán học ở đây là: **cùng một mã Unicode qua kiểu chữ khác nhau phải ra cùng một vector.** Đó
đúng là thứ zero-shot đang thiếu — mục 10.3 cho thấy query viết tay/hành thư vs corpus in khải thư
làm hit@1 rớt xuống 0,0847.

Cái bẫy ở mục 9.4, và cách né
------------------------------
Mục 9.4 cảnh báo: finetune trên nhãn `RADICAL` rồi báo cáo `radical@k` là **đo lại tập huấn luyện**.
Script này né bằng ba việc, và cả ba đều bắt buộc chứ không phải tuỳ chọn:

1. **Không huấn luyện trên `RADICAL`/`STROKE_NUM` gì hết.** Nhãn duy nhất là mã Unicode — thứ đã có
   sẵn, không phải chú giải của con người, nên không có gì để rò rỉ sang `radical@k`.
2. **Chỉ số chính là scan thật** (`test_images/`, 59 ảnh, nhãn người gán ở mục 10.6). Không một pixel
   nào của `test_images/` đi vào huấn luyện — kiểm bằng `assert` ở `_assert_test_images_unseen`.
3. **Chia held-out theo LỚP**: `--holdout` phần trăm mã Unicode bị gỡ *toàn bộ view* khỏi tập
   huấn luyện. Đo trên đó trả lời câu "học được bất biến kiểu chữ" hay chỉ "thuộc lòng 26k ảnh".
   Hai con số được in cạnh nhau; nếu seen ≫ unseen thì là thuộc lòng, phải nói ra.

Suy giảm ảnh: làm ngay lúc nạp, không render sẵn ra đĩa
-------------------------------------------------------
`scan_augment.degrade()` đã hiệu chỉnh theo số đo (mục 13.4). Gọi nó trong DataLoader worker thì mỗi
epoch mỗi ảnh ra một bản khác nhau — nhiều biến thể hơn hẳn `render_fonts.py --augment N`, mà không
tốn thêm GB nào trên đĩa. 16 CPU đủ nuôi ViT-B/16.

ArcFace đã thử và **sập** — giữ lại để không ai làm lại
--------------------------------------------------------
Lần chạy đầu dùng ArcFace trên 23.440 lớp (scale 32, margin 0,30, lr head 1e-3). Kết quả sau 1 epoch:

    render->corpus hit@1   lớp đã thấy 0,7938 -> 0,0035    lớp chưa thấy 0,8041 -> 0,0050
    scan thật hit@1        0,0169 -> 0,0000                loss 20,5 -> 19,4 (gần như đứng yên)

Sụp ở CẢ hai phía nên không phải overfit — là **collapse**: mọi ảnh dồn về một vector. Nguyên nhân là
hình dạng dữ liệu, không phải hyperparameter: 23.440 lớp mà mỗi lớp chỉ ~6 ảnh. Head ArcFace khởi
tạo ngẫu nhiên với 23.440 tâm không bao giờ kịp tổ chức, nên gradient nó đẩy về backbone gần như là
nhiễu — và Adam chuẩn hoá theo độ lớn gradient, nên "lr nhỏ" (1e-5) vẫn dịch chuyển trọng số đủ để
phá cấu trúc pretrain trong ~3.000 bước. `clip_grad_norm_` không cứu được, vì nhiễu vẫn là nhiễu sau
khi cắt chuẩn.

Nên mặc định giờ là **SupCon (InfoNCE giám sát) với bộ lấy mẫu P×K**: mỗi batch gồm P mã Unicode,
mỗi mã K view. Loss chỉ so các ảnh TRONG batch với nhau — không có tâm lớp nào để học, nên không có
gì để sập. Nó cũng khớp bài toán hơn: lúc chạy thật, query là chữ **ngoài** tập lớp huấn luyện, mà
SupCon tối ưu trực tiếp cosine giữa các ảnh chứ không qua bộ phân lớp đóng.

Lấy mẫu ngẫu nhiên thường **không dùng được** cho SupCon ở đây: batch 48 rút từ 23.440 lớp thì xác
suất có hai ảnh cùng lớp ~4%, gần như mọi batch không có cặp dương nào. Bắt buộc phải P×K.
`--loss arcface` vẫn còn để tái lập thất bại trên, đừng đặt làm mặc định.

Ngân sách VRAM
--------------
Máy này có 4 engine vLLM đang chiếm ~74/80 GB. Còn ~6,5 GB, nên:
  * backbone là **ViT-B/16** (`chinese-clip`), không phải large/huge — large không vừa lúc huấn luyện;
  * bật `gradient_checkpointing`, đổi tính lại activation lấy bộ nhớ;
  * bf16 autocast (H100), không cần GradScaler.
Đo thực tế: ~4,3 GB ở `--batch-size 48`. Nếu vLLM nhả bớt thì tăng batch, đừng đổi backbone rồi mới
so bảng — mốc zero-shot trong BENCHMARK.md là large, muốn so phải so cùng cỡ (xem `--compare`).

Ví dụ
-----
    python finetune_glyph.py --smoke                  # 200 lớp, 1 epoch, kiểm đường ống
    python finetune_glyph.py --epochs 6 --device cuda
    python finetune_glyph.py --eval-only --ckpt output/finetune/best.pt
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from evaluate_test_images import decode_filenames, ground_truth
from scan_augment import degrade
from search_all_chars_in_corpus import CHAR_TABLE, set_deterministic
from search_use_QuocNgu_mapping import QUOCNGU_TABLE

MODEL_ID = "OFA-Sys/chinese-clip-vit-base-patch16"
DEFAULT_CKPT = "output/finetune/best.pt"


# --- dữ liệu --------------------------------------------------------------------------------

class GlyphDataset(Dataset):
    """Một dòng manifest -> (tensor ảnh, chỉ số lớp).

    `degrade_prob` là xác suất áp `scan_augment.degrade` cho ảnh **render**. Ảnh `corpus` cũng là
    bản render sạch (mục 13.1) nên cũng suy giảm được; chỉ ảnh scan thật là không, mà scan thật
    không bao giờ có mặt ở đây.
    """

    def __init__(self, rows, class_of, mean, std, size=224, degrade_prob=0.0, strength=1.0, seed=0):
        self.paths = rows["path"].tolist()
        self.labels = [class_of[u] for u in rows["unicode"]]
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.size, self.degrade_prob, self.strength, self.seed = size, degrade_prob, strength, seed
        self.epoch = 0

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert("L")
        if image.size != (self.size, self.size):
            image = image.resize((self.size, self.size), Image.BICUBIC)
        if self.degrade_prob:
            # Seed phụ thuộc (epoch, index) => mỗi epoch một biến thể khác, mà vẫn tái lập được.
            rng = np.random.default_rng((self.seed, self.epoch, index))
            if rng.random() < self.degrade_prob:
                image = degrade(image, rng, strength=self.strength)
        array = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)
        tensor = array.unsqueeze(0).repeat(3, 1, 1)          # xám -> 3 kênh, đúng cái tháp ViT chờ
        return (tensor - self.mean) / self.std, self.labels[index]


class PKSampler(torch.utils.data.Sampler):
    """Mỗi batch = P lớp x K view. Đây là điều kiện SỐNG CÒN của SupCon, không phải tinh chỉnh.

    Rút ngẫu nhiên 48 ảnh từ 23.440 lớp thì kỳ vọng số cặp dương trong batch là ~0,05 — gần như mọi
    batch không có gì để kéo lại gần nhau, loss thành hằng số. P x K bảo đảm mỗi ảnh luôn có đúng
    K-1 ảnh cùng lớp làm dương.
    """

    def __init__(self, labels, batch_size, k=2, seed=0):
        self.k = k
        self.p = max(2, batch_size // k)
        self.rng = np.random.default_rng(seed)
        by_class = {}
        for index, label in enumerate(labels):
            by_class.setdefault(label, []).append(index)
        # Lớp chỉ có 1 ảnh không tạo được cặp dương -> loại khỏi vai trò neo. Trong corpus này chỉ
        # vài chục lớp như vậy (ký tự không font nào phủ), bỏ đi không ảnh hưởng phân phối.
        self.by_class = {c: v for c, v in by_class.items() if len(v) >= 2}
        self.classes = np.array(sorted(self.by_class))
        self.batches = sum(len(v) for v in self.by_class.values()) // (self.p * self.k)

    def __len__(self):
        return self.batches

    def __iter__(self):
        for _ in range(self.batches):
            batch = []
            for cls in self.rng.choice(self.classes, self.p, replace=False):
                pool = self.by_class[cls]
                batch += list(self.rng.choice(pool, self.k, replace=len(pool) < self.k))
            yield [int(i) for i in batch]


def supcon_loss(embeddings, labels, temperature=0.07):
    """InfoNCE giám sát: mọi ảnh cùng mã Unicode trong batch là dương, còn lại là âm.

    Không có tham số học được nào -> không có head ngẫu nhiên bơm nhiễu vào backbone, đó chính là
    thứ đã làm ArcFace sập ở trên.
    """
    z = F.normalize(embeddings, dim=1)
    sim = z @ z.T / temperature
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, float("-inf"))
    positives = (labels[:, None] == labels[None, :]) & ~eye
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    # Đường chéo là -inf; -inf * 0 = nan trong IEEE, nên phải ZERO nó ra trước khi nhân mặt nạ,
    # không thì loss ra nan ngay bước đầu (đã dính đúng lỗi này một lần).
    log_prob = log_prob.masked_fill(eye, 0.0)
    counts = positives.sum(1)
    per_anchor = -(log_prob * positives).sum(1) / counts.clamp(min=1)
    return per_anchor[counts > 0].mean()


def split_classes(manifest, holdout, seed):
    """Gỡ `holdout` phần trăm mã Unicode ra khỏi huấn luyện — gỡ **mọi view** của lớp đó.

    Chia theo lớp chứ không theo ảnh. Chia theo ảnh thì lớp nào cũng có mặt ở cả hai bên, và phép đo
    "nhận ra chữ chưa từng thấy" mất sạch ý nghĩa.
    """
    classes = np.array(sorted(manifest["unicode"].unique()))
    rng = np.random.default_rng(seed)
    held = set(classes[rng.permutation(len(classes))[: int(len(classes) * holdout)]].tolist())
    is_held = manifest["unicode"].isin(held)
    return manifest[~is_held].reset_index(drop=True), manifest[is_held].reset_index(drop=True), held


def _assert_test_images_unseen(manifest, test_dir):
    """Chặn rò rỉ: không dòng manifest nào được trỏ vào `test_images/`.

    Rẻ, và là thứ duy nhất bảo đảm con số scan thật còn nghĩa. Nếu ai đó lỡ thêm scan vào tập
    huấn luyện thì toàn bộ kết luận của mục 10 vô giá trị mà bảng số vẫn trông đẹp.
    """
    real = os.path.realpath(test_dir)
    bad = [p for p in manifest["path"] if os.path.realpath(p).startswith(real)]
    if bad:
        raise SystemExit(f"RÒ RỈ: {len(bad)} dòng manifest nằm trong {test_dir} (vd {bad[0]})")


# --- mô hình --------------------------------------------------------------------------------

class ArcFace(nn.Module):
    """Softmax có biên góc. Ép khoảng cách giữa các lớp, thay vì chỉ cần phân đúng.

    Dùng ArcFace chứ không phải cross-entropy thường vì thứ cần ở đây là **không gian vector để truy
    hồi**, không phải bộ phân lớp: lúc chạy thật, query là chữ nằm ngoài 26.044 lớp cũng phải xếp
    hạng được. Biên góc tối ưu trực tiếp cho cosine — đúng phép đo mà Faiss `IndexFlatIP` dùng.
    """

    def __init__(self, dim, n_classes, scale=32.0, margin=0.30):
        super().__init__()
        self.weight = nn.Parameter(F.normalize(torch.randn(n_classes, dim), dim=1))
        self.scale, self.margin = scale, margin

    def forward(self, embeddings, labels):
        cosine = F.normalize(embeddings, dim=1) @ F.normalize(self.weight, dim=1).T
        theta = torch.acos(cosine.float().clamp(-1 + 1e-7, 1 - 1e-7))
        target = torch.zeros_like(cosine, dtype=torch.bool)
        target[torch.arange(len(labels), device=labels.device), labels] = True
        logits = torch.cos(theta + self.margin * target) * self.scale
        return F.cross_entropy(logits, labels)


def build_model(device, checkpointing=True):
    """Tháp ảnh Chinese-CLIP ViT-B/16 + `visual_projection` (512 chiều), bỏ hẳn tháp văn bản.

    Giữ nguyên `visual_projection` thay vì gắn head mới: như vậy checkpoint nạp thẳng được vào
    `feature_extractors.ChineseClipExtractor`, mọi script cũ chạy y nguyên, không phải viết lại
    đường ống nhúng.
    """
    from transformers import AutoImageProcessor, ChineseCLIPModel

    processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=False)
    model = ChineseCLIPModel.from_pretrained(MODEL_ID)
    if checkpointing:
        # Phải bật trên model gốc (API của PreTrainedModel), và bật TRƯỚC khi xoá tháp văn bản —
        # hàm này duyệt toàn bộ submodule, gọi sau khi xoá thì vướng thuộc tính không còn.
        model.gradient_checkpointing_enable()
    del model.text_model, model.text_projection      # RoBERTa không dùng tới, ~400 MB VRAM
    return model.to(device), processor.image_mean, processor.image_std


def embed(model, paths, device, mean, std, batch_size=64, size=224):
    """Nhúng một danh sách đường dẫn. Cùng tiền xử lý với lúc huấn luyện, không suy giảm."""
    data = GlyphDataset(pd.DataFrame({"path": paths, "unicode": ["?"] * len(paths)}),
                        {"?": 0}, mean, std, size=size)
    loader = DataLoader(data, batch_size=batch_size, num_workers=8, pin_memory=True)
    out = []
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for images, _ in loader:
            features = model.get_image_features(pixel_values=images.to(device, non_blocking=True))
            if not torch.is_tensor(features):
                features = features.pooler_output
            out.append(F.normalize(features.float(), dim=1).cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, 512), dtype=np.float32)


# --- đánh giá -------------------------------------------------------------------------------

def real_scan_eval(model, device, mean, std, images_dir, test_dir, batch_size):
    """Chỉ số CHÍNH: 59 scan thật tìm trên toàn bộ 26.044 ký tự corpus.

    So thẳng được với BENCHMARK.md mục 10.2 (`chinese-clip-large` zero-shot: hit@1 0,0847,
    hit@10 0,1864). Nhãn là tập ký tự đọc đúng ra từ tên file Telex, không phải proxy.
    """
    quocngu, chars = pd.read_excel(QUOCNGU_TABLE), pd.read_excel(CHAR_TABLE)
    corpus_files = sorted(f for f in os.listdir(images_dir) if f.endswith(".jpg"))
    unicodes = np.array([os.path.splitext(f)[0] for f in corpus_files])
    have_image = set(unicodes.tolist())

    matched, _ = decode_filenames(test_dir, quocngu)
    labelled = [(p, s, w, gt) for p, s, w in matched
                if (gt := ground_truth(quocngu, chars, w, have_image))]

    gallery = embed(model, [os.path.join(images_dir, f) for f in corpus_files],
                    device, mean, std, batch_size)
    queries = embed(model, [p for p, _, _, _ in labelled], device, mean, std, batch_size)
    order = np.argsort(-(queries @ gallery.T), axis=1)[:, :20]

    ranks = []
    for i, (_, _, _, gt) in enumerate(labelled):
        hits = [r for r, u in enumerate(unicodes[order[i]]) if u in gt]
        ranks.append(hits[0] + 1 if hits else None)
    # Cosine trung bình giữa 2.000 vector corpus lấy ngẫu nhiên. Đây là máy dò collapse: khi mô hình
    # dồn mọi ảnh về một điểm, hit@1 tụt VÀ số này chạy về 1,0. Không có nó thì nhìn hit@1 tụt sẽ
    # không phân biệt được "học hỏng" với "collapse", mà hai thứ đó sửa theo hai cách khác nhau.
    sample = gallery[np.random.default_rng(0).permutation(len(gallery))[:2000]]
    cos = sample @ sample.T
    spread = float((cos.sum() - np.trace(cos)) / (len(sample) * (len(sample) - 1)))

    n = len(ranks)
    return {"n": n, "cos": spread,
            "hit@1": sum(r == 1 for r in ranks) / n,
            "hit@10": sum(r is not None and r <= 10 for r in ranks) / n,
            "mrr": sum(1.0 / r for r in ranks if r) / n}


def style_transfer_eval(model, device, mean, std, rows, images_dir, batch_size, tag):
    """Query = một bản render font (đã suy giảm), gallery = ảnh corpus. Đo bất biến kiểu chữ.

    Chạy hai lần — trên lớp ĐÃ thấy và lớp CHƯA thấy lúc huấn luyện. Chênh lệch giữa hai con số
    chính là phần "thuộc lòng"; nếu unseen sụp thì mô hình không tổng quát hoá, và phải báo cáo
    đúng như vậy chứ không chỉ khoe con số seen.
    """
    font_rows = rows[rows["view"] != "corpus"]
    if font_rows.empty:
        return None
    picked = font_rows.groupby("unicode", sort=True).head(1)
    unicodes = picked["unicode"].tolist()
    gallery_paths = [os.path.join(images_dir, f"{u}.jpg") for u in unicodes]
    keep = [i for i, p in enumerate(gallery_paths) if os.path.exists(p)]
    unicodes = [unicodes[i] for i in keep]

    gallery = embed(model, [gallery_paths[i] for i in keep], device, mean, std, batch_size)
    queries = embed(model, [picked["path"].tolist()[i] for i in keep], device, mean, std, batch_size)
    top = np.argmax(queries @ gallery.T, axis=1)
    return {"tag": tag, "n": len(unicodes),
            "hit@1": float(np.mean(top == np.arange(len(unicodes))))}


# --- vòng huấn luyện ------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default="output/rendered/manifest.csv")
    parser.add_argument("--images", default="images")
    parser.add_argument("--test-images", default="test_images")
    parser.add_argument("--out-dir", default="output/finetune")
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=48,
                        help="48 vừa ~6,5 GB VRAM còn trống bên cạnh vLLM; tăng nếu GPU rảnh")
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--holdout", type=float, default=0.10,
                        help="tỉ lệ mã Unicode gỡ HẲN khỏi huấn luyện (chia theo lớp, xem docstring)")
    parser.add_argument("--degrade-prob", type=float, default=0.6)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--loss", default="supcon", choices=("supcon", "arcface"),
                        help="supcon = InfoNCE giám sát + lấy mẫu P×K (mặc định). arcface giữ lại "
                             "chỉ để tái lập lần sập đã ghi ở docstring, đừng dùng để chạy thật.")
    parser.add_argument("--views-per-class", type=int, default=2, help="K trong lấy mẫu P×K")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--margin", type=float, default=0.30)
    parser.add_argument("--scale", type=float, default=32.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--eval-steps", type=int, default=600,
                        help="eval giữa epoch sau mỗi N bước. Collapse lộ ra trong vài phút thay vì "
                             "sau cả epoch — bài học từ lần chạy ArcFace.")
    parser.add_argument("--smoke", action="store_true",
                        help="200 lớp, 1 epoch — chỉ để kiểm đường ống chạy thông")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    set_deterministic(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    manifest = pd.read_csv(args.manifest, dtype={"unicode": str})
    _assert_test_images_unseen(manifest, args.test_images)
    if args.smoke:
        keep = sorted(manifest["unicode"].unique())[:200]
        manifest = manifest[manifest["unicode"].isin(keep)].reset_index(drop=True)
        args.epochs, args.eval_every = 1, 1

    train_rows, held_rows, held = split_classes(manifest, args.holdout, args.seed)
    class_of = {u: i for i, u in enumerate(sorted(train_rows["unicode"].unique()))}
    print(f"Huấn luyện: {len(train_rows)} ảnh / {len(class_of)} lớp | "
          f"held-out: {len(held_rows)} ảnh / {len(held)} lớp")

    # Mẫu "lớp đã thấy" phải BẰNG CỠ và CÙNG CÁCH CHỌN với held-out, nếu không hai con số không so
    # được: gallery nhỏ hơn thì hit@1 tự cao hơn, và lấy theo thứ tự mã Unicode thì thiên về khối
    # CJK cơ bản (chữ phổ thông, nhiều nét quen) trong khi held-out rải đều cả Ext-B.
    seen_pool = np.array(sorted(class_of))
    seen_pick = set(seen_pool[np.random.default_rng(args.seed + 1)
                              .permutation(len(seen_pool))[: len(held) or 1]].tolist())
    seen_rows = train_rows[train_rows["unicode"].isin(seen_pick)].reset_index(drop=True)

    model, mean, std = build_model(device, checkpointing=not args.eval_only)
    if args.eval_only or os.path.exists(args.ckpt) and args.eval_only:
        state = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(state["model"], strict=False)
        print(f"Đã nạp {args.ckpt} (epoch {state.get('epoch')})")

    def run_eval(label):
        real = real_scan_eval(model, device, mean, std, args.images, args.test_images,
                              args.batch_size * 2)
        print(f"  [{label}] SCAN THẬT n={real['n']} hit@1={real['hit@1']:.4f} "
              f"hit@10={real['hit@10']:.4f} MRR={real['mrr']:.4f} cos={real['cos']:.3f}   "
              f"(mốc zero-shot large: 0,0847 / 0,1864)")
        for rows, tag in ((held_rows, "lớp CHƯA thấy"), (seen_rows, "lớp đã thấy")):
            got = style_transfer_eval(model, device, mean, std, rows, args.images,
                                      args.batch_size * 2, tag)
            if got:
                print(f"  [{label}] {got['tag']}: render->corpus hit@1={got['hit@1']:.4f} "
                      f"(n={got['n']}, ngẫu nhiên={1/got['n']:.5f})")
        return real

    if args.eval_only:
        run_eval("eval")
        return

    dataset = GlyphDataset(train_rows, class_of, mean, std, degrade_prob=args.degrade_prob,
                           strength=args.strength, seed=args.seed)
    if args.loss == "supcon":
        sampler = PKSampler(dataset.labels, args.batch_size, args.views_per_class, args.seed)
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.workers,
                            pin_memory=True, persistent_workers=args.workers > 0)
        head, groups = None, [{"params": model.parameters(), "lr": args.lr_backbone}]
        print(f"SupCon: {sampler.p} lớp × {sampler.k} view/batch, {len(sampler)} batch/epoch")
    else:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, pin_memory=True, drop_last=True,
                            persistent_workers=args.workers > 0)
        head = ArcFace(model.config.projection_dim, len(class_of), args.scale, args.margin).to(device)
        groups = [{"params": model.parameters(), "lr": args.lr_backbone},
                  {"params": head.parameters(), "lr": args.lr_head}]
    optim = torch.optim.AdamW(groups, weight_decay=0.05)
    total = args.epochs * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: min(1.0, (s + 1) / max(args.warmup, 1))
        * 0.5 * (1 + np.cos(np.pi * min(s / max(total, 1), 1.0))))

    print("\nTrước khi huấn luyện (= zero-shot ViT-B/16, mốc so của chính run này):")
    best = run_eval("epoch 0")["hit@1"]

    for epoch in range(1, args.epochs + 1):
        dataset.epoch = epoch
        model.train()
        running, seen, start = 0.0, 0, time.time()
        for step, (images, labels) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                features = model.get_image_features(pixel_values=images)
                if not torch.is_tensor(features):
                    features = features.pooler_output
            features = features.float()
            loss = (supcon_loss(features, labels, args.temperature) if head is None
                    else head(features, labels))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            running += loss.item()
            seen += 1
            if args.eval_steps and step and step % args.eval_steps == 0:
                run_eval(f"epoch {epoch} step {step}")
                model.train()
            if step % 200 == 0:
                print(f"  epoch {epoch} step {step}/{len(loader)} loss={running/seen:.4f} "
                      f"({seen/(time.time()-start):.1f} it/s)", flush=True)
                running, seen, start = 0.0, 0, time.time()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            score = run_eval(f"epoch {epoch}")["hit@1"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)},
                       os.path.join(args.out_dir, "last.pt"))
            if score >= best:
                best = score
                torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)},
                           os.path.join(args.out_dir, "best.pt"))
                print(f"  -> best.pt (hit@1 scan thật = {score:.4f})")

    print(f"\nTốt nhất trên scan thật: hit@1 = {best:.4f}. Checkpoint: {args.out_dir}/best.pt")
    print("Dùng lại: python evaluate_test_images.py --backend chinese-clip-ft --device cuda")


if __name__ == "__main__":
    main()

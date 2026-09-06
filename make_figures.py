"""Sinh mọi hình cho `report/paper.tex` từ **số liệu thật**, không vẽ tay.

Vì sao là script
---------------
Hình vẽ tay thì con số trong hình không có đường nào truy ngược về phép đo, và sửa code xong quên
sửa hình là chuyện chắc chắn xảy ra. Ở đây mọi số nằm trong khối ``MEASUREMENTS`` ngay dưới, kèm
lệnh đã sinh ra nó, nên soát lại là đọc một chỗ duy nhất.

Màu
---
Bảng màu lấy từ bộ mặc định đã kiểm bằng `validate_palette.js` (light, all-pairs): lightness band,
chroma floor, tách biệt dưới mù màu (ΔE 9,2 deutan) và ngưỡng thị lực thường (ΔE 24,0) đều đạt.
Riêng màu aqua có tương phản 2,74:1 so với nền nên **bắt buộc có nhãn số trực tiếp** trên mọi mark —
đó là lý do các hình dưới đây luôn ghi giá trị, không chỉ dựa vào màu.

Phần lớn hình dùng dạng **nhấn mạnh** (một màu + xám) chứ không phải bảng màu phân loại: câu chuyện
là "cái này khác hẳn", không phải "phân biệt các nhóm".

Dùng
----
    .venv/bin/python make_figures.py            # -> report/figs/*.pdf
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image
from scipy.stats import binomtest

# --- bảng màu (đã qua validate_palette.js) ----------------------------------------------------
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
DIM = "#a8a7a2"          # xám khử nhấn cho dạng "emphasis"

FIG_DIR = "report/figs"

#: MỌI con số dưới đây đến từ các lệnh ở Phụ lục A của báo cáo. Không số nào ước lượng bằng tay.
MEASUREMENTS = {
    # evaluate_test_images.py --backend <b> --labels output/label_sheets/label_template.csv
    "scan_hit1": [                        # (nhãn, số đúng, n)
        ("Chinese-CLIP B/16\n(zero-shot)", 1, 59),
        ("Chinese-CLIP L/14\n(zero-shot)", 5, 59),
        ("Đã fine-tune\n(B/16 + SupCon)", 35, 59),
    ],
    # output/finetune/train.log
    "supcon_epochs": [0, 1, 2, 3, 4, 5, 6],
    "supcon_hit1": [0.0169, 0.5763, 0.5424, 0.5085, 0.5254, 0.5932, 0.5763],
    "supcon_cos": [0.890, 0.019, 0.008, 0.016, 0.003, 0.002, 0.000],
    "arcface_epochs": [0, 1],
    "arcface_hit1": [0.0169, 0.0000],
    # render->corpus hit@1, hai mẫu cùng cỡ 2.604 lớp
    "generalisation": [                   # (điều kiện, lớp đã thấy, lớp chưa thấy)
        ("Trước huấn luyện", 0.7938, 0.8041),
        ("SupCon (6 epoch)", 0.9954, 0.9939),
        ("ArcFace (1 epoch)", 0.0035, 0.0050),
    ],
    # evaluate_rerank.py --suite render --sample 600 --backend chinese-clip-large
    "rerank": [                           # (nhãn, số đúng, n, có phải chẩn đoán không)
        ("Tầng 1, không xếp hạng lại", 448, 600, False),
        ("Xếp hạng lại, oracle", 553, 600, True),
        ("Xếp hạng lại, thật", 434, 600, False),
        ("Trộn điểm thô (chưa chuẩn hoá)", 113, 600, False),
    ],
    # evaluate_rerank.py --suite scans; mốc đoán mò tính từ final_characteristics-v2.xlsx
    "radical_infer": [
        ("Đoán đều tay (1/214)", 0.0047),
        ("Luôn đoán bộ phổ biến nhất", 0.0489),
        ("Bỏ phiếu, tầng 1 zero-shot", 0.1186),
        ("Bỏ phiếu, tầng 1 fine-tune", 0.6610),
    ],
}


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.2,
        "axes.edgecolor": GRID, "axes.linewidth": 0.6, "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
        "axes.titlesize": 7.8, "axes.titleweight": "bold", "axes.titlecolor": INK,
        "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "legend.frameon": False, "legend.fontsize": 6.8,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def ci(hits, n):
    """Khoảng tin cậy nhị thức 95%. Dùng đúng phép tính mà báo cáo dùng, không xấp xỉ chuẩn."""
    lo, hi = binomtest(hits, n).proportion_ci(0.95)
    return lo, hi


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}")
    return path


# --- Hình 1: kết quả chính trên scan thật ------------------------------------------------------

def fig_headline():
    """Dot plot có thanh khoảng tin cậy. Dạng **nhấn mạnh**: một mô hình là điểm chính, còn lại là nền.

    Không dùng cột: cột vẽ từ 0 làm mắt so *diện tích*, trong khi thứ cần đọc ở đây là **vị trí và
    độ rộng khoảng tin cậy**. Thanh CI mới là nội dung của hình — nó là lý do kết luận đứng được.
    """
    rows = MEASUREMENTS["scan_hit1"]
    fig, ax = plt.subplots(figsize=(3.4, 1.55))
    ys = range(len(rows))
    for y, (label, hits, n) in zip(ys, rows):
        p = hits / n
        lo, hi = ci(hits, n)
        last = y == len(rows) - 1
        color = BLUE if last else DIM
        ax.plot([lo, hi], [y, y], color=color, lw=2.0, solid_capstyle="round", zorder=2)
        ax.plot([p], [y], "o", color=color, ms=6.5, mec="white", mew=1.2, zorder=3)
        ax.text(hi + 0.022, y, f"{p:.3f}".replace(".", ","), va="center", ha="left",
                fontsize=7.0, color=INK, fontweight="bold" if last else "normal")
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(-0.55, len(rows) - 0.45)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.86)
    ax.set_xlabel("hit@1 trên 59 ảnh scan thật (thanh = KTC 95%)")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.1f}".replace(".", ","))
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return save(fig, "fig1-headline.pdf")


# --- Hình 2: động lực huấn luyện + máy dò sụp đổ -----------------------------------------------

def fig_training():
    """Hai bảng chồng nhau, KHÔNG phải hai trục y trên một bảng.

    hit@1 và cosine trung bình là hai đại lượng khác thang; ép chung một trục là cách chắc chắn
    nhất để người đọc suy ra một tương quan không có thật.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4, 2.5), sharex=True,
                                   gridspec_kw={"height_ratios": [1.25, 1], "hspace": 0.18})

    e, h = MEASUREMENTS["supcon_epochs"], MEASUREMENTS["supcon_hit1"]
    ae, ah = MEASUREMENTS["arcface_epochs"], MEASUREMENTS["arcface_hit1"]
    ax1.plot(e, h, "-o", color=BLUE, lw=1.8, ms=4.2, mec="white", mew=1.0, zorder=3)
    ax1.plot(ae, ah, "-s", color=ORANGE, lw=1.8, ms=4.2, mec="white", mew=1.0, zorder=3)
    best = int(np.argmax(h))
    ax1.annotate("SupCon", (e[3], h[3]), textcoords="offset points", xytext=(0, 7),
                 color=BLUE, fontsize=7.0, fontweight="bold", ha="center")
    ax1.annotate("ArcFace — sụp đổ", (ae[1], ah[1]), textcoords="offset points", xytext=(6, 4),
                 color=ORANGE, fontsize=7.0, fontweight="bold", ha="left")
    ax1.plot([e[best]], [h[best]], "o", ms=9, mfc="none", mec=BLUE, mew=1.2, zorder=2)
    ax1.text(e[best], h[best] + 0.075, f"{h[best]:.4f}".replace(".", ","), ha="center",
             fontsize=6.8, color=INK, fontweight="bold")
    ax1.set_ylim(-0.05, 0.78)
    ax1.set_ylabel("hit@1 (scan thật)")
    ax1.yaxis.set_major_formatter(lambda v, _: f"{v:.1f}".replace(".", ","))

    ax2.plot(e, MEASUREMENTS["supcon_cos"], "-o", color=AQUA, lw=1.8, ms=4.2,
             mec="white", mew=1.0, zorder=3)
    for x, y in ((e[0], MEASUREMENTS["supcon_cos"][0]), (e[-1], MEASUREMENTS["supcon_cos"][-1])):
        ax2.text(x + (0.12 if x == 0 else -0.12), y + 0.06, f"{y:.3f}".replace(".", ","),
                 ha="left" if x == 0 else "right", fontsize=6.8, color=INK, fontweight="bold")
    ax2.set_ylim(-0.08, 1.02)
    ax2.set_ylabel("cosine trung bình")
    ax2.set_xlabel("epoch  (0 = trước huấn luyện)")
    ax2.yaxis.set_major_formatter(lambda v, _: f"{v:.1f}".replace(".", ","))

    for ax in (ax1, ax2):
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xticks(e)
    return save(fig, "fig2-training.pdf")


# --- Hình 3: tổng quát hoá và sụp đổ -----------------------------------------------------------

def fig_generalisation():
    """Cột NGANG nhóm đôi: lớp đã thấy so với lớp chưa thấy, ba điều kiện.

    Ban đầu vẽ cột dọc và nhãn số của hai cột kề nhau dính vào nhau — trong một cột báo rộng 8,6 cm
    thì không đủ chỗ. Cột ngang cho nhãn nằm bên phải, hết va chạm mà không phải bỏ nhãn (bỏ không
    được: màu aqua tương phản 2,74:1 nên buộc phải có nhãn trực tiếp).
    """
    rows = MEASUREMENTS["generalisation"]
    h = 0.34
    fig, ax = plt.subplots(figsize=(3.4, 1.85))
    ys = np.arange(len(rows))
    for i, (color, key, name) in enumerate(((BLUE, 1, "Lớp đã thấy"), (AQUA, 2, "Lớp chưa thấy"))):
        vals = [r[key] for r in rows]
        pos = ys + (i - 0.5) * (h + 0.02)
        ax.barh(pos, vals, h, color=color, label=name, zorder=3)
        for y, v in zip(pos, vals):
            ax.text(v + 0.018, y, f"{v:.4f}".replace(".", ","), va="center", ha="left",
                    fontsize=6.4, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.20)
    ax.set_xlabel("hit@1  render → corpus")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.1f}".replace(".", ","))
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.1)
    return save(fig, "fig3-generalisation.pdf")


# --- Hình 4: xếp hạng lại, kết quả âm ----------------------------------------------------------

def fig_rerank():
    """Hai bảng: (a) bốn cấu hình xếp hạng lại, (b) chỗ nghẽn đã định lượng.

    Cấu hình oracle vẽ rỗng ruột và ghi rõ "chẩn đoán" — nó cố tình gian lận, nên không được trông
    giống một kết quả.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4, 2.75),
                                   gridspec_kw={"height_ratios": [1, 1], "hspace": 0.95})

    rows = MEASUREMENTS["rerank"]
    for y, (label, hits, n, diag) in enumerate(rows):
        p = hits / n
        lo, hi = ci(hits, n)
        color = ORANGE if diag else (DIM if y else BLUE)
        ax1.plot([lo, hi], [y, y], color=color, lw=2.0, solid_capstyle="round", zorder=2)
        ax1.plot([p], [y], "o", color="white" if diag else color, ms=6.0,
                 mec=color, mew=1.6, zorder=3)
        ax1.text(hi + 0.02, y, f"{p:.3f}".replace(".", ","), va="center", ha="left",
                 fontsize=6.8, color=INK)
    ax1.set_yticks(range(len(rows)))
    ax1.set_yticklabels([r[0] + ("  (chẩn đoán)" if r[3] else "") for r in rows], fontsize=6.6)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 1.12)
    ax1.set_ylim(len(rows) - 0.4, -0.6)
    ax1.set_xlabel("hit@1 trên 600 ảnh render (thanh = KTC 95%)")
    ax1.set_title("(a) Bốn cấu hình xếp hạng lại", loc="left", pad=6)
    ax1.xaxis.set_major_formatter(lambda v, _: f"{v:.1f}".replace(".", ","))
    ax1.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    ax1.set_axisbelow(True)
    ax1.spines["left"].set_visible(False)
    ax1.tick_params(axis="y", length=0)

    labels = [r[0] for r in MEASUREMENTS["radical_infer"]]
    vals = [r[1] for r in MEASUREMENTS["radical_infer"]]
    colors = [DIM, DIM, DIM, BLUE]
    ax2.barh(range(len(vals)), vals, 0.6, color=colors, zorder=3)
    for y, v in enumerate(vals):
        ax2.text(v + 0.012, y, f"{v:.4f}".replace(".", ","), va="center", ha="left",
                 fontsize=6.8, color=INK, fontweight="bold" if y == 3 else "normal")
    ax2.set_yticks(range(len(vals)))
    ax2.set_yticklabels(labels, fontsize=6.6)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 0.82)
    ax2.set_xlabel("độ chính xác suy ra bộ thủ của ảnh truy vấn")
    ax2.set_title("(b) Chỗ nghẽn: suy ra siêu dữ liệu", loc="left", pad=6)
    ax2.xaxis.set_major_formatter(lambda v, _: f"{v:.1f}".replace(".", ","))
    ax2.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    ax2.set_axisbelow(True)
    ax2.spines["left"].set_visible(False)
    ax2.tick_params(axis="y", length=0)
    return save(fig, "fig4-rerank.pdf")


# --- Hình 5: truy hồi định tính trước / sau ----------------------------------------------------

def _load(path, size=120):
    img = Image.open(path).convert("L").resize((size, size), Image.LANCZOS)
    return np.asarray(img)


def fig_qualitative(n_queries=3, top_k=5, seed=0):
    """Ảnh truy vấn thật + 5 láng giềng đầu, trước và sau fine-tune, khung xanh = đúng.

    Đây là hình duy nhất cho thấy mô hình *làm gì*, chứ không phải nó ghi được bao nhiêu điểm. Chọn
    truy vấn theo quy tắc cố định (những ảnh mà bản fine-tune đúng còn zero-shot sai, lấy theo thứ
    tự tên file) chứ không bới tìm ví dụ đẹp — và ghi rõ quy tắc đó ngay trong chú thích hình.
    """
    import pandas as pd
    from evaluate_test_images import decode_filenames, ground_truth
    from search_all_chars_in_corpus import CHAR_TABLE
    from search_use_QuocNgu_mapping import QUOCNGU_TABLE

    quocngu, chars = pd.read_excel(QUOCNGU_TABLE), pd.read_excel(CHAR_TABLE)
    caches = {k: np.load(f"output/embeddings_{k}_fp16.npz", allow_pickle=True)
              for k in ("chinese-clip-large", "chinese-clip-ft")}
    unicodes = np.array([str(u) for u in caches["chinese-clip-ft"]["unicodes"]])
    have = set(unicodes.tolist())

    matched, _ = decode_filenames("test_images", quocngu)
    labelled = [(p, s, w, gt) for p, s, w in matched
                if (gt := ground_truth(quocngu, chars, w, have))]

    from feature_extractors import build_extractor
    ranked = {}
    for key, cache in caches.items():
        os.environ.setdefault("GLYPH_FT_CKPT", "output/finetune/best.pt")
        ex = build_extractor(key, device="cpu", dtype="fp16")
        q = ex.embed_paths([p for p, _, _, _ in labelled], batch_size=16)
        corpus = cache["embeddings"].astype("float32")
        order = np.argsort(-(q @ corpus.T), axis=1)[:, :top_k]
        ranked[key] = unicodes[order]

    # Quy tắc chọn: fine-tune đúng ở hạng 1, zero-shot sai ở hạng 1.
    picks = [i for i, (_, _, _, gt) in enumerate(labelled)
             if ranked["chinese-clip-ft"][i][0] in gt
             and ranked["chinese-clip-large"][i][0] not in gt]
    picks = sorted(picks, key=lambda i: labelled[i][1])[:n_queries]

    # Mỗi truy vấn là một subfigure riêng: cần khe TRẮNG giữa các khối, nếu không ba cặp
    # zero-shot/fine-tune dính liền thành một lưới 6 hàng và người đọc không thấy ranh giới.
    fig = plt.figure(figsize=(3.4, 1.16 * len(picks) + 0.06))
    blocks = fig.subfigures(len(picks), 1, hspace=0.06)
    blocks = np.atleast_1d(blocks)
    for block, i in zip(blocks, picks):
        path, stem, word, gt = labelled[i]
        axes = block.subplots(2, top_k + 1)
        block.subplots_adjust(wspace=0.06, hspace=0.06, top=0.775, bottom=0.02)
        block.suptitle(f'truy vấn: "{word}"', fontsize=6.6, color=INK, y=1.00)
        for half, key in enumerate(("chinese-clip-large", "chinese-clip-ft")):
            ax_row = axes[half]
            ax_row[0].imshow(_load(path), cmap="gray", vmin=0, vmax=255)
            ax_row[0].set_ylabel("zero-shot" if half == 0 else "fine-tune",
                                 fontsize=5.6, color=MUTED, rotation=0,
                                 ha="right", va="center", labelpad=3)
            for c, u in enumerate(ranked[key][i]):
                a = ax_row[c + 1]
                f = os.path.join("images", f"{u}.jpg")
                if os.path.exists(f):
                    a.imshow(_load(f), cmap="gray", vmin=0, vmax=255)
                if u in gt:
                    a.add_patch(Rectangle((0, 0), 119, 119, fill=False, edgecolor=BLUE, lw=2.0))
            for c, a in enumerate(ax_row):
                a.set_xticks([]); a.set_yticks([])
                for sp in a.spines.values():
                    sp.set_edgecolor(MUTED if c == 0 else GRID)
                    sp.set_linewidth(0.9 if c == 0 else 0.5)
        for c in range(top_k + 1):
            axes[0][c].annotate("truy vấn" if c == 0 else f"#{c}", (0.5, 1.10),
                                xycoords="axes fraction", ha="center", fontsize=5.6, color=MUTED)
    return save(fig, "fig5-qualitative.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-qualitative", action="store_true",
                        help="bỏ Hình 5 (cần cache embedding và checkpoint)")
    args = parser.parse_args()

    style()
    print("Đang sinh hình:")
    fig_headline()
    fig_training()
    fig_generalisation()
    fig_rerank()
    if not args.skip_qualitative:
        fig_qualitative()


if __name__ == "__main__":
    main()

"""Trần của `radical@k`: 0,5 là bão hoà, hay là giới hạn của chính nhãn?

Trả lời phản hồi "kết quả bị bão hoà ở 0.5" bằng ba phép đo, tương ứng ba giả thuyết:

theoretical (--stage ceiling)
    Trần lý thuyết của ``radical@k``: nếu corpus chỉ có m ký tự cùng bộ thủ với query thì
    ``radical@k`` không thể vượt ``min(m, k)/k``. Kiểm tra xem kích thước tập top-k có phải
    chỗ nghẽn không.

curve (--stage curve)
    ``radical@k`` theo k = 1..100 cho từng backend. Nếu ``radical@1`` đã thấp thì nghẽn nằm ở
    láng giềng gần nhất, không phải ở đuôi danh sách -- và nới k sẽ không cứu được gì.

oracle (--stage oracle)
    Trần *thực tế*: xếp hạng bằng Jaccard trên thành phần IDS thật (``SHAPE_MORPH``), tức một
    bộ rank "biết trước cấu trúc chữ" -- tốt hơn bất kỳ mô hình thị giác nào. Kèm kiểm tra
    ``RADICAL`` có phải nhãn thị giác không (bộ thủ có thật sự xuất hiện trong hình chữ?).

Kết quả đã đo nằm ở mục 9 của BENCHMARK.md.

Ví dụ
-----
    python analyze_metric_ceiling.py                    # cả ba giai đoạn
    python analyze_metric_ceiling.py --stage oracle
"""

import argparse
import os

import numpy as np
import pandas as pd

from benchmark_extractors import parse_ids_components, parse_radical
from feature_extractors import BACKENDS, DTYPES
from search_all_chars_in_corpus import CHAR_TABLE, cache_tag, load_corpus

KS = (1, 2, 3, 5, 10, 20, 50, 100)


def load_labels(images):
    df = load_corpus(images, limit=None)
    meta = pd.read_excel(CHAR_TABLE)[["UNICODE", "RADICAL", "SHAPE_MORPH"]]
    merged = df.merge(meta, on="UNICODE", how="left")
    return (
        df,
        merged["RADICAL"].map(parse_radical).to_numpy(dtype=object),
        merged["SHAPE_MORPH"].map(parse_ids_components).to_numpy(dtype=object),
    )


def stage_ceiling(radicals, top_k):
    """min(m, k)/k trung bình trên mọi query -- trần nếu retrieval hoàn hảo."""
    sizes = pd.Series([r for r in radicals if r]).value_counts().to_dict()
    known = [r for r in radicals if r]
    total = len(radicals)

    print(f"\n=== 1. Trần lý thuyết ===")
    print(f"corpus N={total}, có RADICAL={len(known)}, số bộ thủ={len(sizes)}")
    counts = pd.Series(list(sizes.values()))
    print(f"kích thước nhóm bộ thủ: median={counts.median():.0f} mean={counts.mean():.1f} "
          f"max={counts.max()} | nhóm ≤{top_k} phần tử: {(counts <= top_k).sum()}/{len(counts)}")

    for k in KS:
        ceil = np.mean([min(sizes[r] - 1, k) / k for r in known])
        mark = "  <-- top-K của đồ án" if k == top_k else ""
        print(f"  trần radical@{k:<4d} = {ceil:.4f}{mark}")

    capped = sum(1 for r in known if sizes[r] - 1 < top_k)
    print(f"  query bị chạm trần ở k={top_k}: {capped}/{len(known)} ({capped/len(known)*100:.0f}%)")
    print(f"  baseline ngẫu nhiên      = "
          f"{np.mean([(sizes[r] - 1) / (total - 1) for r in known]):.4f}")


def stage_curve(radicals, backends, output_dir, cache_dtypes):
    """radical@k theo k, để phân biệt 'nghẽn ở rank 1' với 'loãng dần ở đuôi'."""
    import faiss

    print(f"\n=== 2. radical@k theo k ===")
    print(f"{'backend':22s} " + " ".join(f"@{k:<6d}" for k in KS))
    k_max = max(KS)

    for backend in backends:
        path = next((p for p in (
            os.path.join(output_dir, f"embeddings_{cache_tag(backend, dt)}.npz")
            for dt in cache_dtypes) if os.path.exists(p)), None)
        if path is None:
            print(f"[skip] {backend}: chưa có cache trong {output_dir}")
            continue

        emb = np.ascontiguousarray(np.load(path, allow_pickle=True)["embeddings"].astype("float32"))
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        _, found = index.search(emb, k_max + 1)
        # Bỏ chính query khỏi danh sách lân cận của nó.
        neighbours = np.array([[j for j in row if j != i][:k_max] for i, row in enumerate(found)])

        same = np.array([
            [radicals[j] == radicals[i] if radicals[i] and radicals[j] else np.nan
             for j in neighbours[i]]
            for i in range(len(radicals))
        ], dtype=float)
        row = [np.nanmean(np.nanmean(same[:, :k], axis=1)) for k in KS]
        print(f"{backend:22s} " + " ".join(f"{v:.4f} " for v in row))


def stage_oracle(radicals, components, top_k, chunk=2000):
    """Trần thực tế: xếp hạng bằng thành phần IDS thật, cộng kiểm tra nhãn có thị giác không."""
    from scipy import sparse

    total = len(radicals)
    print(f"\n=== 3. RADICAL có phải nhãn thị giác? ===")
    # 'nhất 一' -> '一': bộ thủ là ký tự Hán cuối chuỗi.
    radchar = np.array([r.split()[-1] if r else None for r in radicals], dtype=object)
    usable = np.array([r is not None and len(c) > 0 for r, c in zip(radchar, components)])
    visible = np.array([radchar[i] in components[i] for i in np.where(usable)[0]])
    print(f"ký tự có cả RADICAL và SHAPE_MORPH: {usable.sum()}")
    print(f"  bộ thủ xuất hiện trong phân rã IDS của chính nó: {visible.mean()*100:.1f}%")
    print(f"  -> {100 - visible.mean()*100:.1f}% trường hợp bộ thủ KHÔNG nhìn thấy được từ hình chữ")

    vocab = {c: i for i, c in enumerate(sorted({c for s in components for c in s}))}
    rows = [(i, vocab[c]) for i, s in enumerate(components) for c in s]
    matrix = sparse.csr_matrix(
        (np.ones(len(rows), "float32"), tuple(zip(*rows))), shape=(total, len(vocab)))
    sizes = np.asarray(matrix.sum(1)).ravel()

    print(f"\n=== 4. Oracle IDS-Jaccard (trần thực tế) ===")
    print(f"ma trận thành phần: {total} x {len(vocab)}")
    hits = []
    for start in range(0, total, chunk):
        block = matrix[start:start + chunk]
        inter = (block @ matrix.T).toarray()
        union = sizes[start:start + chunk, None] + sizes[None, :] - inter
        jaccard = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        for r in range(block.shape[0]):
            i = start + r
            if radicals[i] is None or sizes[i] == 0:
                continue
            jaccard[r, i] = -1.0  # loại chính query
            top = np.argpartition(-jaccard[r], top_k)[:top_k]
            valid = [radicals[j] == radicals[i] for j in top if radicals[j] is not None]
            if valid:
                hits.append(np.mean(valid))
        print(f"  {min(start + chunk, total)}/{total}", end="\r", flush=True)

    print(f"\n>>> oracle radical@{top_k} = {np.mean(hits):.4f}  (n={len(hits)})")
    print("    Đây mới là trần mà một phương pháp dựa trên hình chữ có thể đạt tới.")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", default="all", choices=("all", "ceiling", "curve", "oracle"))
    parser.add_argument("--backends", nargs="+", default=list(BACKENDS), choices=BACKENDS)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--images", default="./images")
    parser.add_argument("--output-dir", default="./output")
    args = parser.parse_args()

    _, radicals, components = load_labels(args.images)

    if args.stage in ("all", "ceiling"):
        stage_ceiling(radicals, args.top_k)
    if args.stage in ("all", "curve"):
        stage_curve(radicals, args.backends, args.output_dir, sorted(DTYPES))
    if args.stage in ("all", "oracle"):
        stage_oracle(radicals, components, args.top_k)


if __name__ == "__main__":
    main()

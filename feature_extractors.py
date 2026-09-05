"""Pluggable image feature extractors for Sino-Nom glyph similarity.

Every backend exposes the same contract:

    extractor = build_extractor("chinese-clip")
    embeddings = extractor.embed_paths(list_of_image_paths)   # (N, D) float32, L2-normalized

Because the vectors are L2-normalized, inner product == cosine similarity, which is what the
Faiss ``IndexFlatIP`` in ``search_all_chars_in_corpus.py`` expects.

Available backends
------------------
resnet18       Baseline from the original script: ImageNet ResNet18 whose first conv is replaced
               by a freshly-initialized 1-channel conv. Kept bit-for-bit so benchmarks have an
               honest "before" number -- see the note in ``ResNet18Extractor``.
resnet18-gray  Same architecture, but the 1-channel conv is seeded by summing the pretrained RGB
               filters instead of being re-initialized, so ImageNet features survive.
chinese-clip   OFA-Sys/chinese-clip-vit-base-patch16 image tower. Pretrained on Chinese
               image-text pairs, so its prior is much closer to Han glyphs than ImageNet.
dinov2         facebook/dinov2-base. Self-supervised ViT, strong general-purpose visual features.
chinese-clip-ft  ``chinese-clip`` after ``finetune_glyph.py``. Same architecture and same embedding
               width, so every downstream script works unchanged -- only the weights differ. The
               checkpoint path comes from ``$GLYPH_FT_CKPT`` (default ``output/finetune/best.pt``);
               loading fails loudly rather than silently falling back to the pretrained weights,
               because a silent fallback would report zero-shot numbers under a finetuned name.
"""

import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

BACKENDS = (
    "resnet18",
    "resnet18-gray",
    "chinese-clip",
    "chinese-clip-large",
    "chinese-clip-huge",
    "chinese-clip-ft",
    "dinov2",
)

_HF_MODEL_IDS = {
    "chinese-clip": "OFA-Sys/chinese-clip-vit-base-patch16",
    "chinese-clip-large": "OFA-Sys/chinese-clip-vit-large-patch14",
    "chinese-clip-huge": "OFA-Sys/chinese-clip-vit-huge-patch14",
    # Cùng kiến trúc ViT-B/16 với ``chinese-clip``; khác ở chỗ trọng số được nạp đè bằng checkpoint
    # của ``finetune_glyph.py``. Giữ nguyên ``visual_projection`` nên hình dạng khớp tuyệt đối.
    "chinese-clip-ft": "OFA-Sys/chinese-clip-vit-base-patch16",
    "dinov2": "facebook/dinov2-base",
}

#: ``--dtype`` values. fp16 halves weight memory, which is what makes ViT-H fit alongside the vLLM
#: workers; it only affects the HuggingFace backends (the ResNets are cheap enough to leave alone).
DTYPES = {"fp32": torch.float32, "fp16": torch.float16}

#: Checkpoint mặc định cho backend ``chinese-clip-ft``; ghi đè bằng biến môi trường ``GLYPH_FT_CKPT``.
FT_CHECKPOINT = "output/finetune/best.pt"


def _from_pretrained(cls, model_id, dtype):
    """``from_pretrained`` with the dtype kwarg that this transformers version actually accepts.

    transformers renamed ``torch_dtype`` to ``dtype`` around 4.56 and warns on the old name; the
    old name is still the only one that works on earlier v4 releases.
    """
    try:
        return cls.from_pretrained(model_id, dtype=dtype)
    except TypeError:
        return cls.from_pretrained(model_id, torch_dtype=dtype)


def _l2_normalize(matrix):
    """Row-wise L2 normalization that leaves all-zero rows untouched instead of producing NaN."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return matrix / norms


class BaseExtractor:
    """Common batching/loading logic. Subclasses only implement image prep + the forward pass."""

    name = "base"
    dim = 0
    #: Mode passed to ``PIL.Image.convert``; glyphs are grayscale but ViT towers want 3 channels.
    image_mode = "L"

    def _forward(self, images):
        """Map a list of PIL images to a (B, D) float32 numpy array (not yet normalized)."""
        raise NotImplementedError

    def embed_paths(self, image_paths, batch_size=64, progress_every=0):
        """Embed images from disk in batches. Returns (N, D) float32, L2-normalized."""
        vectors = []
        total = len(image_paths)
        for start in range(0, total, batch_size):
            chunk = image_paths[start : start + batch_size]
            images = [Image.open(path).convert(self.image_mode) for path in chunk]
            vectors.append(self._forward(images))
            if progress_every and (start // batch_size) % progress_every == 0:
                print(f"  [{self.name}] {min(start + batch_size, total)}/{total}", flush=True)
        if not vectors:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _l2_normalize(np.vstack(vectors).astype(np.float32))


class ResNet18Extractor(BaseExtractor):
    """ImageNet ResNet18 with the classification head removed -> 512-d pooled features.

    ``faithful=True`` reproduces the original script exactly, including the flaw that patching
    ``conv1`` to 1 channel *discards* the pretrained first-layer filters and replaces them with
    random weights. That single line is why the baseline underperforms: everything downstream
    consumes features built on a random edge detector. ``faithful=False`` instead sums the
    pretrained RGB filters into one channel, which is the standard grayscale adaptation and keeps
    the ImageNet prior intact.
    """

    dim = 512

    def __init__(self, faithful=True, device="cpu"):
        from torchvision import models, transforms

        self.name = "resnet18" if faithful else "resnet18-gray"
        self.device = torch.device(device)

        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        pretrained_conv1 = model.conv1.weight.data.clone()
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if not faithful:
            # (64, 3, 7, 7) -> (64, 1, 7, 7); summing preserves the response to grayscale input.
            model.conv1.weight.data = pretrained_conv1.sum(dim=1, keepdim=True)

        self.model = nn.Sequential(*list(model.children())[:-1]).to(self.device).eval()
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((100, 100), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    @torch.no_grad()
    def _forward(self, images):
        batch = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        return self.model(batch).flatten(1).cpu().numpy()


class ChineseClipExtractor(BaseExtractor):
    """Image tower of Chinese-CLIP, using the projected image embedding.

    Serves every ``chinese-clip*`` backend; the size comes from the model id, and the embedding
    width is read off the config (base 512, large 768, huge 1024) rather than hardcoded.
    """

    image_mode = "RGB"

    def __init__(self, backend="chinese-clip", device="cpu", dtype="fp32"):
        from transformers import AutoImageProcessor, ChineseCLIPModel

        model_id = _HF_MODEL_IDS[backend]
        self.name = backend
        self.device = torch.device(device)
        self.dtype = DTYPES[dtype]
        # Deliberately the *slow* (PIL) processor -- see the use_fast note in CLAUDE.md. The fast
        # torchvision path benchmarks faster in isolation but is single-threaded-stable here,
        # whereas fast contends for CPU with the vLLM workers and its throughput swings ~2x.
        self.processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False)

        model = _from_pretrained(ChineseCLIPModel, model_id, self.dtype)
        # ``get_image_features`` only touches vision_model + visual_projection. The text tower is a
        # full RoBERTa that would otherwise be copied to the GPU and never used -- ~1.3 GB wasted on
        # ViT-H. Drop it while the model is still on CPU so only the vision half is transferred.
        del model.text_model, model.text_projection
        if backend.endswith("-ft"):
            path = os.environ.get("GLYPH_FT_CKPT", FT_CHECKPOINT)
            if not os.path.exists(path):
                raise SystemExit(f"Chưa có checkpoint {path} -- chạy finetune_glyph.py trước, "
                                 "hoặc trỏ $GLYPH_FT_CKPT vào file khác.")
            state = torch.load(path, map_location="cpu")["model"]
            # strict=False vì checkpoint còn giữ text_model/logit_scale mà ở đây đã xoá; nhưng phải
            # kiểm tay là các khoá vision ĐỀU khớp, không thì nạp hụt mà vẫn im lặng.
            missing, _ = model.load_state_dict(state, strict=False)
            missing = [k for k in missing if not k.startswith(("text_model", "text_projection"))]
            if missing:
                raise SystemExit(f"Checkpoint thiếu {len(missing)} khoá vision, vd {missing[:3]}")
            model = model.to(self.dtype)
            print(f"[chinese-clip-ft] nạp {path}")
        self.model = model.to(self.device).eval()
        self.dim = self.model.config.projection_dim

    @torch.no_grad()
    def _forward(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        # pixel_values must match the weight dtype under fp16; any integer tensors must not be cast.
        inputs = {
            key: value.to(self.device, dtype=self.dtype)
            if value.is_floating_point()
            else value.to(self.device)
            for key, value in inputs.items()
        }
        features = self.model.get_image_features(**inputs)
        # transformers <5 returns the projected tensor directly; >=5 wraps it in an output object
        # whose ``pooler_output`` holds that same projection.
        if not torch.is_tensor(features):
            features = features.pooler_output
        return features.float().cpu().numpy()


class Dinov2Extractor(BaseExtractor):
    """DINOv2 ViT-B/14. Uses CLS token concatenated with mean-pooled patch tokens (1536-d).

    Concatenating both is the pooling DINOv2's own linear-probe evaluations use: the CLS token
    carries global shape, the patch mean carries stroke-level texture, and glyph similarity needs
    both.
    """

    name = "dinov2"
    image_mode = "RGB"

    def __init__(self, device="cpu", dtype="fp32"):
        from transformers import AutoImageProcessor, AutoModel

        model_id = _HF_MODEL_IDS["dinov2"]
        self.device = torch.device(device)
        self.dtype = DTYPES[dtype]
        # See the note in ChineseClipExtractor -- slow processor is the deliberate choice.
        self.processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False)
        self.model = _from_pretrained(AutoModel, model_id, self.dtype).to(self.device).eval()
        self.dim = self.model.config.hidden_size * 2

    @torch.no_grad()
    def _forward(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {
            key: value.to(self.device, dtype=self.dtype)
            if value.is_floating_point()
            else value.to(self.device)
            for key, value in inputs.items()
        }
        hidden = self.model(**inputs).last_hidden_state  # (B, 1 + n_patches, H)
        pooled = torch.cat([hidden[:, 0], hidden[:, 1:].mean(dim=1)], dim=1)
        return pooled.float().cpu().numpy()


def build_extractor(backend, device="cpu", dtype="fp32"):
    """Instantiate a backend by name. See ``BACKENDS`` for valid values.

    ``dtype`` applies to the HuggingFace backends only; the ResNets always run in fp32 because they
    are fast enough that halving them buys nothing.
    """
    if backend == "resnet18":
        return ResNet18Extractor(faithful=True, device=device)
    if backend == "resnet18-gray":
        return ResNet18Extractor(faithful=False, device=device)
    if backend.startswith("chinese-clip"):
        return ChineseClipExtractor(backend, device=device, dtype=dtype)
    if backend == "dinov2":
        return Dinov2Extractor(device=device, dtype=dtype)
    raise ValueError(f"Unknown backend {backend!r}; expected one of {BACKENDS}")

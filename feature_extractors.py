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
"""

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

BACKENDS = ("resnet18", "resnet18-gray", "chinese-clip", "dinov2")

_HF_MODEL_IDS = {
    "chinese-clip": "OFA-Sys/chinese-clip-vit-base-patch16",
    "dinov2": "facebook/dinov2-base",
}


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
    """Image tower of Chinese-CLIP, using the projected image embedding (512-d)."""

    name = "chinese-clip"
    image_mode = "RGB"

    def __init__(self, device="cpu"):
        from transformers import AutoImageProcessor, ChineseCLIPModel

        model_id = _HF_MODEL_IDS["chinese-clip"]
        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = ChineseCLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.dim = self.model.config.projection_dim

    @torch.no_grad()
    def _forward(self, images):
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        features = self.model.get_image_features(**inputs)
        # transformers <5 returns the projected tensor directly; >=5 wraps it in an output object
        # whose ``pooler_output`` holds that same projection.
        if not torch.is_tensor(features):
            features = features.pooler_output
        return features.cpu().numpy()


class Dinov2Extractor(BaseExtractor):
    """DINOv2 ViT-B/14. Uses CLS token concatenated with mean-pooled patch tokens (1536-d).

    Concatenating both is the pooling DINOv2's own linear-probe evaluations use: the CLS token
    carries global shape, the patch mean carries stroke-level texture, and glyph similarity needs
    both.
    """

    name = "dinov2"
    image_mode = "RGB"

    def __init__(self, device="cpu"):
        from transformers import AutoImageProcessor, AutoModel

        model_id = _HF_MODEL_IDS["dinov2"]
        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.dim = self.model.config.hidden_size * 2

    @torch.no_grad()
    def _forward(self, images):
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        hidden = self.model(**inputs).last_hidden_state  # (B, 1 + n_patches, H)
        pooled = torch.cat([hidden[:, 0], hidden[:, 1:].mean(dim=1)], dim=1)
        return pooled.cpu().numpy()


def build_extractor(backend, device="cpu"):
    """Instantiate a backend by name. See ``BACKENDS`` for valid values."""
    if backend == "resnet18":
        return ResNet18Extractor(faithful=True, device=device)
    if backend == "resnet18-gray":
        return ResNet18Extractor(faithful=False, device=device)
    if backend == "chinese-clip":
        return ChineseClipExtractor(device=device)
    if backend == "dinov2":
        return Dinov2Extractor(device=device)
    raise ValueError(f"Unknown backend {backend!r}; expected one of {BACKENDS}")

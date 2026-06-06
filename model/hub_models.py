from huggingface_hub import PyTorchModelHubMixin

from .wfm import VisionTransformer

MODEL_CARD_TEMPLATE = """\
---
license: mit
pipeline_tag: image-classification
tags:
- image-classification
- vision
- weightformer
library_name: weightformer
---

# WeightFormer

{description}

## Usage

```python
import torch
from torchvision import transforms
from weightformer import {class_name}

model = {class_name}.from_pretrained("{repo_id}")
model.eval()

transform = transforms.Compose([
    transforms.Resize({resize}),
    transforms.CenterCrop({img_size}),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

image = transform(image).unsqueeze(0)
with torch.no_grad():
    output = model(image)
```

## Model Details

- **Architecture:** Vision Transformer with dynamic weight prediction (no explicit attention)
- **Embedding dimension:** {embed_dim}
- **Depth:** {depth}
- **Image size:** {img_size}
- **Patch size:** {patch_size}
- **Params:** {params}
- **Drop path rate:** {drop_path}

## Citation

```bibtex
@article{{he2024weightformer,
  title={{Linear-Time Global Visual Modeling without Explicit Attention}},
  author={{He, Ruize and Han, Dongchen and Huang, Gao}},
  journal={{arXiv preprint arXiv:2605.01711}},
  year={{2026}}
}}
```
"""

_MODEL_CARD_INFO = {
    "WeightFormer_T": {
        "class_name": "WeightFormer_T",
        "description": "**WeightFormer-Tiny** — a lightweight variant of WeightFormer for image classification on ImageNet-1K.",
        "embed_dim": 192,
        "depth": 17,
        "img_size": 224,
        "patch_size": 16,
        "drop_path": "0.0",
        "resize": 256,
    },
    "WeightFormer_S": {
        "class_name": "WeightFormer_S",
        "description": "**WeightFormer-Small** — a mid-size variant of WeightFormer for image classification on ImageNet-1K.",
        "embed_dim": 384,
        "depth": 17,
        "img_size": 224,
        "patch_size": 16,
        "drop_path": "0.2",
        "resize": 256,
    },
    "WeightFormer_B": {
        "class_name": "WeightFormer_B",
        "description": "**WeightFormer-Base** — a large variant of WeightFormer for image classification on ImageNet-1K at 448×448 resolution.",
        "embed_dim": 384,
        "depth": 17,
        "img_size": 448,
        "patch_size": 16,
        "drop_path": "0.2",
        "resize": 512,
    },
}


def _format_model_card(cls, repo_id, params=None):
    info = _MODEL_CARD_INFO[cls.__name__].copy()
    info["repo_id"] = repo_id
    info["params"] = params or "N/A"
    return MODEL_CARD_TEMPLATE.format(**info)


class WeightFormer_T(
    VisionTransformer,
    PyTorchModelHubMixin,
    repo_url="https://github.com/Horizonll/WeightFormer",
    paper_url="https://arxiv.org/abs/2605.01711",
    pipeline_tag="image-classification",
    license="mit",
    library_name="weightformer",
    tags=["vision", "image-classification", "weightformer"],
):
    """WeightFormer-Tiny: embed_dim=192, depth=17, img_size=224, patch_size=16"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=192,
        depth=17,
        mlp_ratio=4.0,
        qkv_bias=True,
        representation_size=None,
        distilled=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            representation_size=representation_size,
            distilled=distilled,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )

    def generate_model_card(self, repo_id="weightformer-t", params=None, **kwargs):
        return _format_model_card(type(self), repo_id, params)


class WeightFormer_S(
    VisionTransformer,
    PyTorchModelHubMixin,
    repo_url="https://github.com/Horizonll/WeightFormer",
    paper_url="https://arxiv.org/abs/2605.01711",
    pipeline_tag="image-classification",
    license="mit",
    library_name="weightformer",
    tags=["vision", "image-classification", "weightformer"],
):
    """WeightFormer-Small: embed_dim=384, depth=17, img_size=224, patch_size=16"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=384,
        depth=17,
        mlp_ratio=4.0,
        qkv_bias=True,
        representation_size=None,
        distilled=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            representation_size=representation_size,
            distilled=distilled,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )

    def generate_model_card(self, repo_id="weightformer-s", params=None, **kwargs):
        return _format_model_card(type(self), repo_id, params)


class WeightFormer_B(
    VisionTransformer,
    PyTorchModelHubMixin,
    repo_url="https://github.com/Horizonll/WeightFormer",
    paper_url="https://arxiv.org/abs/2605.01711",
    pipeline_tag="image-classification",
    license="mit",
    library_name="weightformer",
    tags=["vision", "image-classification", "weightformer"],
):
    """WeightFormer-Base: embed_dim=384, depth=17, img_size=448, patch_size=16"""

    def __init__(
        self,
        img_size=448,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=384,
        depth=17,
        mlp_ratio=4.0,
        qkv_bias=True,
        representation_size=None,
        distilled=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            representation_size=representation_size,
            distilled=distilled,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )

    def generate_model_card(self, repo_id="weightformer-b", params=None, **kwargs):
        return _format_model_card(type(self), repo_id, params)

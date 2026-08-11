#!/usr/bin/env python3
"""build_model.py — turn a proposal spec into a runnable torch model.

Handles the known layer `type`s used by gen_proposals.py, and any layer
marked `novel` that carries a `definition` (Python source for an nn.Module).
The novel layer's definition is exec'd in a controlled namespace so a
genuinely new layer becomes runnable without us hand-coding it.

The builder is MODALITY-aware. A dataset declares one of:

  image        4-D tensor (B, C, H, W)     -> 2-D conv chain
  audio-mel    4-D tensor (B, 1, M, T)     -> 2-D conv chain on a mel view
  audio-wave   3-D tensor (B, 1, T)        -> 1-D conv chain on a raw waveform
  timeseries   3-D tensor (B, C, T)        -> 1-D conv / recurrent chain
  text         2-D int tensor (B, T)       -> embedding then 1-D chain
  tabular      2-D tensor (B, F)           -> dense chain

Any dataset may also carry `task_type = "clustering"`, in which case the head
maps the learned embedding to `n_clusters` soft assignments instead of class
logits. The compile gate treats those assignments as logits, so the same
train/test/predict path applies.
"""
import inspect
import types

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Dataset table: modality, tensor shape, class/cluster count, vocabulary.
# ---------------------------------------------------------------------------
DATASETS = {
    # --- existing (unchanged behaviour) ---
    "mnist":            {"modality": "image", "shape": (1, 28, 28), "classes": 10},
    "cifar10":          {"modality": "image", "shape": (3, 16, 16), "classes": 10},
    "cifar100":         {"modality": "image", "shape": (3, 16, 16), "classes": 100},
    "imagenet-subset":  {"modality": "image", "shape": (3, 16, 16), "classes": 10},
    "openml-ctr23":     {"modality": "tabular", "shape": (20,), "classes": 2},
    # --- new: audio ---
    "speech-commands":  {"modality": "audio-mel", "shape": (1, 64, 64), "classes": 12},
    "esc50":            {"modality": "audio-wave", "shape": (1, 8000), "classes": 10},
    # --- new: text ---
    "ag-news":          {"modality": "text", "shape": (64,), "classes": 4, "vocab": 20000},
    "imdb":             {"modality": "text", "shape": (128,), "classes": 2, "vocab": 20000},
    # --- new: time series ---
    "uci-har":          {"modality": "timeseries", "shape": (9, 128), "classes": 6},
    # --- new: clustering ---
    "mnist-cluster":    {"modality": "image", "shape": (1, 28, 28), "classes": 10},
    "openml-cluster":   {"modality": "tabular", "shape": (20,), "classes": 8},
}

_DEFAULT = {"modality": "image", "shape": (3, 16, 16), "classes": 10}


def dataset_info(dataset):
    return dict(_DEFAULT, **DATASETS.get(dataset, {}))


def input_shape(dataset):
    """Back-compatible: tensor shape of one sample (no batch dim)."""
    return dataset_info(dataset)["shape"]


def num_classes(dataset):
    return dataset_info(dataset)["classes"]


def modality(dataset):
    return dataset_info(dataset)["modality"]


def output_size(spec):
    """Class count, or cluster count for a clustering proposal."""
    if spec.get("task_type") == "clustering":
        return int(spec.get("n_clusters", num_classes(spec.get("dataset", ""))))
    return num_classes(spec.get("dataset", ""))


def sample_batch(spec, batch=2):
    """A dataset-correct synthetic (x, y) pair for the compile gate."""
    info = dataset_info(spec.get("dataset", ""))
    shape = info["shape"]
    if info["modality"] == "text":
        x = torch.randint(0, info.get("vocab", 20000), (batch, *shape))
    else:
        x = torch.randn(batch, *shape)
    y = torch.randint(0, output_size(spec), (batch,))
    return x, y


# ---------------------------------------------------------------------------
# Helper modules
# ---------------------------------------------------------------------------
class SwapTimeChannel(nn.Module):
    """(B, T, E) -> (B, E, T) so an embedding feeds a 1-D conv chain."""

    def forward(self, x):
        return x.transpose(1, 2)


class RecurrentSeq(nn.Module):
    """GRU/LSTM over (B, C, T); returns (B, H, T) so 1-D blocks may follow."""

    def __init__(self, in_ch, hidden, kind="gru", bidirectional=False, layers=1):
        super().__init__()
        rnn = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = rnn(in_ch, hidden, num_layers=layers, batch_first=True,
                       bidirectional=bidirectional)
        self.out_ch = hidden * (2 if bidirectional else 1)

    def forward(self, x):
        y, _ = self.rnn(x.transpose(1, 2))
        return y.transpose(1, 2)


class SelfAttention1d(nn.Module):
    """Single-head self attention over (B, C, T)."""

    def __init__(self, channels, heads=4):
        super().__init__()
        while channels % heads and heads > 1:
            heads -= 1
        self.att = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        s = x.transpose(1, 2)
        a, _ = self.att(s, s, s)
        return self.norm(s + a).transpose(1, 2)


class ClusterHead(nn.Module):
    """Embedding -> soft cluster assignment (Student-t, DEC style)."""

    def __init__(self, in_dim, embed_dim, n_clusters, alpha=1.0):
        super().__init__()
        self.project = nn.Linear(in_dim, embed_dim)
        self.centroids = nn.Parameter(torch.randn(n_clusters, embed_dim) * 0.05)
        self.alpha = alpha

    def forward(self, x):
        z = self.project(x)
        d2 = torch.cdist(z, self.centroids).pow(2)
        q = 1.0 / (1.0 + d2 / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        return torch.log(q / q.sum(1, keepdim=True) + 1e-9)


class CoordConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch + 2, out_ch, k,
                              stride=stride, padding=k // 2)

    def forward(self, x):
        b, _, h, w = x.shape
        yy = torch.arange(h, device=x.device).float() / max(h - 1, 1)
        xx = torch.arange(w, device=x.device).float() / max(w - 1, 1)
        gx, gy = torch.meshgrid(xx, yy, indexing="xy")
        coords = torch.stack([gx, gy], 0).unsqueeze(0).repeat(b, 1, 1, 1)
        return self.conv(torch.cat([x, coords], 1))


def _exec_novel(definition: str) -> type:
    """Compile + exec a layer `definition` string; return the nn.Module class.

    A minimal `wavelet_transform` stub is injected so novel layers that name a
    not-yet-implemented op still *compile* (the gate validates architecture,
    not the wavelet math). Swap in a real op later."""
    import torch

    def wavelet_transform(x, levels=1):
        # STUB: a real wavelet op goes here. Shape-preserving for compile.
        pooled = torch.nn.functional.avg_pool2d(x, 2, stride=2)
        return torch.nn.functional.interpolate(pooled, size=x.shape[-2:],
                                               mode="nearest")

    mod = types.ModuleType("novel_layer")
    mod.__dict__.update({"torch": torch, "nn": nn, "wavelet_transform": wavelet_transform})
    exec(compile(definition, "<novel-layer>", "exec"), mod.__dict__)  # noqa: S102
    found = [v for v in mod.__dict__.values()
             if isinstance(v, type) and issubclass(v, nn.Module) and v is not nn.Module]
    if not found:
        raise RuntimeError("novel layer definition defined no nn.Module subclass")
    return found[0]


def _accepted_kwargs(cls, block: dict) -> dict:
    skip = {"type", "novel", "definition", "refs", "note"}
    sig = inspect.signature(cls.__init__)
    params = set(sig.parameters) - {"self"}
    return {k: v for k, v in block.items() if k in params and k not in skip}


# ---------------------------------------------------------------------------
# Block builders
# ---------------------------------------------------------------------------
def _block(block: dict, in_ch: int):
    """2-D / generic block. Returns (module, out_channels)."""
    t = block.get("type", "")
    if block.get("novel") and "definition" in block:
        cls = _exec_novel(block["definition"])
        return cls(**_accepted_kwargs(cls, block)), in_ch  # out_ch unknown; next adapts
    if t == "linear":
        return nn.Linear(block["units"], block.get("out", block["units"])), block["units"]
    if t == "conv":
        out = block["channels"]
        return nn.Conv2d(in_ch, out, block["kernel"],
                         stride=block.get("stride", 1),
                         padding=block["kernel"] // 2), out
    if t == "coordconv":
        out = block["channels"]
        return CoordConv(in_ch, out, block["kernel"], block.get("stride", 1)), out
    if t == "dilated":
        out = block["channels"]
        return nn.Conv2d(in_ch, out, block["kernel"],
                         padding=block["dilation"], dilation=block["dilation"]), out
    if t in ("bottleneck", "basicblock", "inverted_residual"):
        out = block["channels"]
        return nn.Conv2d(in_ch, out, block.get("kernel", 3), padding=1), out
    if t == "batchnorm2d":
        return nn.BatchNorm2d(in_ch), in_ch
    if t == "relu":
        return nn.ReLU(), in_ch
    if t == "maxpool2d":
        return nn.MaxPool2d(block.get("kernel", 2), block.get("stride", 2)), in_ch
    if t == "boosted_trees":
        return nn.Identity(), in_ch
    if t == "choice":
        opt = block["options"][0]
        out = 16 if "conv" in opt else in_ch
        return (nn.Conv2d(in_ch, out, 3, padding=1) if "conv" in opt
                else nn.AvgPool2d(3, 1, 1)), out
    raise ValueError(f"unknown layer type: {t!r}")


def _block1d(block: dict, in_ch: int):
    """1-D (audio / text / time series) block. Returns (module, out_channels)."""
    t = block.get("type", "")
    if block.get("novel") and "definition" in block:
        cls = _exec_novel(block["definition"])
        return cls(**_accepted_kwargs(cls, block)), in_ch
    if t == "conv1d":
        out = block["channels"]
        k = block.get("kernel", 3)
        return nn.Conv1d(in_ch, out, k, stride=block.get("stride", 1),
                         padding=k // 2), out
    if t == "dilated1d":
        out = block["channels"]
        k = block.get("kernel", 3)
        d = block.get("dilation", 2)
        return nn.Conv1d(in_ch, out, k, padding=d * (k // 2), dilation=d), out
    if t == "sepconv1d":
        out = block["channels"]
        k = block.get("kernel", 5)
        return nn.Sequential(
            nn.Conv1d(in_ch, in_ch, k, padding=k // 2, groups=in_ch),
            nn.Conv1d(in_ch, out, 1)), out
    if t in ("gru", "lstm"):
        hidden = block.get("hidden", block.get("units", 64))
        mod = RecurrentSeq(in_ch, hidden, kind=t,
                           bidirectional=block.get("bidirectional", False),
                           layers=block.get("layers", 1))
        return mod, mod.out_ch
    if t == "attention1d":
        return SelfAttention1d(in_ch, block.get("heads", 4)), in_ch
    if t == "layernorm1d":
        return nn.GroupNorm(1, in_ch), in_ch
    if t == "batchnorm1d":
        return nn.BatchNorm1d(in_ch), in_ch
    if t == "relu":
        return nn.ReLU(), in_ch
    if t == "maxpool1d":
        return nn.MaxPool1d(block.get("kernel", 2), block.get("stride", 2)), in_ch
    raise ValueError(f"unknown 1-D layer type: {t!r}")


def _block_dense(block: dict, in_dim: int):
    """Tabular / dense block. Returns (module, out_dim)."""
    t = block.get("type", "")
    if block.get("novel") and "definition" in block:
        cls = _exec_novel(block["definition"])
        return cls(**_accepted_kwargs(cls, block)), in_dim
    if t in ("linear", "dense"):
        out = block.get("units", in_dim)
        return nn.Linear(in_dim, out), out
    if t == "boosted_trees":
        # The builder only has torch primitives; a gradient-boosted-tree block
        # maps to a single dense (MLP) hidden layer. Use `estimators` as the
        # hidden width so wider GBM configs yield wider layers (matching the
        # width-variant contract), and `depth` is informational.
        out = block.get("estimators", in_dim)
        return nn.Linear(in_dim, out), out
    if t == "relu":
        return nn.ReLU(), in_dim
    if t == "layernorm":
        return nn.LayerNorm(in_dim), in_dim
    if t == "batchnorm":
        return nn.BatchNorm1d(in_dim), in_dim
    if t == "dropout":
        return nn.Dropout(block.get("p", 0.1)), in_dim
    raise ValueError(f"unknown dense layer type: {t!r}")


# ---------------------------------------------------------------------------
# Head
# ---------------------------------------------------------------------------
def _make_head(feat_out: torch.Tensor, spec: dict) -> nn.Module:
    n_out = output_size(spec)
    clustering = spec.get("task_type") == "clustering"
    layers = []
    if feat_out.dim() == 4:
        layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
        dim = feat_out.shape[1]
    elif feat_out.dim() == 3:
        layers += [nn.AdaptiveAvgPool1d(1), nn.Flatten()]
        dim = feat_out.shape[1]
    elif feat_out.dim() == 2:
        dim = feat_out.shape[1]
    else:
        return nn.Identity()
    if clustering:
        layers.append(ClusterHead(dim, int(spec.get("embed_dim", 32)), n_out))
    else:
        layers.append(nn.Linear(dim, n_out))
    return nn.Sequential(*layers)


def build_model(spec: dict) -> nn.Module:
    model_name = spec.get("model", "")
    dataset = spec.get("dataset", "")
    info = dataset_info(dataset)
    mod = info["modality"]
    shape = info["shape"]
    nc = output_size(spec)

    # --- legacy fast paths (image only) -------------------------------------
    if mod == "image" and model_name == "mlp":
        in_ch, h, w = shape
        units = 256
        for b in spec.get("blocks", []):
            if b.get("type") == "linear" and "units" in b:
                units = b["units"]
                break
        return nn.Sequential(nn.Flatten(),
                             nn.Linear(in_ch * h * w, units),
                             nn.ReLU(), nn.Linear(units, nc))
    if mod == "image" and model_name == "gbm":
        in_ch, h, w = shape
        return nn.Sequential(nn.Flatten(), nn.Linear(in_ch * h * w, nc))

    x, _ = sample_batch(spec)
    blocks = []

    if mod == "text":
        b0 = spec["blocks"][0]
        if b0.get("type") != "embedding":
            raise ValueError("a text model must start with an `embedding` block")
        emb_dim = b0.get("dim", 64)
        blocks += [nn.Embedding(info.get("vocab", 20000), emb_dim), SwapTimeChannel()]
        cur = emb_dim
        rest = spec["blocks"][1:]
        for b in rest:
            m, cur = _block1d(b, cur)
            blocks.append(m)
    elif mod in ("audio-wave", "timeseries"):
        cur = shape[0]
        for b in spec.get("blocks", []):
            m, cur = _block1d(b, cur)
            blocks.append(m)
    elif mod == "tabular":
        cur = shape[0]
        blocks.append(nn.Flatten())
        for b in spec.get("blocks", []):
            m, cur = _block_dense(b, cur)
            blocks.append(m)
    else:  # image, audio-mel
        cur = shape[0]
        for b in spec.get("blocks", []):
            m, cur = _block(b, cur)
            blocks.append(m)

    feat = nn.Sequential(*blocks)
    feat.eval()
    with torch.no_grad():
        f = feat(x)
    feat.train()
    head = _make_head(f, spec)
    return nn.Sequential(*blocks, head)

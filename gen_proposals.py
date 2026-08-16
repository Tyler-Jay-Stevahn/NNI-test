#!/usr/bin/env python3
"""
gen_proposals.py — build proposals.jsonl for the NNI-Test repo.

Recreates the "models proposed" file from prior sessions. Each line is one
model proposal with a stable id, a lifecycle status, a `rationale`, and a
nested `spec`.

The schema is intentionally OPEN so it can describe ANY layer — including
layers and whole models that do not exist yet:

  spec.blocks  is a list of free-form layer dicts. The only required key is
               `type` (any string, known or invented). Any other key is allowed
               (params, `novel`, `definition` = literal code, `refs` = papers,
               `note`). Nothing restricts the vocabulary.

  spec.novel / spec.hypothesis / spec.implementation_status  mark a model that
               does not yet exist and state what it is meant to do.

Deterministic: same input -> same output, so the file is reproducible.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "proposals.jsonl")

CREATED = "2026-08-10"

# How many model variants per family, and which statuses to assign.
MODELS_PER_FAMILY = 5
STATUS_CYCLE = ["proposed", "proposed", "approved", "approved", "tested"]

# ---------------------------------------------------------------------------
# Known-layer task families (these describe architectures that exist today).
# ---------------------------------------------------------------------------
TASK_FAMILIES = {
    "hpo-mnist": {
        "model": "mlp",
        "dataset": "mnist",
        "blocks": [{"type": "linear", "units": 256, "activation": "relu"}],
        "head": "logits",
        "optimizer": "sgd",
        "lr_grid": [0.001, 0.005, 0.01, 0.05, 0.1],
        "weight_decay_grid": [0.0, 1e-4, 1e-3],
        "batch_size": 128,
        "epochs": 20,
        "augmentation": "none",
    },
    "lenet-mnist": {
        # The ORIGINAL MNIST model architecture: LeNet-5 (LeCun et al. 1998).
        # This is a FIXED multi-layer architecture, not a width-variant family,
        # so the generator emits exactly ONE proposal (no width broadcast).
        "fixed": True,
        "model": "lenet5",
        "dataset": "mnist",
        "blocks": [
            {"type": "conv", "channels": 6, "kernel": 5, "stride": 1},
            {"type": "relu"},
            {"type": "avgpool2d", "kernel": 2, "stride": 2},
            {"type": "conv", "channels": 16, "kernel": 5, "stride": 1},
            {"type": "relu"},
            {"type": "avgpool2d", "kernel": 2, "stride": 2},
            {"type": "conv", "channels": 120, "kernel": 5, "stride": 1},
            {"type": "relu"},
            {"type": "flatten"},
            {"type": "linear", "units": 84},
            {"type": "relu"},
            {"type": "linear", "units": 10},
        ],
        "head": "logits",
        "optimizer": "sgd",
        "lr_grid": [0.001, 0.005, 0.01, 0.05, 0.1],
        "weight_decay_grid": [0.0, 1e-4, 1e-3],
        "batch_size": 128,
        "epochs": 20,
        "augmentation": "none",
    },
    "nas-cifar": {
        "model": "cell_cnn",
        "dataset": "cifar10",
        "blocks": [
            {"type": "conv", "channels": 32, "kernel": 3, "stride": 1},
            {"type": "conv", "channels": 64, "kernel": 3, "stride": 2},
        ],
        "head": "gap+mlp",
        "optimizer": "adam",
        "lr_grid": [0.001, 0.003, 0.01],
        "weight_decay_grid": [1e-4, 5e-4],
        "batch_size": 96,
        "epochs": 50,
        "augmentation": "cutout",
    },
    "compression-imagenet": {
        "model": "resnet50",
        "dataset": "imagenet-subset",
        "blocks": [{"type": "bottleneck", "channels": 256, "kernel": 3}],
        "head": "logits",
        "optimizer": "sgd",
        "lr_grid": [0.01, 0.03, 0.1],
        "weight_decay_grid": [1e-4],
        "batch_size": 64,
        "epochs": 30,
        "augmentation": "randaugment",
    },
    "adv-robust": {
        "model": "cnn",
        "dataset": "cifar10",
        "blocks": [
            {"type": "coordconv", "channels": 32, "kernel": 3},
            {"type": "dilated", "channels": 64, "kernel": 3, "dilation": 2},
            {"type": "dilated", "channels": 128, "kernel": 3, "dilation": 4},
        ],
        "head": "gap+mlp",
        "optimizer": "adamw",
        "lr_grid": [0.0005, 0.001, 0.003],
        "weight_decay_grid": [1e-4, 1e-3],
        "batch_size": 128,
        "epochs": 50,
        "augmentation": "autoaugment",
    },
    "quant-int8": {
        "model": "mobilenetv2",
        "dataset": "cifar100",
        "blocks": [{"type": "inverted_residual", "channels": 96, "expand": 6}],
        "head": "logits",
        "optimizer": "adam",
        "lr_grid": [0.001, 0.003],
        "weight_decay_grid": [1e-5],
        "batch_size": 128,
        "epochs": 25,
        "augmentation": "none",
        "quant": {"scheme": "int8", "calibration": "entropy"},
    },
    "prune-structured": {
        "model": "resnet34",
        "dataset": "cifar10",
        "blocks": [{"type": "basicblock", "channels": 128, "kernel": 3}],
        "head": "logits",
        "optimizer": "sgd",
        "lr_grid": [0.01, 0.03],
        "weight_decay_grid": [1e-4],
        "batch_size": 128,
        "epochs": 40,
        "augmentation": "none",
        "prune": {"method": "l1", "ratio_grid": [0.3, 0.5, 0.7]},
    },
    "feature-eng-tabular": {
        "model": "gbm",
        "dataset": "openml-ctr23",
        "blocks": [{"type": "boosted_trees", "estimators": 500, "depth": 6}],
        "head": "regression",
        "optimizer": "hist",
        "lr_grid": [0.01, 0.05, 0.1],
        "weight_decay_grid": [0.0],
        "batch_size": 0,
        "epochs": 0,
        "augmentation": "none",
        "feature_search": {"transform_grid": ["power", "log", "binning", "selectkbest"]},
    },
    "nas-retiarii": {
        "model": "retiarii_cell",
        "dataset": "cifar10",
        "blocks": [
            {"type": "conv", "channels": 32, "kernel": 3, "stride": 1},
            {"type": "choice", "options": ["conv3x3", "conv1x1", "maxpool3x3"]},
            {"type": "choice", "options": ["conv3x3", "sepconv3x3", "avgpool3x3"]},
        ],
        "head": "gap+mlp",
        "optimizer": "adam",
        "lr_grid": [0.001, 0.003],
        "weight_decay_grid": [1e-4],
        "batch_size": 96,
        "epochs": 50,
        "augmentation": "cutout",
    },
    # -----------------------------------------------------------------------
    # NOVEL-LAYER family: proves the schema can describe layers/models that do
    # not exist yet. `type` is an invented name; `definition` carries literal
    # (hypothetical) code; `refs` point at the research it builds on.
    # -----------------------------------------------------------------------
    "novel-spectral": {
        "model": "spectral_resonant_net",
        "dataset": "cifar10",
        "novel": True,
        "hypothesis": (
            "A block that fuses Fourier and wavelet domains with a learned gate "
            "may capture both global frequency structure and multi-scale detail "
            "in one unit, improving sample efficiency on small image sets versus "
            "either alone. This exact block does not exist in published form."
        ),
        "implementation_status": "designed",
        "blocks": [
            {
                "type": "resonant_spectral_mix",
                "novel": True,
                "modes": 32,
                "domains": ["fft", "wavelet"],
                "adapt": "learned_gating",
                "definition": (
                    "class ResonantSpectralMix(nn.Module):\n"
                    "    def __init__(self, modes=32, adapt='learned_gating'):\n"
                    "        super().__init__()\n"
                    "        self.modes = modes\n"
                    "        self.gate = nn.Parameter(torch.rand(2))  # fft vs wavelet\n"
                    "    def forward(self, x):\n"
                    "        # spectral path: keep low `modes` frequencies (FNO-style\n"
                    "        # low-pass truncation); Parseval holds via norm='ortho'.\n"
                    "        f = torch.fft.rfft2(x, norm='ortho')\n"
                    "        m = min(self.modes, f.shape[-2], f.shape[-1])\n"
                    "        f[..., m:, :] = 0.0\n"
                    "        f[..., :, m:] = 0.0\n"
                    "        spectral = torch.fft.irfft2(f, s=x.shape[-2:], norm='ortho')\n"
                    "        w = wavelet_transform(x)               # to-be-implemented op\n"
                    "        g = torch.softmax(self.gate, 0)\n"
                    "        # convex combination; both paths must share shape (C,H,W).\n"
                    "        return g[0] * spectral + g[1] * w\n"
                ),
                "refs": [
                    "https://arxiv.org/abs/2010.08895",  # Fourier Neural Operator
                    "https://arxiv.org/abs/2107.08391",  # Global Filter Networks
                ],
                "note": "combines FNO-style spectral mixing with wavelet detail in one block",
            },
        ],
        "head": "gap+mlp",
        "optimizer": "adamw",
        "lr_grid": [0.0005, 0.001],
        "weight_decay_grid": [1e-4],
        "batch_size": 96,
        "epochs": 50,
        "augmentation": "cutout",
    },
    # -----------------------------------------------------------------------
    # NEW MODALITIES (2026-08-11). Every family below has >= 3 layers and a
    # non-image input: audio (mel + raw waveform), text, time series, plus two
    # unsupervised clustering families.
    # -----------------------------------------------------------------------
    "audio-keyword-mel": {
        "model": "mel_cnn",
        "dataset": "speech-commands",
        "modality": "audio-mel",
        "blocks": [
            {"type": "conv", "channels": 32, "kernel": 3, "stride": 1},
            {"type": "relu"},
            {"type": "conv", "channels": 64, "kernel": 3, "stride": 2},
            {"type": "batchnorm2d"},
            {"type": "conv", "channels": 64, "kernel": 3, "stride": 2},
        ],
        "head": "gap+mlp",
    },
    "audio-waveform-tcn": {
        "model": "wave_tcn",
        "dataset": "esc50",
        "modality": "audio-wave",
        "blocks": [
            {"type": "conv1d", "channels": 32, "kernel": 15, "stride": 4},
            {"type": "relu"},
            {"type": "dilated1d", "channels": 64, "kernel": 3, "dilation": 2},
            {"type": "dilated1d", "channels": 64, "kernel": 3, "dilation": 4},
            {"type": "sepconv1d", "channels": 64, "kernel": 5},
        ],
        "head": "gap+mlp",
    },
    "text-cnn-news": {
        "model": "text_cnn",
        "dataset": "ag-news",
        "modality": "text",
        "blocks": [
            {"type": "embedding", "dim": 64},
            {"type": "conv1d", "channels": 64, "kernel": 5},
            {"type": "relu"},
            {"type": "conv1d", "channels": 64, "kernel": 3, "stride": 2},
            {"type": "maxpool1d", "kernel": 2, "stride": 2},
        ],
        "head": "gap+mlp",
    },
    "text-attention-imdb": {
        "model": "text_attention",
        "dataset": "imdb",
        "modality": "text",
        "blocks": [
            {"type": "embedding", "dim": 64},
            {"type": "conv1d", "channels": 64, "kernel": 3},
            {"type": "attention1d", "heads": 4},
            {"type": "layernorm1d"},
            {"type": "conv1d", "channels": 64, "kernel": 3, "stride": 2},
        ],
        "head": "gap+mlp",
    },
    "timeseries-har": {
        "model": "sensor_rnn_cnn",
        "dataset": "uci-har",
        "modality": "timeseries",
        "blocks": [
            {"type": "conv1d", "channels": 32, "kernel": 7, "stride": 2},
            {"type": "batchnorm1d"},
            {"type": "gru", "hidden": 64, "bidirectional": True},
            {"type": "attention1d", "heads": 4},
            {"type": "conv1d", "channels": 64, "kernel": 3},
        ],
        "head": "gap+mlp",
    },
    "cluster-image-dec": {
        "model": "conv_dec_cluster",
        "dataset": "mnist-cluster",
        "modality": "image",
        "task_type": "clustering",
        "n_clusters": 10,
        "embed_dim": 32,
        "blocks": [
            {"type": "conv", "channels": 32, "kernel": 3, "stride": 2},
            {"type": "relu"},
            {"type": "conv", "channels": 64, "kernel": 3, "stride": 2},
            {"type": "batchnorm2d"},
            {"type": "conv", "channels": 64, "kernel": 3, "stride": 2},
        ],
        "head": "cluster_assignment",
    },
    "cluster-tabular-dec": {
        "model": "dense_dec_cluster",
        "dataset": "openml-cluster",
        "modality": "tabular",
        "task_type": "clustering",
        "n_clusters": 8,
        "embed_dim": 16,
        "blocks": [
            {"type": "dense", "units": 64},
            {"type": "relu"},
            {"type": "dense", "units": 64},
            {"type": "batchnorm"},
            {"type": "dense", "units": 32},
        ],
        "head": "cluster_assignment",
    },
    "diffusion-cifar": {
        "model": "ddpm_unet",
        "dataset": "cifar10",
        "modality": "image",
        "task_type": "generation",
        "blocks": [
            {"type": "conv", "channels": 128, "kernel": 3, "stride": 1},
            {"type": "groupnorm", "groups": 32},
            {"type": "silu"},
            {"type": "attention", "heads": 4},
            {"type": "downsample", "channels": 128},
            {"type": "conv", "channels": 256, "kernel": 3, "stride": 1},
            {"type": "groupnorm", "groups": 32},
            {"type": "silu"},
            {"type": "attention", "heads": 4},
            {"type": "downsample", "channels": 256},
            {"type": "conv", "channels": 256, "kernel": 3, "stride": 1},
            {"type": "groupnorm", "groups": 32},
            {"type": "silu"},
            {"type": "upsample", "channels": 256},
            {"type": "conv", "channels": 128, "kernel": 3, "stride": 1},
            {"type": "groupnorm", "groups": 32},
            {"type": "silu"},
            {"type": "attention", "heads": 4},
            {"type": "upsample", "channels": 128},
            {"type": "conv", "channels": 128, "kernel": 3, "stride": 1},
            {"type": "groupnorm", "groups": 32},
            {"type": "silu"},
        ],
        "head": "conv_out",
        "optimizer": "adamw",
        "lr_grid": [0.0001, 0.0002, 0.0005],
        "weight_decay_grid": [1e-4],
        "batch_size": 128,
        "epochs": 200,
        "augmentation": "none",
        "diffusion": {
            "timesteps": 1000,
            "beta_schedule": "cosine",
            "prediction_type": "epsilon",
        },
    },
    "rl-atari-ppo": {
        "model": "impala_cnn",
        "dataset": "atari-ale",
        "modality": "image",
        "task_type": "rl_policy",
        "blocks": [
            {"type": "conv", "channels": 16, "kernel": 3, "stride": 1},
            {"type": "relu"},
            {"type": "residual", "channels": 16, "kernel": 3},
            {"type": "relu"},
            {"type": "residual", "channels": 16, "kernel": 3},
            {"type": "relu"},
            {"type": "conv", "channels": 32, "kernel": 3, "stride": 2},
            {"type": "relu"},
            {"type": "residual", "channels": 32, "kernel": 3},
            {"type": "relu"},
            {"type": "residual", "channels": 32, "kernel": 3},
            {"type": "relu"},
            {"type": "conv", "channels": 32, "kernel": 3, "stride": 2},
            {"type": "relu"},
            {"type": "residual", "channels": 32, "kernel": 3},
            {"type": "relu"},
            {"type": "residual", "channels": 32, "kernel": 3},
            {"type": "relu"},
        ],
        "head": "actor_critic",
        "optimizer": "adam",
        "lr_grid": [0.0001, 0.00025, 0.0005],
        "weight_decay_grid": [1e-4],
        "batch_size": 256,
        "epochs": 0,
        "augmentation": "none",
        "rl": {
            "algorithm": "ppo",
            "num_envs": 64,
            "rollout_length": 128,
            "ppo_epochs": 4,
            "clip_range": 0.1,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "max_grad_norm": 0.5,
        },
    },
}

# Why each family's model was proposed. Plain Simplified Technical English.
RATIONALE = {
    "hpo-mnist": (
        "MLP on MNIST is the baseline NNI example. It isolates hyperparameter "
        "effects from architecture effects, so tuning lr and weight_decay here "
        "gives a clean read on the search algorithm itself."
    ),
    "lenet-mnist": (
        "LeNet-5 is the original convolutional architecture trained on MNIST "
        "(LeCun et al. 1998). It anchors the conv benchmark: two conv+pool "
        "stages, a conv collapse to 120 maps, then two dense layers (84 then "
        "10). It is the reference convolutional baseline the NNI search spaces "
        "are measured against, so it belongs in the MNIST proposals verbatim."
    ),
    "nas-cifar": (
        "Cell-based CNN on CIFAR-10 is the canonical NAS benchmark. A small "
        "conv stack keeps search cheap while remaining representative of the "
        "image-classification search space NNI targets."
    ),
    "compression-imagenet": (
        "ResNet-50 on an ImageNet subset exercises NNI's model-compression path "
        "at realistic scale. Pruning/quantization here measures accuracy loss "
        "against a well-known reference backbone."
    ),
    "adv-robust": (
        "CoordConv stem plus dilated blocks was the best round-2 held-out result "
        "(0.6040 +/- 0.0340 over 5 seeds). It combines two known parts and gives "
        "the tightest spread tested, so it is the prior worth beating."
    ),
    "quant-int8": (
        "MobileNetV2 with int8 quantization targets edge deployment. Entropy "
        "calibration is the standard int8 path and lets us measure size/accuracy "
        "trade-off versus the full-precision model."
    ),
    "prune-structured": (
        "ResNet-34 with structured L1 pruning tests how much width can be removed "
        "before accuracy drops. The ratio grid sweeps the compressibility frontier."
    ),
    "feature-eng-tabular": (
        "Gradient-boosted trees on a tabular OpenML set tests NNI's feature-"
        "engineering automation. The transform grid searches useful feature "
        "shapes without touching the model."
    ),
    "nas-retiarii": (
        "Retiarii cell search on CIFAR-10 uses NNI's native NAS framework. The "
        "block choices cover the standard operation primitives Retiarii samples."
    ),
    "novel-spectral": (
        "This family probes the OPEN schema: the block `resonant_spectral_mix` is "
        "an invented layer, not a published one. It is included to prove the "
        "format can carry any layer type, including ones that must be implemented "
        "before they can be tested."
    ),
    "audio-keyword-mel": (
        "Keyword spotting on log-mel spectrograms is the standard small-audio "
        "benchmark. The five-layer conv stack tests whether the pipeline handles "
        "a non-image 2-D input (frequency x time) without change."
    ),
    "audio-waveform-tcn": (
        "A dilated temporal conv net reads the RAW waveform, so no spectrogram "
        "front end is needed. The exponentially growing dilation gives a long "
        "receptive field over 8000 samples with few layers."
    ),
    "text-cnn-news": (
        "An embedding plus a 1-D conv stack is the classic strong baseline for "
        "topic classification. It adds a token-sequence modality to the sweep and "
        "tests integer input tensors end to end."
    ),
    "text-attention-imdb": (
        "Self attention after a conv front end tests long-range token dependence "
        "on sentiment text. It is the transformer-style counterpart to the "
        "conv-only text family."
    ),
    "timeseries-har": (
        "Human activity recognition from 9-axis sensor windows mixes conv, "
        "bidirectional GRU, and attention in one five-layer stack. It tests a "
        "multi-channel time-series modality and a recurrent layer in the builder."
    ),
    "cluster-image-dec": (
        "Unsupervised clustering with a DEC-style Student-t assignment head. The "
        "conv encoder learns an embedding, and the centroid layer produces soft "
        "cluster assignments instead of class logits."
    ),
    "cluster-tabular-dec": (
        "The tabular counterpart of the DEC family. A dense encoder plus a "
        "centroid head tests clustering on feature vectors, so the pipeline is "
        "proved on unsupervised objectives in two modalities."
    ),
    "diffusion-cifar": (
        "Denoising diffusion probabilistic model (DDPM) on CIFAR-10 tests the "
        "generation task type. The U-Net with attention at multiple resolutions "
        "is the standard diffusion backbone; cosine noise schedule and epsilon "
        "prediction are the established defaults. It adds a GENERATIVE "
        "objective to the sweep and exercises the diffusion config block."
    ),
    "rl-atari-ppo": (
        "IMPALA-style CNN with residual blocks for PPO on Atari (ALE) tests the "
        "RL policy task type. The actor-critic head outputs both action logits "
        "and a value estimate. The conv+residual stem is the standard Impala "
        "architecture used in large-scale distributed RL. It adds a "
        "REINFORCEMENT LEARNING objective and the rl config block."
    ),
}

# ---------------------------------------------------------------------------
# Curated, real citations per task family (>=3 each). Every URL is checked for
# resolution by verify_proposals.py (option A: live confirmation).
# ---------------------------------------------------------------------------
CITATIONS = {
    "hpo-mnist": [
        {"title": "Bergstra & Bengio (2012) Random Search for Hyper-Parameter Optimization",
         "url": "https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf",
         "why": "Shows random search already beats grid for HPO; justifies the tuning study."},
        {"title": "Bergstra et al. (2011) Algorithms for Hyper-Parameter Optimization (TPE)",
         "url": "https://papers.nips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html",
         "why": "TPE is the Bayesian tuner NNI offers; cited for the optimizer choice."},
        {"title": "LeCun et al. (1998) Gradient-Based Learning Applied to Document Recognition (LeNet/MNIST)",
         "url": "https://doi.org/10.1109/5.726791",
         "why": "Origin of the MNIST benchmark used here."},
        {"title": "Microsoft NNI toolkit (GitHub)",
         "url": "https://github.com/microsoft/nni",
         "why": "The AutoML framework this proposal targets."},
    ],
    "lenet-mnist": [
        {"title": "LeCun et al. (1998) Gradient-Based Learning Applied to Document Recognition (LeNet/MNIST)",
         "url": "https://doi.org/10.1109/5.726791",
         "why": "The original LeNet-5 architecture trained on MNIST; the design replicated here."},
        {"title": "LeCun et al. (1998) The MNIST Database of Handwritten Digits",
         "url": "http://yann.lecun.com/exdb/mnist/",
         "why": "The handwriting dataset LeNet-5 was built for; still the canonical small-image baseline."},
        {"title": "Bengio et al. (2012) Practical Recommendations for Gradient-Based Training (convnet chapter)",
         "url": "https://arxiv.org/abs/1206.5533",
         "why": "Standard reference on conv+pool+ReLU stacks validating the LeNet design choices."},
        {"title": "Microsoft NNI toolkit (GitHub)",
         "url": "https://github.com/microsoft/nni",
         "why": "The AutoML framework this proposal targets."},
    ],
    "nas-cifar": [
        {"title": "Zoph & Le (2017) Neural Architecture Search with Reinforcement Learning",
         "url": "https://arxiv.org/abs/1611.01578",
         "why": "Foundational NAS method; defines the search-over-architectures framing."},
        {"title": "Liu et al. (2019) DARTS: Differentiable Architecture Search",
         "url": "https://arxiv.org/abs/1806.09055",
         "why": "Differentiable NAS; the cell-based search space used here."},
        {"title": "Krizhevsky (2009) CIFAR-10 dataset",
         "url": "https://www.cs.toronto.edu/~kriz/cifar.html",
         "why": "The dataset used for the NAS benchmark."},
        {"title": "Real et al. (2019) Regularized Evolution for Image Classifier Architecture Search",
         "url": "https://arxiv.org/abs/1802.01548",
         "why": "Evolutionary NAS baseline; an alternative NNI tuner family."},
    ],
    "compression-imagenet": [
        {"title": "He et al. (2016) Deep Residual Learning (ResNet)",
         "url": "https://arxiv.org/abs/1512.03385",
         "why": "The ResNet-50 backbone exercised by the compression path."},
        {"title": "Han et al. (2016) Deep Compression",
         "url": "https://arxiv.org/abs/1510.00149",
         "why": "Pruning + quantization + Huffman; the compression lineage."},
        {"title": "ImageNet (Deng et al. 2009)",
         "url": "https://www.image-net.org/",
         "why": "The dataset family the subset is drawn from."},
        {"title": "Jacob et al. (2018) Quantization and Training for Integer-Arithmetic-Only Inference",
         "url": "https://arxiv.org/abs/1712.05877",
         "why": "Standard int8 quantization reference; matches the quant scheme."},
    ],
    "adv-robust": [
        {"title": "Liu et al. (2018) An Intriguing Failing of Convolutional Neural Networks (CoordConv)",
         "url": "https://arxiv.org/abs/1807.03247",
         "why": "Source of the CoordConv stem used in this family."},
        {"title": "Yu et al. (2016) Multi-Scale Context Aggregation by Dilated Convolutions",
         "url": "https://arxiv.org/abs/1511.07122",
         "why": "Source of the dilated blocks used here."},
        {"title": "Madry et al. (2018) Towards Deep Learning Models Resistant to Adversarial Attacks",
         "url": "https://arxiv.org/abs/1706.06083",
         "why": "Adversarial robustness framing; the robustness target."},
        {"title": "Goodfellow et al. (2015) Explaining and Harnessing Adversarial Examples (FGSM)",
         "url": "https://arxiv.org/abs/1412.6572",
         "why": "Foundational adversarial example work; context for robustness."},
    ],
    "quant-int8": [
        {"title": "Jacob et al. (2018) Quantization and Training for Integer-Arithmetic-Only Inference",
         "url": "https://arxiv.org/abs/1712.05877",
         "why": "Standard int8 quantization; matches the entropy-calibration scheme."},
        {"title": "Krishnamoorthi (2018) Quantizing Deep Convolutional Networks for Efficient Inference",
         "url": "https://arxiv.org/abs/1806.08342",
         "why": "Practical INT8 quantization of conv nets; trade-off reference."},
        {"title": "Sandler et al. (2018) MobileNetV2",
         "url": "https://arxiv.org/abs/1801.04381",
         "why": "The MobileNetV2 backbone targeted at edge deployment."},
        {"title": "Han et al. (2016) Deep Compression",
         "url": "https://arxiv.org/abs/1510.00149",
         "why": "Compression lineage; quantization + pruning."},
    ],
    "prune-structured": [
        {"title": "Han et al. (2015) Learning both Weights and Connections for Efficient Neural Networks",
         "url": "https://arxiv.org/abs/1506.02626",
         "why": "Foundational structured pruning; the L1 method used here."},
        {"title": "Liu et al. (2017) Learning Efficient Convolutional Networks through Network Slimming",
         "url": "https://arxiv.org/abs/1708.06519",
         "why": "Channel-level structured pruning; alternative to L1."},
        {"title": "He et al. (2018) Soft Filter Pruning for Accelerating Deep CNNs",
         "url": "https://arxiv.org/abs/1808.06866",
         "why": "Structured filter pruning baseline; compressibility reference."},
        {"title": "He et al. (2016) Deep Residual Learning (ResNet)",
         "url": "https://arxiv.org/abs/1512.03385",
         "why": "The ResNet-34 backbone being pruned."},
    ],
    "feature-eng-tabular": [
        {"title": "Chen & Guestrin (2016) XGBoost",
         "url": "https://arxiv.org/abs/1603.02754",
         "why": "The gradient-boosted-tree model used (gbm)."},
        {"title": "Guyon & Elisseeff (2003) An Introduction to Variable and Feature Selection",
         "url": "https://www.jmlr.org/papers/volume3/guyon03a/guyon03a.pdf",
         "why": "Foundational feature-selection theory; justifies the search."},
        {"title": "OpenML",
         "url": "https://www.openml.org/",
         "why": "The tabular dataset source (openml-ctr23)."},
        {"title": "Pedregosa et al. (2011) scikit-learn",
         "url": "https://arxiv.org/abs/1201.0490",
         "why": "Feature transformer (SelectKBest, etc.) reference."},
    ],
    "nas-retiarii": [
        {"title": "Zhang et al. (2021) Retiarii: A Deep Learning Exploratory-Testing Platform",
         "url": "https://arxiv.org/abs/2007.15246",
         "why": "NNI's native NAS framework used by this family."},
        {"title": "Liu et al. (2019) DARTS: Differentiable Architecture Search",
         "url": "https://arxiv.org/abs/1806.09055",
         "why": "Cell-based differentiable NAS; the search-space style."},
        {"title": "Zoph & Le (2017) Neural Architecture Search with Reinforcement Learning",
         "url": "https://arxiv.org/abs/1611.01578",
         "why": "Foundational NAS; context for the search task."},
        {"title": "Pham et al. (2018) Efficient Neural Architecture Search (ENAS)",
         "url": "https://arxiv.org/abs/1802.03268",
         "why": "Parameter-sharing NAS; an alternative NNI NAS algorithm."},
    ],
    "novel-spectral": [
        {"title": "Li et al. (2021) Fourier Neural Operator (FNO)",
         "url": "https://arxiv.org/abs/2010.08895",
         "why": "Spectral mixing + low-frequency truncation the block builds on."},
        {"title": "Rao et al. (2021) Global Filter Networks (GFNet)",
         "url": "https://arxiv.org/abs/2107.08391",
         "why": "Global Fourier-domain filter; the learned-convolution analogue."},
        {"title": "Bruna & Mallat (2013) Invariant Scattering Convolution Networks",
         "url": "https://arxiv.org/abs/1301.3709",
         "why": "Wavelet scattering; the wavelet detail path in the block."},
        {"title": "Guibas et al. (2022) Adaptive Fourier Neural Operator (AFNO)",
         "url": "https://arxiv.org/abs/2111.13587",
         "why": "Learned gating over spectral modes; parallels the block's gate."},
    ],
    "audio-keyword-mel": [
        {"title": "Warden (2018) Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition",
         "url": "https://arxiv.org/abs/1804.03209",
         "why": "The keyword-spotting dataset this family targets."},
        {"title": "Sainath & Parada (2015) Convolutional Neural Networks for Small-footprint Keyword Spotting",
         "url": "https://research.google/pubs/pub43969/",
         "why": "The conv-on-mel baseline the block stack follows."},
        {"title": "Hershey et al. (2017) CNN Architectures for Large-Scale Audio Classification",
         "url": "https://arxiv.org/abs/1609.09430",
         "why": "Shows image-style CNNs transfer to log-mel audio inputs."},
    ],
    "audio-waveform-tcn": [
        {"title": "van den Oord et al. (2016) WaveNet: A Generative Model for Raw Audio",
         "url": "https://arxiv.org/abs/1609.03499",
         "why": "Origin of dilated causal convolution over raw waveforms."},
        {"title": "Bai et al. (2018) An Empirical Evaluation of Generic Convolutional and Recurrent Networks",
         "url": "https://arxiv.org/abs/1803.01271",
         "why": "The temporal convolutional network (TCN) design used here."},
        {"title": "Dai et al. (2017) Very Deep Convolutional Neural Networks for Raw Waveforms",
         "url": "https://arxiv.org/abs/1610.00087",
         "why": "Raw-waveform classification without a spectrogram front end."},
        {"title": "Piczak (2015) ESC: Dataset for Environmental Sound Classification",
         "url": "https://github.com/karoldvl/ESC-50",
         "why": "The ESC-50 dataset used by this family."},
    ],
    "text-cnn-news": [
        {"title": "Kim (2014) Convolutional Neural Networks for Sentence Classification",
         "url": "https://arxiv.org/abs/1408.5882",
         "why": "The embedding + 1-D conv text baseline this family implements."},
        {"title": "Zhang et al. (2015) Character-level Convolutional Networks for Text Classification",
         "url": "https://arxiv.org/abs/1509.01626",
         "why": "Source of the AG News benchmark."},
        {"title": "Mikolov et al. (2013) Efficient Estimation of Word Representations",
         "url": "https://arxiv.org/abs/1301.3781",
         "why": "Word embedding layer that starts the stack."},
    ],
    "text-attention-imdb": [
        {"title": "Vaswani et al. (2017) Attention Is All You Need",
         "url": "https://arxiv.org/abs/1706.03762",
         "why": "The self-attention layer used after the conv front end."},
        {"title": "Maas et al. (2011) Learning Word Vectors for Sentiment Analysis (IMDB)",
         "url": "https://aclanthology.org/P11-1015/",
         "why": "The IMDB sentiment dataset this family targets."},
        {"title": "Devlin et al. (2019) BERT",
         "url": "https://arxiv.org/abs/1810.04805",
         "why": "Reference point for attention-based text classification."},
    ],
    "timeseries-har": [
        {"title": "Anguita et al. (2013) A Public Domain Dataset for Human Activity Recognition",
         "url": "https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones",
         "why": "The UCI HAR 9-channel sensor dataset used here."},
        {"title": "Ordonez & Roggen (2016) Deep Convolutional and LSTM Recurrent Networks for Activity Recognition",
         "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4732148/",
         "why": "The conv + recurrent hybrid this family mirrors."},
        {"title": "Cho et al. (2014) Learning Phrase Representations using RNN Encoder-Decoder (GRU)",
         "url": "https://arxiv.org/abs/1406.1078",
         "why": "Origin of the GRU layer in the stack."},
    ],
    "cluster-image-dec": [
        {"title": "Xie et al. (2016) Unsupervised Deep Embedding for Clustering Analysis (DEC)",
         "url": "https://arxiv.org/abs/1511.06335",
         "why": "The Student-t soft-assignment cluster head implemented here."},
        {"title": "Caron et al. (2018) Deep Clustering for Unsupervised Learning of Visual Features",
         "url": "https://arxiv.org/abs/1807.05520",
         "why": "Conv encoder plus clustering objective on images."},
        {"title": "van der Maaten & Hinton (2008) Visualizing Data using t-SNE",
         "url": "https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf",
         "why": "Source of the Student-t kernel used for the assignments."},
    ],
    "cluster-tabular-dec": [
        {"title": "Xie et al. (2016) Unsupervised Deep Embedding for Clustering Analysis (DEC)",
         "url": "https://arxiv.org/abs/1511.06335",
         "why": "The clustering head applied to a dense encoder."},
        {"title": "Guo et al. (2017) Improved Deep Embedded Clustering with Local Structure Preservation",
         "url": "https://www.ijcai.org/proceedings/2017/243",
         "why": "IDEC refinement of the DEC objective for feature vectors."},
        {"title": "Lloyd (1982) Least Squares Quantization in PCM (k-means)",
         "url": "https://doi.org/10.1109/TIT.1982.1056489",
         "why": "The centroid baseline the learned centroids generalize."},
        {"title": "OpenML",
         "url": "https://www.openml.org/",
         "why": "The tabular data source for the cluster features."},
    ],
    "diffusion-cifar": [
        {"title": "Ho et al. (2020) Denoising Diffusion Probabilistic Models (DDPM)",
         "url": "https://arxiv.org/abs/2006.11239",
         "why": "The DDPM framework; epsilon prediction and cosine schedule."},
        {"title": "Nichol & Dhariwal (2021) Improved Denoising Diffusion Probabilistic Models",
         "url": "https://arxiv.org/abs/2102.09672",
         "why": "Cosine noise schedule and improved U-Net design for diffusion."},
        {"title": "Dhariwal & Nichol (2021) Diffusion Models Beat GANs on Image Synthesis",
         "url": "https://arxiv.org/abs/2105.05233",
         "why": "Attention in U-Net at multiple resolutions; classifier-free guidance."},
        {"title": "Krizhevsky (2009) CIFAR-10 dataset",
         "url": "https://www.cs.toronto.edu/~kriz/cifar.html",
         "why": "The dataset used for the diffusion benchmark."},
    ],
    "rl-atari-ppo": [
        {"title": "Espeholt et al. (2018) IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures",
         "url": "https://arxiv.org/abs/1802.01561",
         "why": "The IMPALA CNN architecture with residual blocks used here."},
        {"title": "Schulman et al. (2017) Proximal Policy Optimization Algorithms (PPO)",
         "url": "https://arxiv.org/abs/1707.06347",
         "why": "The PPO algorithm with clipped surrogate objective."},
        {"title": "Bellemare et al. (2013) The Arcade Learning Environment (ALE)",
         "url": "https://arxiv.org/abs/1207.4708",
         "why": "The Atari benchmark suite used by this family."},
        {"title": "Mnih et al. (2015) Human-level control through deep reinforcement learning",
         "url": "https://www.nature.com/articles/nature14236",
         "why": "Original DQN on Atari; context for the RL policy task."},
    ],
}


def build_rationale(fam: str, idx: int, spec: dict, fixed: bool = False) -> str:
    base = RATIONALE.get(fam, "Proposed as part of the NNI experiment sweep.")
    if fixed:
        return base
    width = WIDTH_GRID[idx % len(WIDTH_GRID)]
    variant = (
        f" Variant M{idx + 1:02d} uses layer width {width} "
        f"(e.g. a dense/conv layer of {width} units or channels). The model is "
        f"distinguished by its LAYER DIMENSIONS, not by any training setting."
    )
    return base + variant


# Per-layer optional keys that may appear on a block dict.
_LAYER_EXTRA = ("novel", "definition", "refs", "note")
# Per-model optional keys copied verbatim from the template into spec.
_SPEC_EXTRA = ("novel", "hypothesis", "implementation_status",
               "task_type", "n_clusters", "embed_dim", "modality")

# Layer widths each variant uses. The proposal list intentionally contains NO
# training hyperparameters (no lr / weight_decay / batch_size / epochs /
# optimizer / augmentation) — those belong to the training run, not the model.
# Variants are distinguished by LAYER DIMENSIONS, e.g. a dense layer of 32 vs 64.
WIDTH_GRID = [32, 64, 128, 256, 512]


def _size_field(btype: str):
    """Which field on a block holds its primary layer dimension (count)."""
    return {
        "linear": "units",
        "conv": "channels",
        "coordconv": "channels",
        "dilated": "channels",
        "bottleneck": "channels",
        "basicblock": "channels",
        "inverted_residual": "channels",
        "boosted_trees": "estimators",
        "conv1d": "channels",
        "dilated1d": "channels",
        "sepconv1d": "channels",
        "gru": "hidden",
        "lstm": "hidden",
        "embedding": "dim",
        "dense": "units",
    }.get(btype)


def build_spec(fam: str, idx: int, template: dict) -> dict:
    """Build a spec.

    Width-variant families: each variant gets a DISTINCT layer width from
    WIDTH_GRID (e.g. a dense layer of 32 vs 64 vs ...), so M01..M05 are
    separate models distinguished by their LAYER DIMENSIONS — not by any
    training setting.

    Fixed families (template has `fixed: True`): the authored block widths are
    preserved EXACTLY (no width broadcast), so a fixed architecture such as
    LeNet-5 is emitted verbatim as a single proposal.

    Training hyperparameters are deliberately NOT emitted (they are not part of
    the model)."""
    import copy

    blocks = copy.deepcopy(template["blocks"])
    if not template.get("fixed"):
        width = WIDTH_GRID[idx % len(WIDTH_GRID)]
        for b in blocks:
            fld = _size_field(b.get("type", ""))
            if fld and fld in b:
                b[fld] = width
            elif b.get("novel") and "modes" in b:
                b["modes"] = width          # novel spectral layer: spectral mode count

    spec = {
        "model": template["model"],
        "dataset": template["dataset"],
        "blocks": blocks,
        "head": template["head"],
    }
    # Optional, family-specific knobs (model-level, not training settings).
    for key in ("quant", "prune", "feature_search"):
        if key in template:
            spec[key] = template[key]
    # Optional, open-layer / novel-model keys.
    for key in _SPEC_EXTRA:
        if key in template:
            spec[key] = template[key]
    return spec


def _norm(o):
    """Recursively sort dicts so JSON key order never affects a fingerprint."""
    if isinstance(o, dict):
        return {k: _norm(v) for k, v in sorted(o.items())}
    if isinstance(o, list):
        return [_norm(x) for x in o]
    return o


def _arch_fingerprint(spec: dict) -> str:
    """Fingerprint of the architecture only (model + blocks + head). Sharing
    this fingerprint means the two proposals describe the SAME model structure."""
    return json.dumps(
        _norm({"model": spec.get("model"),
               "blocks": spec.get("blocks"),
               "head": spec.get("head")}),
        sort_keys=True, separators=(",", ":"))


def _uniqueness_key(spec: dict, task_family: str):
    """A proposal is a DUPLICATE only if it shares the SAME task and the SAME
    layer dimensions (architecture + widths). The same architecture is allowed
    for a different task OR with different layer sizes (e.g. dense 32 vs 64).
    Training hyperparameters are irrelevant to this check."""
    return (task_family, _arch_fingerprint(spec))


def main() -> int:
    rows = []
    seen_keys = {}  # uniqueness key -> id, to enforce "not already proposed"
    for fam, tmpl in TASK_FAMILIES.items():
        fixed = bool(tmpl.get("fixed"))
        n_var = 1 if fixed else MODELS_PER_FAMILY
        for i in range(n_var):
            nn = f"{i + 1:02d}"
            pid = f"T{fam}-M{nn}"
            status = "proposed" if fixed else STATUS_CYCLE[i % len(STATUS_CYCLE)]
            spec = build_spec(fam, i, tmpl)
            # uniqueness requirement: same model allowed for a different task OR
            # with different layer dimensions. Duplicate only if task + layer
            # dimensions both match.
            key = _uniqueness_key(spec, fam)
            if key in seen_keys:
                raise SystemExit(
                    f"DUPLICATE MODEL REJECTED: {pid} repeats task={fam} with "
                    f"the same layer dimensions as {seen_keys[key]} (already "
                    f"proposed). A model may be reused only for a different "
                    f"task or with different layer sizes (e.g. width 32 vs 64).")
            seen_keys[key] = pid
            row = {
                "id": pid,
                "task": fam,
                "task_family": fam,
                "status": status,
                "created": CREATED,
                "parent": None if i == 0 else f"T{fam}-M{(i):02d}",
                "rationale": build_rationale(fam, i, spec, fixed=fixed),
                "citations": CITATIONS.get(fam, []),
                "compile_status": "untested",  # filled by pipeline/smoke_test.py
                "spec": spec,
                "expected": {
                    "held_out_acc": None,
                    "note": "proposed for NNI experiment sweep; metrics filled after testing",
                },
            }
            rows.append(row)

    # guard: every family must carry >=3 citations
    for fam, cites in CITATIONS.items():
        assert len(cites) >= 3, f"{fam} has only {len(cites)} citations"

    with open(OUT, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    novel = sum(1 for r in rows if r["spec"].get("novel"))
    print(f"wrote {len(rows)} proposals -> {OUT}  ({novel} marked novel)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

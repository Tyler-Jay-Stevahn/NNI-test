#!/usr/bin/env python3
"""datasets_real.py — real-data loaders for NNI-Test family training.

Each loader returns (train_loader, val_loader, n_classes) where every sample is
a tensor already in the shape build_model.dataset_info(dataset)["shape"] expects,
and labels are in [0, n_classes). Data is downloaded once and cached under
<ROOT>/data/<dataset>/.

Sources:
  mnist, cifar10, cifar100            -> torchvision (real)
  mnist-cluster                      -> torchvision MNIST (real, used as clustering)
  uci-har                            -> UCI HAR zip (real, inertial 9x128 signals)
  ag-news                            -> AG-News CSV mirror (real, 4-class text)
  imdb                               -> aclImdb tar (real, 2-class text)
  esc50                              -> ESC-50 github zip (real WAV -> 1x8000 wave)
  speech-commands                    -> FSDD github zip (real WAV -> 1x64x64 mel)
  imagenet-subset, openml-ctr23,
  openml-cluster, atari-ale           -> LOCAL generated stand-ins (no reachable
                                       external source in this environment);
                                       real-shaped + labeled, clearly marked.
"""
import os
import sys
import json
import gzip
import tarfile
import zipfile
import urllib.request
import io
import math

import torch
from torch.utils.data import Dataset, DataLoader, Subset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
UA = {"User-Agent": "Mozilla/5.0 (NNI-Test dataset fetch)"}


def _get(url, timeout=60):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _cached(path, fetch_fn, timeout=120):
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = fetch_fn()
    with open(path, "wb") as f:
        f.write(data)
    return path


def _basic_spec_dataset(folder, shape, n_classes, n_train=600, n_val=200, seed=0):
    """Generic in-memory Dataset of (tensor(shape), label)."""
    g = torch.Generator().manual_seed(seed)
    n = n_train + n_val
    X = torch.randn(n, *shape, generator=g)
    # inject a class-correlated signal so a real model can learn above chance
    for c in range(n_classes):
        idx = torch.arange(n) % n_classes == c
        X[idx] += (torch.arange(1, n_classes + 1)[c] / n_classes) * torch.ones(shape)
    y = (torch.arange(n) % n_classes).long()
    return _split_xy(X, y, n_train, n_val, shape)


def _split_xy(X, y, n_train, n_val, shape=None):
    tr = TensorXY(X[:n_train], y[:n_train])
    va = TensorXY(X[n_train:n_train + n_val], y[n_train:n_train + n_val])
    return tr, va


class TensorXY(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# ---------------------------------------------------------------------------
# Torchvision datasets
# ---------------------------------------------------------------------------
def _tv(dataset, n_train, n_val, batch):
    from torchvision import datasets, transforms
    if dataset == "mnist" or dataset == "mnist-cluster":
        tfm = transforms.Compose([transforms.ToTensor()])
        train = datasets.MNIST(DATA, train=True, download=True, transform=tfm)
        val = datasets.MNIST(DATA, train=False, download=True, transform=tfm)
        nc = 10
    elif dataset == "cifar10":
        tfm = transforms.Compose([transforms.ToTensor()])
        train = datasets.CIFAR10(DATA, train=True, download=True, transform=tfm)
        val = datasets.CIFAR10(DATA, train=False, download=True, transform=tfm)
        nc = 10
    elif dataset == "cifar100":
        tfm = transforms.Compose([transforms.ToTensor()])
        train = datasets.CIFAR100(DATA, train=True, download=True, transform=tfm)
        val = datasets.CIFAR100(DATA, train=False, download=True, transform=tfm)
        nc = 100
    else:
        raise ValueError(dataset)
    tr = Subset(train, list(range(min(n_train, len(train)))))
    va = Subset(val, list(range(min(n_val, len(val)))))
    return (DataLoader(tr, batch_size=batch, shuffle=True, drop_last=True),
            DataLoader(va, batch_size=batch, shuffle=False, drop_last=False), nc)


# ---------------------------------------------------------------------------
# UCI HAR (timeseries 9x128, 6 classes)
# ---------------------------------------------------------------------------
def _uci_har(n_train, n_val, batch):
    url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
           "00240/UCI%20HAR%20Dataset.zip")
    zip_path = _cached(os.path.join(DATA, "uci-har.zip"), lambda: _get(url, 120))
    base = os.path.join(DATA, "uci-har")
    if not os.path.isdir(os.path.join(base, "UCI HAR Dataset")):
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(base)
    d = os.path.join(base, "UCI HAR Dataset", "train", "Inertial Signals")
    # 9 signals, each file is (n,128)
    sigs = ["body_acc_x", "body_acc_y", "body_acc_z",
            "body_gyro_x", "body_gyro_y", "body_gyro_z",
            "total_acc_x", "total_acc_y", "total_acc_z"]
    import numpy as np
    arrs = []
    for s in sigs:
        a = np.loadtxt(os.path.join(d, s + "_train.txt"))
        arrs.append(torch.tensor(a, dtype=torch.float32))
    Xtr = torch.stack(arrs, 1)  # (n,9,128)
    ytr = torch.tensor(np.loadtxt(os.path.join(base, "UCI HAR Dataset", "train", "y_train.txt")),
                       dtype=torch.long).squeeze() - 1
    d2 = os.path.join(base, "UCI HAR Dataset", "test", "Inertial Signals")
    arrs2 = [torch.tensor(np.loadtxt(os.path.join(d2, s + "_test.txt")), dtype=torch.float32)
             for s in sigs]
    Xte = torch.stack(arrs2, 1)
    yte = torch.tensor(np.loadtxt(os.path.join(base, "UCI HAR Dataset", "test", "y_test.txt")),
                       dtype=torch.long).squeeze() - 1
    tr = TensorXY(Xtr[:n_train], ytr[:n_train])
    va = TensorXY(Xte[:n_val], yte[:n_val])
    return DataLoader(tr, batch_size=batch, shuffle=True, drop_last=True), \
           DataLoader(va, batch_size=batch, shuffle=False, drop_last=False), 6


# ---------------------------------------------------------------------------
# AG-News (text 64, 4 classes)
# ---------------------------------------------------------------------------
def _tok(s, length, vocab=20000):
    toks = [abs(hash(w) % vocab) + 1 for w in s.lower().split() if w.isalpha()]
    if len(toks) >= length:
        return torch.tensor(toks[:length], dtype=torch.long)
    return torch.tensor(toks + [0] * (length - len(toks)), dtype=torch.long)


def _agnews(n_train, n_val, batch):
    url = ("https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/"
           "data/ag_news_csv/train.csv")
    csv = _cached(os.path.join(DATA, "ag_news", "train.csv"), lambda: _get(url, 120))
    import numpy as np
    import csv as _csv
    rows = []
    with open(csv, encoding="utf-8", errors="ignore", newline="") as f:
        for r in _csv.reader(f):
            if len(r) < 3:
                continue
            try:
                c = int(r[0]) - 1
            except ValueError:
                continue
            text = (r[1] + " " + r[2])
            rows.append((c, text))
            if len(rows) >= n_train + n_val:
                break
    X = torch.stack([_tok(t, 64) for _, t in rows])
    y = torch.tensor([c for c, _ in rows], dtype=torch.long)
    tr = TensorXY(X[:n_train], y[:n_train])
    va = TensorXY(X[n_train:n_train + n_val], y[n_train:n_train + n_val])
    return DataLoader(tr, batch_size=batch, shuffle=True, drop_last=True), \
           DataLoader(va, batch_size=batch, shuffle=False, drop_last=False), 4


# ---------------------------------------------------------------------------
# IMDB (text 128, 2 classes)
# ---------------------------------------------------------------------------
def _imdb(n_train, n_val, batch):
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    tgz = _cached(os.path.join(DATA, "aclImdb.tar.gz"), lambda: _get(url, 240))
    base = os.path.join(DATA, "aclImdb")
    if not os.path.isdir(base):
        with tarfile.open(tgz) as t:
            t.extractall(DATA)
    rows = []
    for lab, sub in ((1, "pos"), (0, "neg")):
        d = os.path.join(base, "train", sub)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".txt"):
                continue
            with open(os.path.join(d, fn), encoding="utf-8", errors="ignore") as f:
                rows.append((lab, f.read()))
            if len(rows) >= (n_train + n_val) // 2 * 2:
                break
    X = torch.stack([_tok(t, 128) for _, t in rows])
    y = torch.tensor([lab for lab, _ in rows], dtype=torch.long)
    tr = TensorXY(X[:n_train], y[:n_train])
    va = TensorXY(X[n_train:n_train + n_val], y[n_train:n_train + n_val])
    return DataLoader(tr, batch_size=batch, shuffle=True, drop_last=True), \
           DataLoader(va, batch_size=batch, shuffle=False, drop_last=False), 2


# ---------------------------------------------------------------------------
# ESC-50 (audio-wave 1x8000, 10 classes) -- decode WAV via stdlib wave
# ---------------------------------------------------------------------------
def _wav_to_tensor(path, target_len=8000):
    import wave
    import numpy as np
    with wave.open(path, "rb") as w:
        n = w.getnframes()
        raw = w.readframes(n)
        sig = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if len(sig) > target_len:
        # naive downsample to target_len
        idx = np.linspace(0, len(sig) - 1, target_len).astype(int)
        sig = sig[idx]
    elif len(sig) < target_len:
        sig = np.pad(sig, (0, target_len - len(sig)))
    return torch.tensor(sig[:target_len], dtype=torch.float32).unsqueeze(0)


def _esc50(n_train, n_val, batch, n_classes=10):
    url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
    zp = _cached(os.path.join(DATA, "esc50.zip"), lambda: _get(url, 240))
    base = os.path.join(DATA, "esc50", "ESC-50-master", "audio")
    if not os.path.isdir(base) or not os.listdir(base):
        with zipfile.ZipFile(zp) as z:
            z.extractall(os.path.join(DATA, "esc50"))
    rows = []
    for fn in sorted(os.listdir(base)):
        if not fn.endswith(".wav"):
            continue
        parts = fn[:-4].split("-")
        if len(parts) < 4:
            continue
        try:
            cid = int(parts[3]) - 1
        except ValueError:
            continue
        rows.append((cid, os.path.join(base, fn)))
        if len(rows) >= n_train + n_val:
            break
    X = torch.stack([_wav_to_tensor(p) for _, p in rows])
    y = torch.tensor([c for c, _ in rows], dtype=torch.long) % n_classes
    tr = TensorXY(X[:n_train], y[:n_train])
    va = TensorXY(X[n_train:n_train + n_val], y[n_train:n_train + n_val])
    return DataLoader(tr, batch_size=batch, shuffle=True, drop_last=True), \
           DataLoader(va, batch_size=batch, shuffle=False, drop_last=False), n_classes


# ---------------------------------------------------------------------------
# Speech commands (FSDD) -> mel 1x64x64, 10 classes (digits 0-9)
# ---------------------------------------------------------------------------
def _mel_spec(wav_tensor, n_mels=64, n_frames=64, sr=8000):
    # simple magnitude spectrogram via torch.stft, log-scaled, cropped
    x = wav_tensor.squeeze(0)
    if x.numel() < 256:
        x = torch.nn.functional.pad(x, (0, 256 - x.numel()))
    spec = torch.stft(x, n_fft=128, hop_length=32, win_length=128,
                      window=torch.hann_window(128), return_complex=True)
    mag = spec.abs()  # (freq, time)
    mag = torch.log1p(mag)
    # downsample freq/time to (n_mels, n_frames)
    if mag.size(0) > n_mels:
        mag = mag[:n_mels, :]
    else:
        mag = torch.nn.functional.pad(mag, (0, 0, 0, n_mels - mag.size(0)))
    if mag.size(1) > n_frames:
        mag = mag[:, :n_frames]
    else:
        mag = torch.nn.functional.pad(mag, (0, n_frames - mag.size(1)))
    return mag.unsqueeze(0).float()  # (1,64,64)


def _speech_commands(n_train, n_val, batch):
    url = "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/master.zip"
    zp = _cached(os.path.join(DATA, "fsdd.zip"), lambda: _get(url, 120))
    base = os.path.join(DATA, "fsdd", "free-spoken-digit-dataset-master", "recordings")
    if not os.path.isdir(base):
        with zipfile.ZipFile(zp) as z:
            z.extractall(os.path.join(DATA, "fsdd"))
    rows = []
    for fn in sorted(os.listdir(base)):
        if not fn.endswith(".wav"):
            continue
        dig = int(fn.split("_")[0])
        rows.append((dig, os.path.join(base, fn)))
        if len(rows) >= n_train + n_val:
            break
    X = torch.stack([_mel_spec(_wav_to_tensor(p, 8000)) for _, p in rows])
    y = torch.tensor([c for c, _ in rows], dtype=torch.long)
    tr = TensorXY(X[:n_train], y[:n_train])
    va = TensorXY(X[n_train:n_train + n_val], y[n_train:n_train + n_val])
    return DataLoader(tr, batch_size=batch, shuffle=True, drop_last=True), \
           DataLoader(va, batch_size=batch, shuffle=False, drop_last=False), 10


# ---------------------------------------------------------------------------
# Local generated stand-ins (no reachable external source)
# ---------------------------------------------------------------------------
def _local_standin(dataset, shape, n_classes, n_train, n_val, batch):
    tr, va = _basic_spec_dataset(dataset, shape, n_classes, n_train, n_val,
                                 seed=hash(dataset) % 1000)
    return (DataLoader(tr, batch_size=batch, shuffle=True, drop_last=True),
            DataLoader(va, batch_size=batch, shuffle=False, drop_last=False), n_classes)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
def get_loaders(dataset, task_type=None, n_classes=None, n_train=600, n_val=200, batch=64):
    """Return (train_loader, val_loader, n_classes).

    n_classes should come from build_model.output_size(spec) so labels match the
    model head. If None, a dataset-appropriate default is used.
    """
    ds = dataset
    if ds in ("mnist", "mnist-cluster", "cifar10", "cifar100"):
        tr, va, nc = _tv(ds, n_train, n_val, batch)
        return tr, va, (n_classes or nc)
    if ds == "uci-har":
        tr, va, nc = _uci_har(n_train, n_val, batch)
        return tr, va, (n_classes or nc)
    if ds == "ag-news":
        tr, va, nc = _agnews(n_train, n_val, batch)
        return tr, va, (n_classes or nc)
    if ds == "imdb":
        tr, va, nc = _imdb(n_train, n_val, batch)
        return tr, va, (n_classes or nc)
    if ds == "esc50":
        tr, va, nc = _esc50(n_train, n_val, batch, n_classes=(n_classes or 10))
        return tr, va, nc
    if ds == "speech-commands":
        tr, va, nc = _speech_commands(n_train, n_val, batch)
        return tr, va, (n_classes or nc)
    if ds in ("imagenet-subset",):
        return _local_standin(ds, (3, 16, 16), n_classes or 10, n_train, n_val, batch)
    if ds == "openml-ctr23":
        # tabular dataset: match build_model.DATASETS["openml-ctr23"]["shape"]=(20,)
        return _local_standin(ds, (20,), n_classes or 2, n_train, n_val, batch)
    if ds == "openml-cluster":
        return _local_standin(ds, (20,), n_classes or 8, n_train, n_val, batch)
    if ds == "atari-ale":
        # Local generated stand-in: no reachable Atari ROM mirror in this env.
        # Real-shaped (4x84x84 stacked frames) + class-correlated signal so the
        # IMPALA CNN stem can compile and learn above chance. Marked stand-in.
        return _local_standin(ds, (4, 84, 84), n_classes or 18, n_train, n_val, batch)
    raise ValueError(f"no real loader for dataset {ds!r}")


if __name__ == "__main__":
    import sys
    for d in sys.argv[1:] or ["cifar10", "mnist-cluster"]:
        tr, va, nc = get_loaders(d, 64, 32, 8)
        x, y = next(iter(tr))
        print(f"{d}: n_classes={nc} batch_x={tuple(x.shape)} y={y.tolist()[:4]}")

"""
dataset.py

PyTorch Dataset for landmark sequences extracted from ASL videos.
Handles variable-length sequences via padding/truncation and
provides augmentation for training.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Augmentation helpers (operate on numpy arrays, shape T x F)
# ---------------------------------------------------------------------------

def time_warp(seq: np.ndarray, sigma: float = 0.15) -> np.ndarray:
    """Randomly stretch/compress time via interpolation."""
    T = len(seq)
    orig_steps = np.arange(T)
    # Random warp knots
    knot = max(2, T // 8)
    warp_steps = np.sort(np.random.choice(T, knot, replace=False))
    warp_values = warp_steps + np.random.randn(knot) * sigma * T
    warp_values = np.clip(warp_values, 0, T - 1)
    warped_steps = np.interp(orig_steps, warp_steps, warp_values)
    warped_steps = np.clip(warped_steps, 0, T - 1)
    new_seq = np.array([
        seq[int(s)] * (1 - (s % 1)) + seq[min(int(s) + 1, T - 1)] * (s % 1)
        for s in warped_steps
    ])
    return new_seq


def add_noise(seq: np.ndarray, sigma: float = 0.005) -> np.ndarray:
    """Add small Gaussian noise to landmark coordinates."""
    return seq + np.random.randn(*seq.shape) * sigma


def drop_frames(seq: np.ndarray, drop_prob: float = 0.05) -> np.ndarray:
    """Randomly drop frames and fill with neighbours."""
    T = len(seq)
    mask = np.random.rand(T) > drop_prob
    if mask.all() or not mask.any():
        return seq
    indices = np.where(mask)[0]
    # Interpolate dropped frames from nearest kept frame
    new_seq = seq.copy()
    for i in range(T):
        if not mask[i]:
            nearest = indices[np.argmin(np.abs(indices - i))]
            new_seq[i] = seq[nearest]
    return new_seq


def mirror_hands(seq: np.ndarray) -> np.ndarray:
    """
    Flip left/right hands (simulates left-handed signers).
    Assumes feature layout: [pose(18), lh(63), rh(63)]
    """
    seq = seq.copy()
    lh = seq[:, 18:81].copy()
    rh = seq[:, 81:144].copy()
    seq[:, 18:81] = rh
    seq[:, 81:144] = lh
    # Flip x-coordinates for all landmarks
    seq[:, 0::3] = 1.0 - seq[:, 0::3]
    return seq


def augment(seq: np.ndarray) -> np.ndarray:
    if random.random() < 0.8:
        seq = time_warp(seq)
    if random.random() < 0.7:
        seq = add_noise(seq)
    if random.random() < 0.5:
        seq = drop_frames(seq)
    if random.random() < 0.3:
        seq = mirror_hands(seq)
    return seq


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ASLLandmarkDataset(Dataset):
    """
    Expects data structured as:
        landmark_dir/
            label_map.json
            <gloss>/
                <video_stem>.npy   # shape (T, 138)

    Args:
        landmark_dir: Path to extracted landmarks root
        split:        'train' | 'val' | 'test'
        max_len:      Pad/truncate all sequences to this length
        augment:      Apply augmentation (training only)
        val_frac:     Fraction of data held out for val (per gloss)
        test_frac:    Fraction held out for test (per gloss)
        seed:         Random seed for reproducible splits
    """

    def __init__(
        self,
        landmark_dir: Path,
        split: str = "train",
        max_len: int = 128,
        augment: bool = False,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        seed: int = 42,
    ):
        self.landmark_dir = Path(landmark_dir)
        self.split = split
        self.max_len = max_len
        self.do_augment = augment

        with open(self.landmark_dir / "label_map.json") as f:
            self.label_map = json.load(f)
        self.num_classes = len(self.label_map)

        self.samples = self._build_split(val_frac, test_frac, seed)

    def _build_split(self, val_frac, test_frac, seed) -> list[tuple[Path, int]]:
        rng = random.Random(seed)
        samples = []

        for gloss, label_idx in self.label_map.items():
            gloss_dir = self.landmark_dir / gloss
            if not gloss_dir.exists():
                continue
            files = sorted(gloss_dir.glob("*.npy"))
            rng.shuffle(files)

            n = len(files)
            n_test = max(1, int(n * test_frac))
            n_val = max(1, int(n * val_frac))

            if self.split == "test":
                chosen = files[:n_test]
            elif self.split == "val":
                chosen = files[n_test:n_test + n_val]
            else:
                chosen = files[n_test + n_val:]

            samples.extend((f, label_idx) for f in chosen)

        return samples

    def _pad_or_truncate(self, seq: np.ndarray) -> tuple[np.ndarray, int]:
        """Returns (padded_seq, valid_length)."""
        T, F = seq.shape
        if T >= self.max_len:
            return seq[:self.max_len], self.max_len
        pad = np.zeros((self.max_len - T, F), dtype=np.float32)
        return np.concatenate([seq, pad], axis=0), T

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        seq = np.load(path).astype(np.float32)

        if self.do_augment:
            seq = augment(seq)

        seq, valid_len = self._pad_or_truncate(seq)

        # Key padding mask: True = ignore this position
        pad_mask = torch.zeros(self.max_len, dtype=torch.bool)
        pad_mask[valid_len:] = True

        return (
            torch.from_numpy(seq),        # (max_len, 138)
            pad_mask,                      # (max_len,)
            torch.tensor(label, dtype=torch.long),
        )


def make_dataloaders(
    landmark_dir: Path,
    batch_size: int = 32,
    max_len: int = 128,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = ASLLandmarkDataset(landmark_dir, split="train", max_len=max_len,
                                   augment=True, seed=seed)
    val_ds   = ASLLandmarkDataset(landmark_dir, split="val",   max_len=max_len,
                                   augment=False, seed=seed)
    test_ds  = ASLLandmarkDataset(landmark_dir, split="test",  max_len=max_len,
                                   augment=False, seed=seed)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)

    print(f"Dataset splits — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")
    print(f"Classes: {train_ds.num_classes}")
    return train_dl, val_dl, test_dl
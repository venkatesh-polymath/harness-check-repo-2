"""
Togglable-Signal CIFAR-10 Dataset.

Two spurious signals:
  1. background_color: the entire image is tinted with a class-correlated hue
  2. color_patch: a small 4×4 solid-color patch is placed in the top-left corner

Both are controllable via:
  - spurious_rate: fraction of training samples where signal is class-correlated
  - enabled: whether the signal is rendered at all (for signal-ON vs signal-OFF test forks)
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as T
from PIL import Image
import colorsys


# 10 distinct hues (degrees 0-360) for 10 CIFAR classes
# background tint colors (spaced 36 degrees apart)
BG_HUES = [i * 36.0 for i in range(10)]  # 0, 36, 72, ..., 324

# Patch colors for 10 classes (complementary set — shifted 18 degrees)
PATCH_HUES = [(i * 36.0 + 18.0) % 360 for i in range(10)]


def hue_to_rgb(hue_deg: float, saturation: float = 0.8, value: float = 0.9):
    """Convert HSV hue (degrees) to RGB in [0,255]."""
    h = hue_deg / 360.0
    r, g, b = colorsys.hsv_to_rgb(h, saturation, value)
    return int(r * 255), int(g * 255), int(b * 255)


def apply_background_tint(img_np: np.ndarray, hue_deg: float, alpha: float = 0.35) -> np.ndarray:
    """Blend the image with a solid-color background tint.
    img_np: H×W×3 uint8 array.
    Returns: H×W×3 uint8 array.
    """
    r, g, b = hue_to_rgb(hue_deg, saturation=0.6, value=0.95)
    tint = np.array([r, g, b], dtype=np.float32)
    blended = img_np.astype(np.float32) * (1 - alpha) + tint * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def apply_color_patch(img_np: np.ndarray, hue_deg: float, patch_size: int = 4) -> np.ndarray:
    """Place a solid-color patch in the top-left corner.
    img_np: H×W×3 uint8 array.
    Returns: H×W×3 uint8 array.
    """
    img_np = img_np.copy()
    r, g, b = hue_to_rgb(hue_deg, saturation=1.0, value=1.0)
    img_np[:patch_size, :patch_size, 0] = r
    img_np[:patch_size, :patch_size, 1] = g
    img_np[:patch_size, :patch_size, 2] = b
    return img_np


class TogglableCIFAR10(Dataset):
    """
    CIFAR-10 with two togglable spurious signals.

    Parameters
    ----------
    root : str            – path to CIFAR-10 root (contains cifar-10-batches-py/)
    train : bool          – train or test split
    spurious_rate : float – fraction of examples where spurious signal is class-correlated
                           (remainder get a random class's color signal); shared rate for
                           bg and patch when bg_spurious_rate/patch_spurious_rate not set.
    bg_enabled : bool     – whether background-color tint is rendered
    patch_enabled : bool  – whether color-patch is rendered
    seed : int            – RNG seed for determining which examples get spurious signal
    subset_size : int     – if > 0, subsample this many examples (for quick probe runs)
    transform             – additional torchvision transforms applied AFTER signal injection
    bg_spurious_rate : float or None  – independent spurious rate for BG signal only.
                           When not None, BG and patch get INDEPENDENT Bernoulli draws,
                           enabling balanced 4-group WGA evaluation sets. Use 0.5 to get
                           ~25% in each of groups 0-3.
    patch_spurious_rate : float or None – independent spurious rate for patch signal only.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        spurious_rate: float = 1.0,
        bg_enabled: bool = True,
        patch_enabled: bool = True,
        seed: int = 0,
        subset_size: int = 0,
        transform=None,
        return_group: bool = False,
        bg_spurious_rate: float = None,
        patch_spurious_rate: float = None,
    ):
        self.train = train
        self.spurious_rate = spurious_rate
        self.bg_enabled = bg_enabled
        self.patch_enabled = patch_enabled
        self.seed = seed
        self.transform = transform
        self.return_group = return_group

        # Load raw CIFAR-10
        base = torchvision.datasets.CIFAR10(root, train=train, download=False)
        images = np.array(base.data)        # N×32×32×3 uint8
        labels = np.array(base.targets)     # N,

        # Optional subset
        if subset_size > 0 and subset_size < len(images):
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(images), subset_size, replace=False)
            images = images[idx]
            labels = labels[idx]

        n = len(images)
        rng = np.random.RandomState(seed + 1000)

        if bg_spurious_rate is None and patch_spurious_rate is None:
            # Original behavior: single shared spurious_rate for both bg and patch
            # (RNG order preserved for backward compat)
            is_spurious_correlated = rng.rand(n) < spurious_rate
            rand_bg_class = rng.randint(0, 10, size=n)
            rand_patch_class = rng.randint(0, 10, size=n)
            self.bg_class = np.where(is_spurious_correlated, labels, rand_bg_class)
            self.patch_class = np.where(is_spurious_correlated, labels, rand_patch_class)
        else:
            # NEW: independent spurious rates for bg and patch
            # Allows creating balanced 4-group WGA evaluation sets
            bg_rate = bg_spurious_rate if bg_spurious_rate is not None else spurious_rate
            patch_rate = patch_spurious_rate if patch_spurious_rate is not None else spurious_rate
            rand_bg_class = rng.randint(0, 10, size=n)
            rand_patch_class = rng.randint(0, 10, size=n)
            is_bg_correlated = rng.rand(n) < bg_rate
            is_patch_correlated = rng.rand(n) < patch_rate
            self.bg_class = np.where(is_bg_correlated, labels, rand_bg_class)
            self.patch_class = np.where(is_patch_correlated, labels, rand_patch_class)

        # Precompute spurious-correlation group labels (for WGA)
        # group encoding: (bg_matches_label << 1) | (patch_matches_label)
        # Group 0 (00): neither signal matches label
        # Group 1 (01): only patch matches label
        # Group 2 (10): only bg matches label
        # Group 3 (11): both signals match label
        bg_matches = (self.bg_class == labels).astype(int)
        patch_matches = (self.patch_class == labels).astype(int)
        self.group_labels = bg_matches * 2 + patch_matches  # 0,1,2,3
        self.bg_matches = bg_matches    # stored for binary WGA analysis
        self.patch_matches = patch_matches

        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].copy()   # H×W×3 uint8

        # Apply spurious signals
        if self.bg_enabled:
            img = apply_background_tint(img, BG_HUES[self.bg_class[idx]])
        if self.patch_enabled:
            img = apply_color_patch(img, PATCH_HUES[self.patch_class[idx]])

        img_pil = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img_pil)
        else:
            img = T.ToTensor()(img_pil)

        label = int(self.labels[idx])
        if self.return_group:
            group = int(self.group_labels[idx])
            return img, label, group
        return img, label


def get_hsv_histogram(img_np: np.ndarray, bins: int = 16) -> np.ndarray:
    """Compute a per-channel HSV histogram for a single H×W×3 uint8 image."""
    from PIL import Image as PILImage
    import cv2
    img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    h_hist = np.histogram(img_hsv[:, :, 0], bins=bins, range=(0, 180))[0]
    s_hist = np.histogram(img_hsv[:, :, 1], bins=bins, range=(0, 256))[0]
    v_hist = np.histogram(img_hsv[:, :, 2], bins=bins, range=(0, 256))[0]
    feat = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
    return feat / (feat.sum() + 1e-8)


def get_patch_histogram(img_np: np.ndarray, patch_size: int = 4, bins: int = 16) -> np.ndarray:
    """Compute HSV histogram only on the top-left patch region."""
    patch = img_np[:patch_size, :patch_size, :]
    return get_hsv_histogram(patch, bins=bins)


# Standard CIFAR-10 normalization
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)

TRAIN_TRANSFORM = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(CIFAR_MEAN, CIFAR_STD),
])

TEST_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(CIFAR_MEAN, CIFAR_STD),
])

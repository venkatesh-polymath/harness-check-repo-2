#!/usr/bin/env python3
"""
Experiment: Online Mid-Training Invariance Auditor via Unlabeled Feature-Tail Consistency Signal
Round: refine-02 (probe)

CRITICAL FIX from main-01:
  main-01 skipped CIFAR-10-C mCE evaluation entirely.
  This round evaluates BOTH baseline (FixMatch+RandAugment) and method
  (FixMatch+OnlineAuditor) on CIFAR-10-C mCE using 5 representative
  corruption types × 5 severities.

Probe parameters (same as main-01 for comparability):
  - 50 labeled / class  (500 total labeled)
  - 4 000 unlabeled training samples
  - 500 unlabeled audit pool
  - 30 training epochs
  - 3 seeds

Sanity gates (re-run to confirm pipeline integrity):
  1. Fixed-seed reproducibility
  2. CE loss at init ≈ log(10) ≈ 2.3026
  3. Input-independent baseline accuracy ≈ 10 %
  4. Overfit one batch to CE < 0.01 within 500 SGD steps
"""

import os, sys, json, random, time, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────────────────

class Config:
    # Probe scale (same as main-01)
    LABELED_PER_CLASS  = 50
    UNLABELED_SIZE     = 4000
    AUDIT_POOL_SIZE    = 500
    NUM_EPOCHS         = 30
    LABELED_BATCH      = 32
    UNLABELED_BATCH    = 128

    # FixMatch
    FM_THRESHOLD = 0.95
    FM_LAMBDA_U  = 1.0

    # Optimizer
    LR           = 0.03
    MOMENTUM     = 0.9
    WEIGHT_DECAY = 5e-4

    # Auditor
    AUDIT_INTERVAL         = 10     # run audit every N epochs
    TAIL_PERCENTILE        = 0.20   # top-20% highest-entropy = tail
    CONSISTENCY_THRESHOLD  = 0.82   # flag op if cosine-sim < this
    READMIT_EPOCHS         = 10     # re-admit op after N epochs
    AUDIT_MAGNITUDE        = 9      # severity (0-10) for consistency check

    # Multi-seed
    SEEDS = [42, 123, 456]

    # CIFAR-10-C eval (representative subset)
    # 5 corruptions × 5 severities = 25 (c,s) pairs
    CIFAR10C_TYPES = [
        "gaussian_noise",   # general noise
        "contrast",         # ← known-harmful op in auditor
        "brightness",       # ← known-harmful-adjacent in auditor
        "gaussian_blur",    # tests sharpness robustness (inverse of sharpness aug)
        "pixelate",         # tests spatial fidelity
    ]
    CIFAR10C_SEVERITIES = [1, 2, 3, 4, 5]

    # Paths
    DATA_DIR    = "/opt/datasets"
    RESULTS_DIR = "/workspace/results/refine-02"
    WEIGHTS_DIR = "/workspace/_weights"

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()

# ────────────────────────────────────────────────────────────────────────────
# NORMALISATION CONSTANTS
# ────────────────────────────────────────────────────────────────────────────

MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2023, 0.1994, 0.2010)

NORM_T = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

# ────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE — ResNet-8 (unchanged from main-01)
# ────────────────────────────────────────────────────────────────────────────

class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(out_c)
        self.c2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(out_c)
        self.skip = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.skip = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x):
        out = F.relu(self.b1(self.c1(x)))
        out = self.b2(self.c2(out))
        return F.relu(out + self.skip(x))


class ResNet8(nn.Module):
    """8 weight layers: 1 stem conv + 3×2 residual + 1 FC."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem   = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(),
        )
        self.layer1 = BasicBlock(16, 16)
        self.layer2 = BasicBlock(16, 32, stride=2)
        self.layer3 = BasicBlock(32, 64, stride=2)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(64, num_classes)

    def forward(self, x, return_feat=False):
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.pool(h)
        feat = h.view(h.size(0), -1)
        logit = self.fc(feat)
        return (logit, feat) if return_feat else logit


# ────────────────────────────────────────────────────────────────────────────
# AUGMENTATION POOL (RandAugment-style)
# ────────────────────────────────────────────────────────────────────────────

OPERATIONS = [
    "autocontrast",   # 0
    "equalize",       # 1
    "color",          # 2 ← known harmful (saturation)
    "contrast",       # 3 ← known harmful
    "brightness",     # 4
    "sharpness",      # 5 ← known harmful
    "shear_x",        # 6
    "translate_x",    # 7
    "rotate",         # 8
    "posterize",      # 9
]
KNOWN_HARMFUL = {"contrast", "color", "sharpness"}


def _apply_op_pil(img_pil, op_name, magnitude):
    """Apply one named augmentation op to a PIL image."""
    m = magnitude / 10.0
    try:
        if op_name == "autocontrast":
            return TF.autocontrast(img_pil)
        elif op_name == "equalize":
            return TF.equalize(img_pil)
        elif op_name == "color":
            return ImageEnhance.Color(img_pil).enhance(1 + m * 1.8)
        elif op_name == "contrast":
            return ImageEnhance.Contrast(img_pil).enhance(1 + m * 1.8)
        elif op_name == "brightness":
            return ImageEnhance.Brightness(img_pil).enhance(1 + m * 0.9)
        elif op_name == "sharpness":
            return ImageEnhance.Sharpness(img_pil).enhance(1 + m * 8.0)
        elif op_name == "shear_x":
            return TF.affine(img_pil, angle=0, translate=[0, 0], scale=1,
                             shear=[m * 30, 0])
        elif op_name == "translate_x":
            return TF.affine(img_pil, angle=0, translate=[int(m * 10), 0],
                             scale=1, shear=[0, 0])
        elif op_name == "rotate":
            return TF.rotate(img_pil, m * 30)
        elif op_name == "posterize":
            return TF.posterize(img_pil, max(1, int(8 - m * 4)))
    except Exception:
        pass
    return img_pil


class RandAugPool:
    """Randomly sample N active ops and apply them to a PIL image."""

    def __init__(self, n=2, magnitude=9):
        self.n = n
        self.magnitude = magnitude
        self.active    = set(range(len(OPERATIONS)))
        self.history   = []

    def disable(self, op_name, epoch):
        idx = OPERATIONS.index(op_name)
        self.active.discard(idx)
        self.history.append((epoch, op_name))

    def enable(self, op_name):
        idx = OPERATIONS.index(op_name)
        self.active.add(idx)

    def active_names(self):
        return [OPERATIONS[i] for i in sorted(self.active)]

    def __call__(self, img_pil):
        pool = self.active_names()
        if not pool:
            return img_pil
        chosen = random.sample(pool, min(self.n, len(pool)))
        for op in chosen:
            img_pil = _apply_op_pil(img_pil, op, self.magnitude)
        return img_pil


# ────────────────────────────────────────────────────────────────────────────
# CIFAR-10-C CORRUPTION FUNCTIONS
# ────────────────────────────────────────────────────────────────────────────
# Implemented from scratch — no external imagecorruptions package required.
# Severity levels follow the CIFAR-10-C benchmark conventions (1=mild, 5=severe).

def _corrupt_gaussian_noise(img_pil, severity):
    """Add Gaussian noise (CIFAR-10-C type: gaussian_noise)."""
    # stddev parameters calibrated to match benchmark approximate severity
    stds = [0.04, 0.06, 0.08, 0.09, 0.10]
    std = stds[severity - 1]
    arr = np.array(img_pil).astype(np.float32) / 255.0
    rng = np.random.RandomState(severity)  # deterministic per severity
    noise = rng.randn(*arr.shape).astype(np.float32) * std
    out = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((out * 255).astype(np.uint8))


def _corrupt_brightness(img_pil, severity):
    """Reduce brightness (CIFAR-10-C type: brightness)."""
    # factors < 1 = darker image
    factors = [0.88, 0.79, 0.68, 0.55, 0.40]
    return ImageEnhance.Brightness(img_pil).enhance(factors[severity - 1])


def _corrupt_contrast(img_pil, severity):
    """Reduce contrast (CIFAR-10-C type: contrast)."""
    factors = [0.80, 0.65, 0.50, 0.35, 0.20]
    return ImageEnhance.Contrast(img_pil).enhance(factors[severity - 1])


def _corrupt_gaussian_blur(img_pil, severity):
    """Gaussian blur (CIFAR-10-C type: defocus_blur / gaussian_blur)."""
    radii = [1.0, 1.5, 2.0, 2.5, 3.0]
    return img_pil.filter(ImageFilter.GaussianBlur(radius=radii[severity - 1]))


def _corrupt_pixelate(img_pil, severity):
    """Pixelate by downscaling then upscaling (CIFAR-10-C type: pixelate)."""
    sizes = [28, 24, 20, 16, 12]
    size = sizes[severity - 1]
    small = img_pil.resize((size, size), Image.BOX)
    return small.resize((32, 32), Image.NEAREST)


CORRUPTION_FNS = {
    "gaussian_noise": _corrupt_gaussian_noise,
    "brightness":     _corrupt_brightness,
    "contrast":       _corrupt_contrast,
    "gaussian_blur":  _corrupt_gaussian_blur,
    "pixelate":       _corrupt_pixelate,
}


def apply_corruption(img_pil, corruption_name, severity):
    """Apply a single CIFAR-10-C corruption to a PIL image."""
    fn = CORRUPTION_FNS.get(corruption_name)
    if fn is None:
        raise ValueError(f"Unknown corruption: {corruption_name}")
    return fn(img_pil, severity)


def eval_cifar10c(model, raw_test_ds, device, corruptions=None, severities=None):
    """
    Evaluate model on CIFAR-10-C corruptions.

    Args:
        model: trained ResNet8
        raw_test_ds: torchvision CIFAR-10 test dataset (no transform, returns PIL)
        device: torch device
        corruptions: list of corruption names (default: cfg.CIFAR10C_TYPES)
        severities: list of severity levels (default: cfg.CIFAR10C_SEVERITIES)

    Returns:
        mCE (float): mean corruption error (mean error rate over all (c,s) pairs)
        per_cs (dict): {(corruption, severity): error_rate}
        per_c_mean (dict): {corruption: mean_error_rate_over_severities}
    """
    if corruptions is None:
        corruptions = cfg.CIFAR10C_TYPES
    if severities is None:
        severities = cfg.CIFAR10C_SEVERITIES

    model.eval()
    per_cs = {}

    for corruption in corruptions:
        for severity in severities:
            correct = 0
            total   = 0
            batch_imgs = []
            batch_lbls = []

            for idx in range(len(raw_test_ds)):
                img_pil, label = raw_test_ds[idx]
                corr_img = apply_corruption(img_pil, corruption, severity)
                batch_imgs.append(NORM_T(corr_img))
                batch_lbls.append(label)

                if len(batch_imgs) == 512 or idx == len(raw_test_ds) - 1:
                    with torch.no_grad():
                        x = torch.stack(batch_imgs).to(device)
                        y = torch.tensor(batch_lbls).to(device)
                        preds = model(x).argmax(1)
                        correct += preds.eq(y).sum().item()
                        total   += y.size(0)
                    batch_imgs, batch_lbls = [], []

            error_rate = 1.0 - (correct / total)
            per_cs[(corruption, severity)] = float(error_rate)

    # Mean error over all (c, s) pairs = raw mCE (no AlexNet normalization)
    mCE = float(np.mean(list(per_cs.values())))

    # Per-corruption mean (useful for analysis)
    per_c_mean = {}
    for c in corruptions:
        vals = [per_cs[(c, s)] for s in severities]
        per_c_mean[c] = float(np.mean(vals))

    return mCE, per_cs, per_c_mean


# ────────────────────────────────────────────────────────────────────────────
# DATASETS
# ────────────────────────────────────────────────────────────────────────────

_WEAK_PIL = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()])


class RawCIFAR10:
    """Thin wrapper returning raw PIL images (no transform)."""
    def __init__(self, root, train=True):
        self.ds = torchvision.datasets.CIFAR10(root, train=train, download=True)

    def __len__(self): return len(self.ds)

    def __getitem__(self, i):
        return self.ds[i]


class LabeledSet(Dataset):
    def __init__(self, raw, indices):
        self.raw     = raw
        self.indices = indices

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        img, lbl = self.raw[self.indices[i]]
        img = _WEAK_PIL(img)
        return NORM_T(img), lbl


class UnlabeledSet(Dataset):
    def __init__(self, raw, indices, aug_pool):
        self.raw      = raw
        self.indices  = indices
        self.aug_pool = aug_pool

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        img_pil, _ = self.raw[self.indices[i]]
        weak   = _WEAK_PIL(img_pil)
        strong = self.aug_pool(weak)
        return NORM_T(weak), NORM_T(strong)


class AuditSet(Dataset):
    def __init__(self, raw, indices):
        self.raw     = raw
        self.indices = indices

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        img_pil, lbl = self.raw[self.indices[i]]
        return NORM_T(img_pil), lbl, i


def make_ssl_split(root, labeled_per_class, unlabeled_size, audit_size, seed=42):
    """Return (raw_train, labeled_idx, unlabeled_idx, audit_idx)."""
    raw = RawCIFAR10(root, train=True)
    targets = np.array(raw.ds.targets)
    rng = np.random.RandomState(seed)

    labeled = []
    for c in range(10):
        cidx = np.where(targets == c)[0]
        labeled.extend(rng.choice(cidx, labeled_per_class, replace=False).tolist())
    labeled_set = set(labeled)

    pool = [i for i in range(len(raw)) if i not in labeled_set]
    rng.shuffle(pool)

    return raw, labeled, pool[:unlabeled_size], pool[unlabeled_size: unlabeled_size + audit_size]


# ────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ────────────────────────────────────────────────────────────────────────────

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


def eval_acc(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += pred.eq(y).sum().item()
            total   += y.size(0)
    return correct / total


# ────────────────────────────────────────────────────────────────────────────
# SANITY GATES (re-run; main-01 confirmed all passing)
# ────────────────────────────────────────────────────────────────────────────

def gate1_reproducibility():
    print("\n=== GATE 1 : fixed-seed reproducibility ===")
    raw = RawCIFAR10(cfg.DATA_DIR)
    x   = torch.stack([NORM_T(raw[i][0]) for i in range(64)]).to(cfg.DEVICE)
    y   = torch.tensor([raw[i][1] for i in range(64)]).to(cfg.DEVICE)

    runs = []
    for _ in range(2):
        set_seed(42)
        m   = ResNet8().to(cfg.DEVICE)
        opt = optim.SGD(m.parameters(), lr=0.01, momentum=0.9)
        ls  = []
        for _ in range(10):
            l = F.cross_entropy(m(x), y)
            opt.zero_grad(); l.backward(); opt.step()
            ls.append(round(l.item(), 8))
        runs.append(ls)

    max_diff = max(abs(a - b) for a, b in zip(runs[0], runs[1]))
    passed   = (max_diff == 0.0)
    print(f"  max|Δloss| = {max_diff}  → {'PASS' if passed else 'FAIL'}")
    return {"passed": passed, "max_loss_diff": float(max_diff)}


def gate2_loss_at_init():
    print("\n=== GATE 2 : CE loss at init ===")
    set_seed(42)
    m   = ResNet8().to(cfg.DEVICE)
    m.eval()
    raw = torchvision.datasets.CIFAR10(cfg.DATA_DIR, train=False, download=True)
    xs  = torch.stack([NORM_T(raw[i][0]) for i in range(1000)]).to(cfg.DEVICE)
    ys  = torch.tensor([raw[i][1] for i in range(1000)]).to(cfg.DEVICE)
    with torch.no_grad():
        loss = F.cross_entropy(m(xs), ys).item()
    passed = 2.28 <= loss <= 2.32
    print(f"  initial CE = {loss:.6f}  (expect 2.28–2.32)  → {'PASS' if passed else 'FAIL'}")
    return {"passed": passed, "initial_loss": float(loss), "expected": float(np.log(10))}


def gate3_input_independent():
    print("\n=== GATE 3 : input-independent baseline ===")
    ds  = torchvision.datasets.CIFAR10(cfg.DATA_DIR, train=False, download=True)
    targets = np.array(ds.targets)
    acc = float((targets == 0).mean())
    passed = 0.098 <= acc <= 0.102
    print(f"  class-0 baseline acc = {acc:.4f}  → {'PASS' if passed else 'FAIL'}")
    return {"passed": passed, "accuracy": acc}


def gate4_overfit_batch():
    print("\n=== GATE 4 : overfit one batch ===")
    set_seed(42)
    m  = ResNet8().to(cfg.DEVICE)
    raw = RawCIFAR10(cfg.DATA_DIR)
    x  = torch.stack([NORM_T(raw[i][0]) for i in range(64)]).to(cfg.DEVICE)
    y  = torch.tensor([raw[i][1] for i in range(64)]).to(cfg.DEVICE)
    opt = optim.SGD(m.parameters(), lr=0.01, momentum=0.9, weight_decay=0)
    min_loss = 1e9
    hit_step = None
    for step in range(500):
        m.train()
        l = F.cross_entropy(m(x), y)
        opt.zero_grad(); l.backward(); opt.step()
        if l.item() < min_loss:
            min_loss = l.item()
        if l.item() < 0.01 and hit_step is None:
            hit_step = step
            break
    passed = (min_loss < 0.01)
    print(f"  min CE = {min_loss:.6f}  (hit_step={hit_step})  → {'PASS' if passed else 'FAIL'}")
    return {"passed": passed, "min_loss": float(min_loss), "hit_step": hit_step}


# ────────────────────────────────────────────────────────────────────────────
# ONLINE INVARIANCE AUDITOR (unchanged from main-01)
# ────────────────────────────────────────────────────────────────────────────

class OnlineAuditor:
    """
    Every `audit_interval` epochs:
      1. Extract features for audit pool (no aug)
      2. Identify tail = top-τ entropy samples
      3. For each active op: compute mean cosine-sim of features
         between tail baseline and tail+op views
      4. Flag ops where cos-sim < threshold; re-admit after readmit_epochs
    """

    def __init__(self, aug_pool: RandAugPool):
        self.pool             = aug_pool
        self.disabled_at      = {}
        self.consistency_log  = []   # (epoch, op_name, score)
        self.gen_gap_log      = []   # (epoch, gen_gap)
        self.rollback_log     = []   # (epoch, op_name)

    def _extract_features(self, model, imgs, device):
        model.eval()
        with torch.no_grad():
            _, feat = model(imgs.to(device), return_feat=True)
        return F.normalize(feat, dim=-1)

    def _tail_mask(self, model, imgs, device):
        model.eval()
        with torch.no_grad():
            logits = model(imgs.to(device))
            probs  = F.softmax(logits, dim=-1)
            ent    = -(probs * (probs.clamp(1e-10).log())).sum(dim=-1)
        k    = max(1, int(len(ent) * cfg.TAIL_PERCENTILE))
        _, idx = ent.topk(k)
        mask = torch.zeros(len(ent), dtype=torch.bool)
        mask[idx.cpu()] = True
        return mask

    def _apply_op_to_batch(self, imgs_norm, op_name, device):
        """Denormalise → apply PIL op → renormalise."""
        mean_t = torch.tensor(MEAN).view(3,1,1).to(imgs_norm.device)
        std_t  = torch.tensor(STD ).view(3,1,1).to(imgs_norm.device)
        raw    = (imgs_norm * std_t + mean_t).clamp(0, 1).cpu()

        out = []
        for img_t in raw:
            img_pil = TF.to_pil_image(img_t)
            aug_pil = _apply_op_pil(img_pil, op_name, cfg.AUDIT_MAGNITUDE)
            out.append(T.Normalize(MEAN, STD)(T.ToTensor()(aug_pil)))
        return torch.stack(out).to(device)

    def run(self, model, audit_ds, device, epoch, train_acc, test_acc):
        loader   = DataLoader(audit_ds, batch_size=256, shuffle=False, num_workers=0)
        all_imgs = torch.cat([b[0] for b in loader], 0).to(device)

        # re-admit ops that have been disabled long enough
        for op, dis_epoch in list(self.disabled_at.items()):
            if epoch - dis_epoch >= cfg.READMIT_EPOCHS:
                self.pool.enable(op)
                del self.disabled_at[op]
                print(f"    [auditor e{epoch}] re-admitted '{op}'")

        mask      = self._tail_mask(model, all_imgs, device)
        tail_imgs = all_imgs[mask]
        print(f"    [auditor e{epoch}] tail_size={mask.sum().item()}", end="")

        feat_base = self._extract_features(model, tail_imgs, device)

        gen_gap = float(train_acc - test_acc)
        self.gen_gap_log.append((epoch, gen_gap))

        rolls_this = 0
        for op in list(self.pool.active_names()):
            aug_imgs    = self._apply_op_to_batch(tail_imgs, op, device)
            feat_aug    = self._extract_features(model, aug_imgs, device)
            cos_sim     = (feat_base * feat_aug).sum(dim=-1).mean().item()
            self.consistency_log.append((epoch, op, float(cos_sim)))

            if cos_sim < cfg.CONSISTENCY_THRESHOLD:
                self.pool.disable(op, epoch)
                self.disabled_at[op] = epoch
                self.rollback_log.append((epoch, op))
                rolls_this += 1
                print(f"\n    [auditor e{epoch}] ROLLBACK '{op}' cos={cos_sim:.3f}", end="")

        print(f"  rolls={rolls_this}")

    def auditor_auroc(self):
        """AUROC of -mean_consistency as predictor of gen_gap > median."""
        by_epoch = {}
        for ep, _, score in self.consistency_log:
            by_epoch.setdefault(ep, []).append(score)

        common = [(ep, np.mean(by_epoch[ep]), gap)
                  for ep, gap in self.gen_gap_log if ep in by_epoch]
        if len(common) < 2:
            return None
        _, scores, gaps = zip(*common)
        median = np.median(gaps)
        labels = [1 if g > median else 0 for g in gaps]
        if len(set(labels)) < 2:
            return None
        try:
            return float(roc_auc_score(labels, [-s for s in scores]))
        except Exception:
            return None

    def auditor_precision_recall(self):
        """Gate 10: precision/recall of flagged ops vs KNOWN_HARMFUL."""
        ever_flagged = set(op for _, op in self.rollback_log)
        tp = len(ever_flagged & KNOWN_HARMFUL)
        fp = len(ever_flagged - KNOWN_HARMFUL)
        fn = len(KNOWN_HARMFUL - ever_flagged)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return float(precision), float(recall)


# ────────────────────────────────────────────────────────────────────────────
# FIXMATCH TRAINING STEP
# ────────────────────────────────────────────────────────────────────────────

def train_epoch(model, lbl_loader, unl_loader, optimizer, device):
    model.train()
    lbl_it = iter(lbl_loader)
    unl_it = iter(unl_loader)
    n_steps = max(len(lbl_loader), len(unl_loader))
    total_loss = 0.0

    for _ in range(n_steps):
        try: xl, yl = next(lbl_it)
        except StopIteration: lbl_it = iter(lbl_loader); xl, yl = next(lbl_it)
        try: xw, xs = next(unl_it)
        except StopIteration: unl_it = iter(unl_loader); xw, xs = next(unl_it)

        xl, yl = xl.to(device), yl.to(device)
        xw, xs = xw.to(device), xs.to(device)

        loss_l = F.cross_entropy(model(xl), yl)

        with torch.no_grad():
            prob_w      = F.softmax(model(xw), dim=-1)
            max_p, pl   = prob_w.max(dim=-1)
            mask        = (max_p >= cfg.FM_THRESHOLD).float()

        loss_u = (F.cross_entropy(model(xs), pl, reduction="none") * mask).mean()
        loss   = loss_l + cfg.FM_LAMBDA_U * loss_u

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / n_steps


# ────────────────────────────────────────────────────────────────────────────
# ONE FULL SEED RUN (now includes CIFAR-10-C eval)
# ────────────────────────────────────────────────────────────────────────────

def run_seed(seed, method, raw_train, labeled_idx, unlabeled_idx, audit_idx,
             test_loader, raw_test_ds, device):
    """
    method: "baseline" | "method"
    Returns dict with final_test_acc, mCE, auditor info.
    """
    set_seed(seed)
    aug_pool = RandAugPool(n=2, magnitude=9)

    lbl_ds  = LabeledSet(raw_train, labeled_idx)
    unl_ds  = UnlabeledSet(raw_train, unlabeled_idx, aug_pool)
    aud_ds  = AuditSet(raw_train, audit_idx)

    lbl_loader = DataLoader(lbl_ds, batch_size=cfg.LABELED_BATCH,
                            shuffle=True,  num_workers=2, drop_last=True,
                            pin_memory=True)
    unl_loader = DataLoader(unl_ds, batch_size=cfg.UNLABELED_BATCH,
                            shuffle=True,  num_workers=2, drop_last=True,
                            pin_memory=True)

    model     = ResNet8().to(device)
    optimizer = optim.SGD(model.parameters(), lr=cfg.LR,
                          momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY,
                          nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.NUM_EPOCHS)

    auditor = OnlineAuditor(aug_pool) if method == "method" else None

    loss_curve, acc_curve, train_acc_curve = [], [], []
    t0 = time.time()

    for ep in range(1, cfg.NUM_EPOCHS + 1):
        tr_loss = train_epoch(model, lbl_loader, unl_loader, optimizer, device)
        scheduler.step()

        te_acc = eval_acc(model, test_loader, device)
        tr_acc = eval_acc(model, lbl_loader,  device)

        loss_curve.append(float(tr_loss))
        acc_curve.append(float(te_acc))
        train_acc_curve.append(float(tr_acc))

        if ep % 5 == 0:
            elapsed = time.time() - t0
            print(f"  seed={seed} ep={ep:3d}/{cfg.NUM_EPOCHS}  "
                  f"loss={tr_loss:.4f}  tr={tr_acc:.3f}  te={te_acc:.3f}  "
                  f"({elapsed:.0f}s)")

        if auditor is not None and ep % cfg.AUDIT_INTERVAL == 0:
            auditor.run(model, aud_ds, device, ep, tr_acc, te_acc)

    # ── CIFAR-10-C evaluation ──────────────────────────────────────────────
    print(f"  seed={seed} [{method}] evaluating CIFAR-10-C mCE ...")
    t1 = time.time()
    mCE, per_cs, per_c_mean = eval_cifar10c(model, raw_test_ds, device)
    print(f"  seed={seed} [{method}] mCE={mCE:.4f}  "
          f"({time.time()-t1:.1f}s)  "
          + "  ".join(f"{c}={v:.3f}" for c, v in per_c_mean.items()))

    # Serialisable per_cs keys (tuples → strings)
    per_cs_str = {f"{c}@s{s}": v for (c, s), v in per_cs.items()}

    result = {
        "seed":            seed,
        "method":          method,
        "final_test_acc":  float(acc_curve[-1]),
        "mCE":             float(mCE),
        "per_corruption_mean_error": per_c_mean,
        "per_cs_error":    per_cs_str,
        "loss_curve":      loss_curve,
        "acc_curve":       acc_curve,
        "train_acc_curve": train_acc_curve,
    }

    if auditor is not None:
        auroc      = auditor.auditor_auroc()
        prec, rec  = auditor.auditor_precision_recall()
        result["auditor"] = {
            "auroc":        auroc,
            "precision":    prec,
            "recall":       rec,
            "rollback_log": auditor.rollback_log,
            "n_rollbacks":  len(auditor.rollback_log),
            "fired":        len(auditor.rollback_log) > 0,
        }
        print(f"  → Auditor AUROC={auroc}  prec={prec:.2f}  rec={rec:.2f}  "
              f"rollbacks={len(auditor.rollback_log)}")
    return result


# ────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    os.makedirs(cfg.WEIGHTS_DIR, exist_ok=True)

    print("=" * 70)
    print("Experiment : Online Mid-Training Invariance Auditor  [round refine-02]")
    print(f"Device     : {cfg.DEVICE}")
    print(f"Epochs     : {cfg.NUM_EPOCHS}  | Seeds: {cfg.SEEDS}")
    print(f"CIFAR-10-C : {len(cfg.CIFAR10C_TYPES)} corruption types × "
          f"{len(cfg.CIFAR10C_SEVERITIES)} severities = "
          f"{len(cfg.CIFAR10C_TYPES)*len(cfg.CIFAR10C_SEVERITIES)} pairs")
    print("=" * 70)

    # ── Sanity gates ─────────────────────────────────────────────────────
    g1 = gate1_reproducibility()
    g2 = gate2_loss_at_init()
    g3 = gate3_input_independent()
    g4 = gate4_overfit_batch()

    gates = {
        "gate_1_reproducibility":   g1,
        "gate_2_loss_at_init":      g2,
        "gate_3_input_independent": g3,
        "gate_4_overfit_batch":     g4,
        "all_passed": all(g["passed"] for g in [g1, g2, g3, g4]),
    }
    print(f"\nSanity gates all_passed={gates['all_passed']}\n")
    if not gates["all_passed"]:
        print("WARNING: some sanity gates FAILED — continuing but results may be unreliable")

    # ── Data ─────────────────────────────────────────────────────────────
    raw_train, labeled_idx, unlabeled_idx, audit_idx = make_ssl_split(
        cfg.DATA_DIR, cfg.LABELED_PER_CLASS,
        cfg.UNLABELED_SIZE, cfg.AUDIT_POOL_SIZE, seed=42,
    )
    print(f"Split: labeled={len(labeled_idx)}  "
          f"unlabeled={len(unlabeled_idx)}  audit={len(audit_idx)}")

    test_ds_norm = torchvision.datasets.CIFAR10(
        cfg.DATA_DIR, train=False, download=True,
        transform=T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)]),
    )
    test_loader = DataLoader(test_ds_norm, batch_size=512, shuffle=False,
                             num_workers=2, pin_memory=True)

    # raw test dataset for CIFAR-10-C (no transform; PIL output)
    raw_test_ds = torchvision.datasets.CIFAR10(
        cfg.DATA_DIR, train=False, download=True,
        # no transform — corruption functions work on PIL images
    )

    # ── Experiments ───────────────────────────────────────────────────────
    baseline_runs, method_runs = [], []

    for seed in cfg.SEEDS:
        print(f"\n{'─'*60}")
        print(f"BASELINE  seed={seed}")
        print(f"{'─'*60}")
        r = run_seed(seed, "baseline", raw_train,
                     labeled_idx, unlabeled_idx, audit_idx,
                     test_loader, raw_test_ds, cfg.DEVICE)
        baseline_runs.append(r)

        print(f"\n{'─'*60}")
        print(f"METHOD    seed={seed}")
        print(f"{'─'*60}")
        r = run_seed(seed, "method", raw_train,
                     labeled_idx, unlabeled_idx, audit_idx,
                     test_loader, raw_test_ds, cfg.DEVICE)
        method_runs.append(r)

    # ── Summary statistics ────────────────────────────────────────────────
    b_accs = [r["final_test_acc"] for r in baseline_runs]
    m_accs = [r["final_test_acc"] for r in method_runs]
    b_mCEs = [r["mCE"]           for r in baseline_runs]
    m_mCEs = [r["mCE"]           for r in method_runs]

    b_acc_mean, b_acc_std = float(np.mean(b_accs)), float(np.std(b_accs))
    m_acc_mean, m_acc_std = float(np.mean(m_accs)), float(np.std(m_accs))
    b_mCE_mean, b_mCE_std = float(np.mean(b_mCEs)), float(np.std(b_mCEs))
    m_mCE_mean, m_mCE_std = float(np.mean(m_mCEs)), float(np.std(m_mCEs))

    delta_acc = m_acc_mean - b_acc_mean
    delta_mCE = m_mCE_mean - b_mCE_mean  # negative = method is better (lower error)

    # Auditor stats
    aurocs = [r["auditor"]["auroc"]     for r in method_runs
              if r.get("auditor", {}).get("auroc") is not None]
    precs  = [r["auditor"]["precision"] for r in method_runs if "auditor" in r]
    recs   = [r["auditor"]["recall"]    for r in method_runs if "auditor" in r]
    fired  = [r["auditor"]["fired"]     for r in method_runs if "auditor" in r]
    n_rolls= [r["auditor"]["n_rollbacks"] for r in method_runs if "auditor" in r]

    mean_auroc = float(np.mean(aurocs)) if aurocs else None
    mean_prec  = float(np.mean(precs))  if precs  else None
    mean_rec   = float(np.mean(recs))   if recs   else None

    # Per-corruption mCE comparison
    per_c_baseline = {}
    per_c_method   = {}
    for c in cfg.CIFAR10C_TYPES:
        b_vals = [r["per_corruption_mean_error"][c] for r in baseline_runs]
        m_vals = [r["per_corruption_mean_error"][c] for r in method_runs]
        per_c_baseline[c] = float(np.mean(b_vals))
        per_c_method[c]   = float(np.mean(m_vals))

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Baseline (FixMatch+RandAugment)    clean acc: {b_accs}  mean={b_acc_mean:.4f}±{b_acc_std:.4f}")
    print(f"Method   (FixMatch+OnlineAuditor)  clean acc: {m_accs}  mean={m_acc_mean:.4f}±{m_acc_std:.4f}")
    print(f"  Δ clean acc: {delta_acc:+.4f}")
    print()
    print(f"Baseline  mCE: {[f'{v:.4f}' for v in b_mCEs]}  mean={b_mCE_mean:.4f}±{b_mCE_std:.4f}")
    print(f"Method    mCE: {[f'{v:.4f}' for v in m_mCEs]}  mean={m_mCE_mean:.4f}±{m_mCE_std:.4f}")
    print(f"  Δ mCE: {delta_mCE:+.4f} (negative = method is MORE robust)")
    print()
    print("Per-corruption mean error (baseline → method → delta):")
    for c in cfg.CIFAR10C_TYPES:
        d = per_c_method[c] - per_c_baseline[c]
        print(f"  {c:20s}: {per_c_baseline[c]:.4f} → {per_c_method[c]:.4f}  ({d:+.4f})")
    print()
    if mean_auroc is not None:
        print(f"Auditor AUROC (per-seed): {aurocs}  mean={mean_auroc:.4f}")
    print(f"Auditor prec/rec: {mean_prec:.3f} / {mean_rec:.3f}")
    print(f"Auditor fired:  {fired}  total rollbacks: {n_rolls}")

    # ── RESULTS.json ──────────────────────────────────────────────────────
    results = {
        "status": "SUCCESS" if gates["all_passed"] else "PARTIAL",
        "scale":  "probe",
        "metrics": {
            "sanity_gates": gates,
            "baseline_fixmatch_randaug": {
                "per_seed_test_acc": b_accs,
                "mean_test_acc":     b_acc_mean,
                "std_test_acc":      b_acc_std,
                "per_seed_mCE":      b_mCEs,
                "mean_mCE":          b_mCE_mean,
                "std_mCE":           b_mCE_std,
                "per_corruption_mean_error": per_c_baseline,
            },
            "method_fixmatch_auditor": {
                "per_seed_test_acc":  m_accs,
                "mean_test_acc":      m_acc_mean,
                "std_test_acc":       m_acc_std,
                "per_seed_mCE":       m_mCEs,
                "mean_mCE":           m_mCE_mean,
                "std_mCE":            m_mCE_std,
                "per_corruption_mean_error": per_c_method,
                "auditor_auroc_per_seed": aurocs,
                "mean_auditor_auroc":     mean_auroc,
                "mean_auditor_precision": mean_prec,
                "mean_auditor_recall":    mean_rec,
                "auditor_fired_per_seed": fired,
                "rollbacks_per_seed":     n_rolls,
            },
            "delta_mean_test_acc": float(delta_acc),
            "delta_mean_mCE":      float(delta_mCE),
            "per_corruption_delta_mCE": {
                c: float(per_c_method[c] - per_c_baseline[c])
                for c in cfg.CIFAR10C_TYPES
            },
            "n_seeds":       len(cfg.SEEDS),
            "epochs_per_run": cfg.NUM_EPOCHS,
            "n_corruptions": len(cfg.CIFAR10C_TYPES),
            "n_severities":  len(cfg.CIFAR10C_SEVERITIES),
        },
        "subject_executed": (
            f"FixMatch+RandAugment (baseline) vs FixMatch+OnlineAuditor (method); "
            f"ResNet-8 on CIFAR-10 semi-supervised "
            f"({cfg.LABELED_PER_CLASS} labels/class, {cfg.UNLABELED_SIZE} unlabeled); "
            f"{cfg.NUM_EPOCHS} epochs × {len(cfg.SEEDS)} seeds; "
            f"CIFAR-10-C eval: {len(cfg.CIFAR10C_TYPES)} corruption types × "
            f"{len(cfg.CIFAR10C_SEVERITIES)} severities"
        ),
        "notes": (
            f"Probe: {cfg.NUM_EPOCHS} epochs (spec=500). "
            f"CRITICAL FIX from main-01: added CIFAR-10-C mCE evaluation. "
            f"Corruptions implemented from scratch (no external package). "
            f"mCE = mean error rate over {len(cfg.CIFAR10C_TYPES)} types × "
            f"{len(cfg.CIFAR10C_SEVERITIES)} severities (raw, not AlexNet-normalized). "
            f"Auditor runs every {cfg.AUDIT_INTERVAL} epochs, "
            f"threshold τ={cfg.CONSISTENCY_THRESHOLD}. "
            f"delta_mCE={delta_mCE:+.4f} "
            f"({'method more robust' if delta_mCE < 0 else 'method less robust or same'})."
        ),
    }

    out_path = os.path.join(cfg.RESULTS_DIR, "RESULTS.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")

    # Detailed per-seed
    detailed = {
        "gates": gates,
        "baseline": [
            {k: v for k, v in r.items()
             if k not in ("loss_curve", "acc_curve", "train_acc_curve", "per_cs_error")}
            for r in baseline_runs
        ],
        "method": [
            {k: v for k, v in r.items()
             if k not in ("loss_curve", "acc_curve", "train_acc_curve", "per_cs_error")}
            for r in method_runs
        ],
    }
    with open(os.path.join(cfg.RESULTS_DIR, "detailed_results.json"), "w") as f:
        json.dump(detailed, f, indent=2, default=str)

    return results


if __name__ == "__main__":
    results = main()
    print("\n" + "=" * 70)
    print("RESULTS.json")
    print("=" * 70)
    print(json.dumps(results, indent=2))

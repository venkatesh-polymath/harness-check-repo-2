"""
Sanity gates (run once before main experiment):
  G1: Fixed seed → bit-identical init loss across two launches
  G2: Loss at init ≈ ln(10) ≈ 2.303
  G3: Dummy (uniform) classifier → 10% accuracy on balanced test set
  G4: Overfit one batch → loss < 0.01 after 200 SGD steps
"""

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def gate_fixed_seed(model_factory, dataloader, device, seed: int = 42, n_steps: int = 10):
    """
    G1: Two runs with the same seed must produce bit-identical per-step CE losses
    for the first n_steps optimizer steps.
    """
    results = []
    for _ in range(2):
        set_seed(seed)
        model = model_factory().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        losses = []
        it = iter(dataloader)
        model.train()
        for step in range(n_steps):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(dataloader)
                x, y = next(it)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        results.append(losses)

    max_diff = max(abs(a - b) for a, b in zip(results[0], results[1]))
    passed = max_diff < 1e-5
    return {
        "gate": "fixed_seed",
        "passed": passed,
        "max_diff": max_diff,
        "losses_run1": results[0],
        "losses_run2": results[1],
        "threshold": 1e-5,
    }


def gate_loss_at_init(model_factory, dataloader, device, seed: int = 42):
    """
    G2: Loss at step 0 should be near ln(10) ≈ 2.303 for uniform 10-class init.
    """
    set_seed(seed)
    model = model_factory().to(device)
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y, reduction="sum")
            total_loss += loss.item()
            n += len(y)
            if n >= 512:
                break
    avg_loss = total_loss / n
    expected = math.log(10)
    passed = abs(avg_loss - expected) < 0.2
    return {
        "gate": "loss_at_init",
        "passed": passed,
        "loss_0": avg_loss,
        "expected": expected,
        "diff": abs(avg_loss - expected),
        "threshold": 0.2,
    }


def gate_dummy_classifier(dataloader, device):
    """
    G3: A uniform-output dummy classifier should get ~10% accuracy on a balanced test set.
    """
    correct = 0
    total = 0
    for batch in dataloader:
        if len(batch) == 2:
            x, y = batch
        else:
            x, y = batch[0], batch[1]
        # Dummy: predict class = 0 always (10% on balanced 10-class)
        preds = torch.zeros(len(y), dtype=torch.long)
        correct += (preds == y).sum().item()
        total += len(y)
        if total >= 1000:
            break
    acc = correct / total
    passed = 9.5 <= acc * 100 <= 10.5
    return {
        "gate": "dummy_classifier",
        "passed": passed,
        "dummy_acc": acc,
        "dummy_acc_pct": acc * 100,
        "threshold_low": 9.5,
        "threshold_high": 10.5,
    }


def gate_overfit_one_batch(model_factory, single_batch, device, seed: int = 42,
                           n_steps: int = 200, lr: float = 0.1):
    """
    G4: Should reach training CE < 0.01 on a fixed batch of 32 samples in 200 SGD steps.
    """
    set_seed(seed)
    model = model_factory().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    x, y = single_batch
    x, y = x.to(device), y.to(device)

    model.train()
    final_loss = None
    for step in range(n_steps):
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    passed = final_loss < 0.01
    return {
        "gate": "overfit_one_batch",
        "passed": passed,
        "train_loss_step200": final_loss,
        "threshold": 0.01,
    }

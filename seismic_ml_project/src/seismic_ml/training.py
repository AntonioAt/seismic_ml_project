"""
training.py
===========
Training loop with Dice + CrossEntropy loss, Adam optimizer, CosineAnnealing scheduler.
"""
from __future__ import annotations
import os
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.shape[1]
        probs     = torch.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets, n_classes).permute(0, 4, 1, 2, 3).float()
        dims  = (0, 2, 3, 4)
        inter = (probs * targets_oh).sum(dim=dims)
        union = probs.sum(dim=dims) + targets_oh.sum(dim=dims)
        dice  = (2.0 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.dice = DiceLoss()
        self.ce   = nn.CrossEntropyLoss()
        self.w    = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.w * self.dice(logits, targets) + (1 - self.w) * self.ce(logits, targets)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    model.train()
    total = 0.0
    for patches, labels in loader:
        patches = patches.to(device, non_blocking=True)
        labels  = labels.to(device,  non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(patches), labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)


def _validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for patches, labels in loader:
            patches = patches.to(device, non_blocking=True)
            labels  = labels.to(device,  non_blocking=True)
            total  += criterion(model(patches), labels).item()
    return total / max(len(loader), 1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    n_epochs: int = 50,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    checkpoint_dir: Optional[str] = None,
    grad_clip: float = 1.0,
) -> Dict[str, List[float]]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model     = model.to(device)
    criterion = CombinedLoss(dice_weight=0.5)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)
    history   = {"train_loss": [], "val_loss": []}
    best_val  = float("inf")

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, grad_clip)
        scheduler.step()
        history["train_loss"].append(train_loss)

        val_str = ""
        if val_loader:
            val_loss = _validate(model, val_loader, criterion, device)
            history["val_loss"].append(val_loss)
            val_str = f"  val={val_loss:.4f}"
            if checkpoint_dir and val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(),
                           os.path.join(checkpoint_dir, "best_model.pt"))

        print(f"Epoch {epoch:03d}/{n_epochs}  train={train_loss:.4f}{val_str}")

    return history

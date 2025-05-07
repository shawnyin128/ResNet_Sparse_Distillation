import torch
import torch.nn as nn

from torch.utils.data import DataLoader


def validate(model: nn.Module,
             val_loader: DataLoader,
             device: str="cuda") -> float:
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

    return 100. * correct / total
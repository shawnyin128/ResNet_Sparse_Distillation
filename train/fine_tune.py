import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from evaluation.validate import validate
from evaluation.sparsity import calculate_sparsity
from train.distillation import get_weights_norm


def fine_tune_loss(model: nn.Module,
                   alpha: float=1.0,
                   beta: float=1e-4,
                   norm_type: str='l2') -> torch.Tensor:
    layer_norm_dict = get_weights_norm(model, norm_type)
    layer_names = list(layer_norm_dict.keys())

    norm_inter = 0.0
    for layer in layer_names:
        norm_inter += layer_norm_dict[layer]

    norm = beta * norm_inter

    total_loss = alpha * norm
    return total_loss


def fine_tune(model: nn.Module,
              train_loader: DataLoader,
              val_loader: DataLoader,
              criterion: nn.Module,
              optimizer: optim.Optimizer,
              scheduler: optim.lr_scheduler,
              epoch: int,
              norm_type: str='l2',
              theta: float=0.1,
              alpha: float=1.0,
              beta: float=1e-5,
              device: str='cuda') -> None:
    model.to(device)

    out_bar = tqdm(range(epoch), desc="Epoch")
    for epoch in out_bar:
        model.train()
        pbar = tqdm(train_loader, desc=f"Train E{epoch}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            output = model(x)

            loss_cls = criterion(output, y)

            loss_norm = fine_tune_loss(model, alpha, beta, norm_type)

            loss = theta * loss_cls + (1-theta) * loss_norm

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            pbar.set_postfix({"loss": loss_cls.item(), "norm": loss_norm.item(), "sparsity": calculate_sparsity(model)})

        out_bar.set_postfix({"accuracy": validate(model, val_loader, device)})

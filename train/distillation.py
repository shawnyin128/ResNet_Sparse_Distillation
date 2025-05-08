import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from evaluation.validate import validate
from evaluation.sparsity import calculate_sparsity


# l1 norm
def get_weights_norm(model: nn.Module) -> dict:
    conv_weight_stats = {}

    for name, module in model.named_children():
        if isinstance(module, nn.Sequential) and "downsample" not in name:
            l1_total = 0.0
            for submodule in module.modules():
                if isinstance(submodule, nn.Conv2d):
                    weight = submodule.weight
                    l1_total += weight.abs().sum()
            conv_weight_stats[name] = l1_total

    return conv_weight_stats


# soft KL divergence
def solve_sigma(threshold: float,
                temperature: int,
                penalty_bar: float) -> float:
    return penalty_bar + (np.log((penalty_bar/threshold) - 1)) / temperature


def soft_sigmoid(x: torch.Tensor,
                 T: int,
                 offset: float,
                 T2: int,
                 offset2: float) -> torch.Tensor:
    return 0.5 * ((1 / (1 + torch.exp(-T*(x - offset)))) + (1 / (1 + torch.exp(-T2*(x - offset2)))))


def soft_kl(student_logits: torch.Tensor,
            teacher_logits: torch.Tensor,
            T: float,
            sigmoid_T1: int,
            sigmoid_T2: int,
            offset1: float,
            offset2: float,
            use_soft: bool) -> torch.Tensor:
    if use_soft:
        soft_teacher_logits = teacher_logits.detach() * soft_sigmoid(teacher_logits.detach(), sigmoid_T1, offset1, sigmoid_T2, offset2)
    else:
        soft_teacher_logits = teacher_logits.detach()

    soft_kl_score = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(soft_teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T ** 2)

    return soft_kl_score


def distill_loss(student_logits: torch.Tensor,
                 teacher_logits: torch.Tensor,
                 student_features: list,
                 teacher_features: list,
                 student_model: nn.Module,
                 is_l1: bool=True,
                 is_soft_kl: bool=True,
                 alpha: float=1.0,
                 beta: float=1e-4,
                 T: float=4.0,
                 sigmoid_T1: int=10000,
                 offset1: float=0.001,
                 sigmoid_T2: int=10000,
                 offset2: float=0.001,
                 use_soft: bool=True) -> torch.Tensor:
    kl_final = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T ** 2)

    layer_norm_dict = get_weights_norm(student_model)
    layer_names = list(layer_norm_dict.keys())

    kl_inter = 0.0
    norm_inter = 0.0
    for i, layer in enumerate(layer_names):
        if is_soft_kl:
            s_feat = student_features[i]
            t_feat = teacher_features[i]
            kl = soft_kl(s_feat.view(s_feat.size(0), -1),
                         t_feat.view(t_feat.size(0), -1),
                         T, sigmoid_T1, sigmoid_T2, offset1, offset2, use_soft)
            kl_inter += kl
        if is_l1:
            norm_inter += layer_norm_dict[layer]

    kl = kl_inter + beta * norm_inter

    total_loss = kl_final + alpha * kl
    return total_loss


# forward logic
def forward_with_intermediate(model: nn.Module,
                              x: torch.Tensor,
                              return_layers: tuple=("layer1", "layer2", "layer3", "layer4")) -> tuple:
    features = []

    x = model.conv1(x)
    x = model.bn1(x)
    x = F.relu(x)

    for name in ["layer1", "layer2", "layer3", "layer4"]:
        x = getattr(model, name)(x)
        if name in return_layers:
            features.append(x)

    x = F.avg_pool2d(x, 4)
    x = x.view(x.size(0), -1)
    x = model.linear(x)

    return x, features


# distillation function
def distillation(teacher: nn.Module,
                 student: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 criterion: nn.Module,
                 optimizer: optim.Optimizer,
                 scheduler: optim.lr_scheduler,
                 epoch: int,
                 theta: float=0.1,
                 alpha: float=1.0,
                 beta: float=1e-5,
                 is_l1: bool=True,
                 is_soft_kl: bool=True,
                 T: float=4.0,
                 sigmoid_T1: int=10000,
                 sigmoid_T2: int=10000,
                 penalty_output: float=0.0015,
                 use_soft: bool=True,
                 device: str='cuda') -> None:
    teacher.to(device)
    student.to(device)

    offset = solve_sigma(0.001, sigmoid_T1, penalty_output)

    out_bar = tqdm(range(epoch), desc="Epoch")
    for epoch in out_bar:
        teacher.eval()
        student.train()

        pbar = tqdm(train_loader, desc=f"Train E{epoch}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                t_logits, t_feats = forward_with_intermediate(teacher, x)
            s_logits, s_feats = forward_with_intermediate(student, x)

            loss_cls = criterion(s_logits, y)

            loss_kd = distill_loss(s_logits, t_logits, s_feats, t_feats, student,
                                   is_l1, is_soft_kl, alpha, beta, T, sigmoid_T1, offset, sigmoid_T2, offset, use_soft)

            loss = theta * loss_cls + (1-theta) * loss_kd

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            pbar.set_postfix({"loss": loss_cls.item(), "kl": loss_kd.item(), "sparsity": calculate_sparsity(student)})

        out_bar.set_postfix({"accuracy": validate(student, val_loader, device)})
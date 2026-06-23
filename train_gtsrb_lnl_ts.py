import argparse
import copy
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader

from LNL_TS import LNL_TS_Ti


def parse_args():
    parser = argparse.ArgumentParser(description='Train LNL-TS on GTSRB.')
    parser.add_argument('--train-dir', default='./data/GTSRB/Final_Training/Images')
    parser.add_argument('--test-dir', default='./data/GTSRB/test')
    parser.add_argument('--output-dir', default='./checkpoints')
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=3e-2)
    parser.add_argument('--warmup-epochs', type=int, default=10)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--label-smoothing', type=float, default=0.05)
    parser.add_argument('--moex-prob', type=float, default=0.25)
    parser.add_argument('--moex-lam', type=float, default=0.85)
    parser.add_argument('--mixup-alpha', type=float, default=0.05)
    parser.add_argument('--cutmix-alpha', type=float, default=0.0)
    parser.add_argument('--ema-decay', type=float, default=0.999)
    parser.add_argument('--drop-rate', type=float, default=0.03)
    parser.add_argument('--drop-path-rate', type=float, default=0.05)
    parser.add_argument('--topk-average', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            224,
            scale=(0.82, 1.0),
            ratio=(0.90, 1.10),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=10,
                translate=(0.04, 0.04),
                scale=(0.92, 1.08),
                shear=5,
                interpolation=InterpolationMode.BILINEAR,
            )
        ], p=0.6),
        transforms.RandomPerspective(distortion_scale=0.12, p=0.20),
        transforms.RandAugment(num_ops=2, magnitude=7),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.18, hue=0.02),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))], p=0.08),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.10, scale=(0.005, 0.035), ratio=(0.3, 3.3), value='random'),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])
    return train_transform, eval_transform


def cosine_with_warmup(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class ModelEma:
    def __init__(self, model, decay):
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        for param in self.module.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        ema_state = self.module.state_dict()
        model_state = model.state_dict()
        for key, value in ema_state.items():
            model_value = model_state[key].detach()
            if value.dtype.is_floating_point:
                value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)
            else:
                value.copy_(model_value)


def accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total


def smooth_one_hot(labels, num_classes, smoothing):
    confidence = 1.0 - smoothing
    off_value = smoothing / num_classes
    targets = torch.full((labels.size(0), num_classes), off_value, device=labels.device)
    targets.scatter_(1, labels.unsqueeze(1), confidence + off_value)
    return targets


def soft_cross_entropy(outputs, targets):
    return torch.sum(-targets * torch.log_softmax(outputs, dim=1), dim=1).mean()


def rand_bbox(size, lam):
    width = size[-1]
    height = size[-2]
    cut_ratio = math.sqrt(1.0 - lam)
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)

    cx = np.random.randint(width)
    cy = np.random.randint(height)
    x1 = np.clip(cx - cut_w // 2, 0, width)
    y1 = np.clip(cy - cut_h // 2, 0, height)
    x2 = np.clip(cx + cut_w // 2, 0, width)
    y2 = np.clip(cy + cut_h // 2, 0, height)
    return x1, y1, x2, y2


def apply_mixup_cutmix(images, labels, num_classes, args):
    targets = smooth_one_hot(labels, num_classes, args.label_smoothing)
    use_mixup = args.mixup_alpha > 0
    use_cutmix = args.cutmix_alpha > 0
    if not use_mixup and not use_cutmix:
        return images, targets

    do_cutmix = use_cutmix and (not use_mixup or torch.rand(1).item() < 0.5)
    alpha = args.cutmix_alpha if do_cutmix else args.mixup_alpha
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(images.size(0), device=images.device)
    mixed_targets = targets * lam + targets[index] * (1.0 - lam)

    if do_cutmix:
        x1, y1, x2, y2 = rand_bbox(images.size(), lam)
        images = images.clone()
        images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (images.size(-1) * images.size(-2)))
        mixed_targets = targets * lam + targets[index] * (1.0 - lam)
        return images, mixed_targets

    return images * lam + images[index] * (1.0 - lam), mixed_targets


def train_one_epoch(model, loader, criterion, optimizer, scaler, ema, device, args):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        use_moex = args.moex_prob > 0 and torch.rand(1).item() < args.moex_prob
        with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
            if use_moex:
                swap_index = torch.randperm(images.size(0), device=device)
                target_a = labels
                target_b = labels[swap_index]
                outputs = model(
                    images,
                    swap_index=swap_index,
                    moex_norm='pono',
                    moex_epsilon=1e-5,
                    moex_layer='stem',
                    moex_positive_only=False,
                )
                loss = criterion(outputs, target_a) * args.moex_lam
                loss = loss + criterion(outputs, target_b) * (1.0 - args.moex_lam)
            else:
                images, targets = apply_mixup_cutmix(images, labels, model.num_classes, args)
                outputs = model(images)
                loss = soft_cross_entropy(outputs, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


def save_state_dict(path, model):
    torch.save(model.state_dict(), path)


def average_state_dicts(state_dicts):
    avg = {}
    for key in state_dicts[0]:
        values = [state_dict[key] for state_dict in state_dicts]
        if values[0].dtype.is_floating_point:
            avg[key] = torch.stack([value.float() for value in values], dim=0).mean(dim=0).to(values[0].dtype)
        else:
            avg[key] = values[0]
    return avg


def main():
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_transform, eval_transform = build_transforms()
    trainset = torchvision.datasets.ImageFolder(args.train_dir, transform=train_transform)
    testset = torchvision.datasets.ImageFolder(args.test_dir, transform=eval_transform)

    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        testset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = LNL_TS_Ti(
        pretrained=False,
        num_classes=43,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path_rate,
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = cosine_with_warmup(optimizer, args.warmup_epochs, args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
    ema = ModelEma(model, args.ema_decay)

    best_acc = 0.0
    top_states = []
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, ema, device, args)
        scheduler.step()

        raw_acc = accuracy(model, test_loader, device)
        ema_acc = accuracy(ema.module, test_loader, device)
        current_acc = max(raw_acc, ema_acc)
        print(
            f'Epoch [{epoch + 1}/{args.epochs}] '
            f'loss={train_loss:.4f} raw_acc={raw_acc:.2f} ema_acc={ema_acc:.2f}'
        )

        if current_acc > best_acc:
            best_acc = current_acc
            best_model = ema.module if ema_acc >= raw_acc else model
            save_state_dict(os.path.join(args.output_dir, 'lnl_ts_ti_gtsrb_best.pth'), best_model)
            torch.save(
                {
                    'model_state': best_model.state_dict(),
                    'epoch': epoch + 1,
                    'best_acc': best_acc,
                    'args': vars(args),
                },
                os.path.join(args.output_dir, 'lnl_ts_ti_gtsrb_best_checkpoint.pth'),
            )
            print(f'Saved new best checkpoint: {best_acc:.2f}%')

        if args.topk_average > 0:
            top_states.append((current_acc, copy.deepcopy((ema.module if ema_acc >= raw_acc else model).state_dict())))
            top_states = sorted(top_states, key=lambda item: item[0], reverse=True)[:args.topk_average]
            if len(top_states) >= 2:
                averaged_state = average_state_dicts([state for _, state in top_states])
                torch.save(averaged_state, os.path.join(args.output_dir, 'lnl_ts_ti_gtsrb_topk_avg.pth'))

    print(f'Best Top-1 accuracy: {best_acc:.2f}%')


if __name__ == '__main__':
    main()

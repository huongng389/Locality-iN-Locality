import argparse
import copy
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from GTSRB_StrongModel import LNL_Ti


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_loaders(data_root, batch_size, workers):
    train_tf = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.72, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomApply([transforms.RandomRotation(15)], p=0.45),
            transforms.RandomPerspective(distortion_scale=0.18, p=0.25),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.04),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
        ]
    )
    test_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    trainset = datasets.ImageFolder(
        root=os.path.join(data_root, "GTSRB/Final_Training/Images"),
        transform=train_tf,
    )
    testset = datasets.ImageFolder(
        root=os.path.join(data_root, "GTSRB/test"),
        transform=test_tf,
    )

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        testset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    return trainset, testset, train_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        pred = model(images).argmax(dim=1)
        correct += pred.eq(labels).sum().item()
        total += labels.numel()
    return 100.0 * correct / total


def train(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, train_loader, test_loader = build_loaders(args.data_root, args.batch_size, args.workers)

    model = LNL_Ti(pretrained=args.imagenet_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")

    best_acc = 0.0
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=args.amp and device.type == "cuda"):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        scheduler.step()
        acc = evaluate(model, test_loader, device)
        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {"state_dict": best_state, "acc": best_acc, "epoch": epoch + 1, "args": vars(args)},
                args.output,
            )
        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} "
            f"loss={running_loss / len(train_loader):.4f} "
            f"acc={acc:.3f}% best={best_acc:.3f}%"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Best Top-1: {best_acc:.3f}%")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--output", default="pretrained_gtsrb_strong.pth")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--imagenet-pretrained", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

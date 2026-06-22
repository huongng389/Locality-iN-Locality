import argparse

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from LNL_TS import LNL_TS_Ti, load_lnl_ts_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate LNL-TS on GTSRB.')
    parser.add_argument('--test-dir', default='./data/GTSRB/test')
    parser.add_argument('--weights', default='./checkpoints/lnl_ts_ti_gtsrb_best.pth')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--tta', action='store_true', help='Use 5-crop test-time augmentation.')
    return parser.parse_args()


def build_transform(use_tta):
    if not use_tta:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    return transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.FiveCrop(224),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    ])


@torch.no_grad()
def evaluate(model, loader, device, use_tta):
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        labels = labels.to(device, non_blocking=True)
        if use_tta:
            batch_size, crops, channels, height, width = images.shape
            images = images.view(batch_size * crops, channels, height, width)
            images = images.to(device, non_blocking=True)
            outputs = model(images).view(batch_size, crops, -1).mean(dim=1)
        else:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
        predicted = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = torchvision.datasets.ImageFolder(args.test_dir, transform=build_transform(args.tta))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = LNL_TS_Ti(pretrained=False, num_classes=43)
    model.load_state_dict(load_lnl_ts_checkpoint(args.weights, map_location='cpu'))
    model.to(device)

    acc = evaluate(model, loader, device, args.tta)
    print(f'Top-1 accuracy: {acc:.2f}%')


if __name__ == '__main__':
    main()

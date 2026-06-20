"""
Plug-and-play GTSRB model for the Locality-iN-Locality notebook.

Usage in Instructions.ipynb:
    from GTSRB_StrongModel import LNL_Ti as small
    model = small(pretrained=False)
    model.head = torch.nn.Linear(in_features=192, out_features=43, bias=True)
"""
import torch
import torch.nn as nn
import timm


class _InputNormalize(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class GTSRBConvNeXtTiny(nn.Module):
    """ConvNeXt-Tiny backbone with a 192-dim head adapter.

    The adapter keeps compatibility with the original notebook line:
        model.head = torch.nn.Linear(in_features=192, out_features=43)
    """

    def __init__(self, num_classes=43, pretrained=False, drop_rate=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.normalize = _InputNormalize()
        self.backbone = timm.create_model(
            "convnext_tiny.fb_in22k_ft_in1k",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            drop_path_rate=0.1,
        )
        in_features = self.backbone.num_features
        self.neck = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(drop_rate),
            nn.Linear(in_features, 192),
            nn.GELU(),
            nn.Dropout(drop_rate),
        )
        self.head = nn.Linear(192, num_classes)

    def reset_classifier(self, num_classes, global_pool="avg"):
        self.num_classes = num_classes
        self.head = nn.Linear(192, num_classes)

    def get_classifier(self):
        return self.head

    def forward_features(self, x):
        x = self.normalize(x)
        x = self.backbone(x)
        return self.neck(x)

    def forward(self, x):
        return self.head(self.forward_features(x))


def LNL_Ti(pretrained=False, **kwargs):
    return GTSRBConvNeXtTiny(pretrained=pretrained, **kwargs)


def GTSRB_Strong(pretrained=False, **kwargs):
    return GTSRBConvNeXtTiny(pretrained=pretrained, **kwargs)

"""
Traffic-sign tuned LNL model for GTSRB.

This module keeps the original LNL/MoEx forward API so it can replace
LNL_MoEx.py in the provided Colab notebook with only the import changed.
"""
import torch
import torch.nn as nn
from timm.models.helpers import load_pretrained
from timm.models.registry import register_model

from LNL_MoEx import LocalViT_TNT, default_cfgs


class TrafficSignHead(nn.Module):
    def __init__(self, in_features=192, num_classes=43, drop_rate=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.drop = nn.Dropout(drop_rate)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(self.drop(self.norm(x)))


class TrafficSignLNL(LocalViT_TNT):
    """LNL-TNT with a regularized traffic-sign classification head."""

    def __init__(self, num_classes=43, head_drop_rate=0.1, **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)
        self.head = TrafficSignHead(self.embed_dim, num_classes, head_drop_rate)

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = TrafficSignHead(self.embed_dim, num_classes)


@register_model
def LNL_TS_Ti(pretrained=False, num_classes=43, **kwargs):
    model = TrafficSignLNL(
        patch_size=16,
        embed_dim=192,
        in_dim=12,
        depth=12,
        num_heads=3,
        in_num_head=3,
        qkv_bias=False,
        num_classes=num_classes,
        **kwargs,
    )
    model.default_cfg = default_cfgs['tnt_t_conv_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


def load_lnl_ts_checkpoint(path, map_location='cpu'):
    """Load either a pure state_dict or a training checkpoint dict."""
    checkpoint = torch.load(path, map_location=map_location)
    if isinstance(checkpoint, dict):
        for key in ('model', 'model_state', 'state_dict'):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint

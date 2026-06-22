"""Traffic-sign tuned LNL model for GTSRB.

This module keeps the original LNL/MoEx forward API and classifier shape, so
the provided Colab notebook can use it by changing only the import line.
"""
import os
import torch
import torch.nn as nn
from timm.models.registry import register_model

from LNL_MoEx import LocalViT_TNT, default_cfgs


DEFAULT_PRETRAINED_PATHS = (
    'pretrained/lnl_ts_ti_gtsrb.pth',
    'lnl_ts_ti_gtsrb.pth',
    'checkpoints/lnl_ts_ti_gtsrb_best.pth',
)


class TrafficSignStem(nn.Sequential):
    """A stronger local stem with the same output shape as the original 7x7 conv."""

    def __init__(self, in_chans=3, out_chans=12):
        super().__init__()
        self.add_module('conv1', nn.Conv2d(in_chans, out_chans, 3, stride=2, padding=1, bias=False))
        self.add_module('bn1', nn.BatchNorm2d(out_chans))
        self.add_module('act1', nn.GELU())
        self.add_module(
            'dwconv',
            nn.Conv2d(out_chans, out_chans, 3, stride=2, padding=1, groups=out_chans, bias=False),
        )
        self.add_module('bn2', nn.BatchNorm2d(out_chans))
        self.add_module('act2', nn.GELU())
        self.add_module('pwconv', nn.Conv2d(out_chans, out_chans, 1, bias=False))
        self.add_module('bn3', nn.BatchNorm2d(out_chans))


class TrafficSignLNL(LocalViT_TNT):
    """LNL-TNT with a traffic-sign oriented convolutional stem."""

    def __init__(self, in_chans=3, num_classes=43, **kwargs):
        in_dim = kwargs.get('in_dim', 12)
        kwargs['in_chans'] = in_chans
        super().__init__(num_classes=num_classes, **kwargs)
        self.pixel_embed.proj = TrafficSignStem(in_chans=in_chans, out_chans=in_dim)

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()


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
        load_pretrained_weights(model, pretrained)
    return model


def LNL_Ti(pretrained=False, **kwargs):
    """Drop-in replacement for ``from LNL import LNL_Ti as small``."""
    return LNL_TS_Ti(pretrained=pretrained, **kwargs)


def LNL_MoEx_Ti(pretrained=False, **kwargs):
    """Drop-in replacement for ``from LNL_MoEx import LNL_MoEx_Ti as small``."""
    return LNL_TS_Ti(pretrained=pretrained, **kwargs)


def load_lnl_ts_checkpoint(path, map_location='cpu'):
    """Load either a pure state_dict or a training checkpoint dict."""
    checkpoint = torch.load(path, map_location=map_location)
    if isinstance(checkpoint, dict):
        for key in ('model', 'model_state', 'state_dict'):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def load_pretrained_weights(model, pretrained=True):
    if isinstance(pretrained, str):
        candidate_paths = (pretrained,)
    else:
        candidate_paths = DEFAULT_PRETRAINED_PATHS

    for path in candidate_paths:
        if os.path.exists(path):
            state_dict = load_lnl_ts_checkpoint(path, map_location='cpu')
            model.load_state_dict(state_dict)
            return model

    searched = ', '.join(candidate_paths)
    raise FileNotFoundError(
        'LNL-TS pretrained weights were not found. '
        f'Expected one of: {searched}. '
        'Train the model first and save it as pretrained/lnl_ts_ti_gtsrb.pth.'
    )

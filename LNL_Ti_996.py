import torch
import torch.nn as nn

from LNL import LocalViT_TNT, default_cfgs

class TrafficSignTokenPool(nn.Module):
    """Fuse CLS, average patch, and max patch information back to 192 features."""

    def __init__(self, embed_dim=192, drop_rate=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim * 3),
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, tokens):
        cls_token = tokens[:, 0]
        patch_tokens = tokens[:, 1:]
        avg_token = patch_tokens.mean(dim=1)
        max_token = patch_tokens.max(dim=1).values
        return self.proj(torch.cat([cls_token, avg_token, max_token], dim=1))


class LNLTi996(LocalViT_TNT):
    """LocalViT-TNT tiny with a GTSRB-oriented token pooling neck."""

    def __init__(
        self,
        num_classes=1000,
        img_size=224,
        drop_rate=0.05,
        attn_drop_rate=0.0,
        drop_path_rate=0.08,
        head_drop_rate=0.10,
        **kwargs,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=16,
            in_chans=3,
            num_classes=num_classes,
            embed_dim=192,
            in_dim=12,
            depth=12,
            num_heads=3,
            in_num_head=3,
            mlp_ratio=4.0,
            qkv_bias=False,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            **kwargs,
        )
        self.default_cfg = default_cfgs["tnt_t_conv_patch16_224"]
        self.token_pool = TrafficSignTokenPool(self.embed_dim, drop_rate=head_drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        attn_weights = []
        batch_size = x.shape[0]
        pixel_embed = self.pixel_embed(x, self.pixel_pos)

        patch_embed = self.norm2_proj(
            self.proj(self.norm1_proj(pixel_embed.reshape(batch_size, self.num_patches, -1)))
        )
        patch_embed = torch.cat((self.cls_token.expand(batch_size, -1, -1), patch_embed), dim=1)
        patch_embed = patch_embed + self.patch_pos
        patch_embed = self.pos_drop(patch_embed)

        for block in self.blocks:
            pixel_embed, patch_embed, weights = block(pixel_embed, patch_embed)
            attn_weights.append(weights)

        patch_embed = self.norm(patch_embed)
        return self.token_pool(patch_embed), attn_weights


def LNL_Ti_996(pretrained=False, **kwargs):
    if pretrained:
        print("LNL_Ti_996 keeps the LNL/TNT architecture and trains from scratch; pretrained=True is ignored.")
    return LNLTi996(**kwargs)


def LNL_Ti_996_GTSRB(pretrained=False, **kwargs):
    kwargs.setdefault("num_classes", 43)
    return LNL_Ti_996(pretrained=pretrained, **kwargs)


def LNL_Ti_Improved(pretrained=False, **kwargs):
    return LNL_Ti_996(pretrained=pretrained, **kwargs)

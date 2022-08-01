# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# Copyright 2020 Ross Wightman
# Modified Model definition

import torch
import torch.nn as nn
from functools import partial
import math
import warnings
import torch.nn.functional as F
import numpy as np

from lib.models.vit_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from lib.models.helpers import load_pretrained
from lib.models.vit_utils import DropPath, to_2tuple, trunc_normal_

from .build import MODEL_REGISTRY
from torch import einsum
from einops import rearrange, reduce, repeat


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "first_conv": "patch_embed.proj",
        "classifier": "head",
        **kwargs,
    }


default_cfgs = {
    "vit_base_patch16_224": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth",
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
}


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        with_qkv=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        if self.with_qkv:
            qkv = (
                self.qkv(x)
                .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(
                0, 2, 1, 3
            )
            q, k, v = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
            x = self.proj(x)
            x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.1,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attention_type="divided_space_time",
    ):
        super().__init__()
        self.attention_type = attention_type
        assert attention_type in [
            "divided_space_time",
            "space_only",
            "joint_space_time",
        ]

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        ## Temporal Attention Parameters
        if self.attention_type == "divided_space_time":
            self.temporal_norm1 = norm_layer(dim)
            self.temporal_attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
            self.temporal_fc = nn.Linear(dim, dim)

        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, B, T, W):
        num_spatial_tokens = (x.size(1) - 1) // T
        H = num_spatial_tokens // W

        if self.attention_type in ["space_only", "joint_space_time"]:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        elif self.attention_type == "divided_space_time":
            ## Temporal
            xt = x[:, 1:, :]
            xt = rearrange(xt, "b (h w t) m -> (b h w) t m", b=B, h=H, w=W, t=T)
            res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt)))
            res_temporal = rearrange(
                res_temporal, "(b h w) t m -> b (h w t) m", b=B, h=H, w=W, t=T
            )
            res_temporal = self.temporal_fc(res_temporal)
            xt = x[:, 1:, :] + res_temporal

            ## Spatial
            init_cls_token = x[:, 0, :].unsqueeze(1)
            cls_token = init_cls_token.repeat(1, T, 1)
            cls_token = rearrange(cls_token, "b t m -> (b t) m", b=B, t=T).unsqueeze(1)
            xs = xt
            xs = rearrange(xs, "b (h w t) m -> (b t) (h w) m", b=B, h=H, w=W, t=T)
            xs = torch.cat((cls_token, xs), 1)
            res_spatial = self.drop_path(self.attn(self.norm1(xs)))

            ### Taking care of CLS token
            cls_token = res_spatial[:, 0, :]
            cls_token = rearrange(cls_token, "(b t) m -> b t m", b=B, t=T)
            cls_token = torch.mean(cls_token, 1, True)  ## averaging for every frame
            res_spatial = res_spatial[:, 1:, :]
            res_spatial = rearrange(
                res_spatial, "(b t) (h w) m -> b (h w t) m", b=B, h=H, w=W, t=T
            )
            res = res_spatial
            x = xt

            ## Mlp
            x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.proj(x)
        W = x.size(-1)
        x = x.flatten(2).transpose(1, 2)
        return x, T, W


class VisionTransformer(nn.Module):
    """Vision Transformer"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        hybrid_backbone=None,
        norm_layer=nn.LayerNorm,
        num_frames=8,
        attention_type="divided_space_time",
        dropout=0.0,
        label_emb="",
        mlp=0,
        text_model="",
        lp=False,
        num_seg=0,
        extra_tr="",
        normp=False,
        drope=0.0,
        cfg=None,
    ):
        super().__init__()
        self.attention_type = attention_type
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        self.act = nn.Softmax(dim=1)

        ## Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        if self.attention_type != "space_only":
            self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
            self.time_drop = nn.Dropout(p=drop_rate)

        ## Attention Blocks
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, self.depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    attention_type=self.attention_type,
                )
                for i in range(self.depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.mlp = mlp
        self.label = label_emb
        if not label_emb == "" and text_model == "":
            self.label_emb = torch.load(label_emb)
            self.head = (
                nn.Linear(embed_dim, self.label_emb.shape[1])
                if num_classes > 0
                else nn.Identity()
            )
            if mlp == 3:
                self.text_mlp = nn.Sequential(
                    nn.Linear(self.label_emb.shape[1], self.label_emb.shape[1]),
                    nn.ReLU(),
                    nn.Linear(self.label_emb.shape[1], self.label_emb.shape[1]),
                    nn.ReLU(),
                    nn.Linear(self.label_emb.shape[1], self.label_emb.shape[1]),
                )
                self.apply(self._init_weights)
            else:
                self.apply(self._init_weights)
                self.head2 = nn.Linear(num_classes, self.label_emb.shape[0], bias=False)
                self.head2.weight = nn.Parameter(torch.load(label_emb))
        elif not label_emb == "" and not text_model == "":
            self.head = (
                nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
            )
            self.apply(self._init_weights)
            self.label_emb = torch.load(label_emb)
        else:
            self.label_emb = False
            self.head = (
                nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
            )
            self.apply(self._init_weights)
        self.text = text_model
        if num_classes == 10588:
            self.step2task = torch.load(
                "/data/home/medhini/video-distant-supervision/data/step_to_task.pth"
            ).transpose(0, 1)
        if text_model == "paraphrase-mpnet-base-v2" and not len(cfg.TRAIN.TEXT_EMB) > 1:
            if not (not label_emb == "" and not text_model == ""):
                self.head = nn.Linear(embed_dim, 768)
            if lp:
                self.text_lp = nn.Linear(768, 768)

            self.apply(self._init_weights)
            from sentence_transformers import SentenceTransformer

            self.text_model = SentenceTransformer("paraphrase-mpnet-base-v2")
            if lp:
                self.text_model[0].auto_model.pooler.activation = nn.Identity()
            self.lp = lp

        if "-space" in extra_tr and num_seg > 0:
            self.extra_tr = int(extra_tr[0])
            if hasattr(cfg.MODEL, "RET_HEAD") and cfg.MODEL.RET_HEAD > 0:
                self.num_seg = num_seg
                self.ret = cfg.MODEL.RET_HEAD
                self.drope = nn.Dropout(drope)
                self.head = nn.Linear(embed_dim, cfg.MODEL.PRE_CLASSES)
                if extra_tr[0] == "1":
                    self.head_tr = nn.ModuleList(
                        [
                            Block(
                                dim=embed_dim,
                                num_heads=num_heads,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                qk_scale=qk_scale,
                                drop=drope,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[-1],
                                norm_layer=norm_layer,
                                attention_type="space_only",
                            ),
                            norm_layer(embed_dim),
                        ]
                    )
                else:
                    self.head_tr = nn.ModuleList(
                        [
                            nn.ModuleList(
                                [
                                    Block(
                                        dim=embed_dim,
                                        num_heads=num_heads,
                                        mlp_ratio=mlp_ratio,
                                        qkv_bias=qkv_bias,
                                        qk_scale=qk_scale,
                                        drop=drope,
                                        attn_drop=attn_drop_rate,
                                        drop_path=dpr[-(int(extra_tr[0]) - i)],
                                        norm_layer=norm_layer,
                                        attention_type="space_only",
                                    )
                                    for i in range(int(extra_tr[0]))
                                ]
                            ),
                            norm_layer(embed_dim),
                        ]
                    )
                self.head_cls = nn.Linear(embed_dim, num_classes)
                self.apply(self._init_weights)
                print("ret head drop path rate", dpr[-1])
                if "next" in cfg.DATA.PATH_TO_DATA_DIR:
                    self.next = True
                if hasattr(cfg.MODEL, "RET_POS") and cfg.MODEL.RET_POS:
                    self.head_ret_pos_embed = nn.Parameter(
                        torch.zeros(1, num_seg * self.ret, embed_dim)
                    )
                    if hasattr(cfg.MODEL, "RET_POS_MUL") and cfg.MODEL.RET_POS_MUL:
                        self.ret_pos_mul = True
                    else:
                        self.ret_pos_mul = False
            else:
                self.num_seg = num_seg
                self.drope = nn.Dropout(drope)
                if extra_tr[0] == "1":
                    self.head = nn.ModuleList(
                        [
                            Block(
                                dim=embed_dim,
                                num_heads=num_heads,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                qk_scale=qk_scale,
                                drop=drope,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[-1],
                                norm_layer=norm_layer,
                                attention_type="space_only",
                            ),
                            norm_layer(embed_dim),
                        ]
                    )
                else:
                    self.head = nn.ModuleList(
                        [
                            nn.ModuleList(
                                [
                                    Block(
                                        dim=embed_dim,
                                        num_heads=num_heads,
                                        mlp_ratio=mlp_ratio,
                                        qkv_bias=qkv_bias,
                                        qk_scale=qk_scale,
                                        drop=drope,
                                        attn_drop=attn_drop_rate,
                                        drop_path=dpr[-(int(extra_tr[0]) - i)],
                                        norm_layer=norm_layer,
                                        attention_type="space_only",
                                    )
                                    for i in range(int(extra_tr[0]))
                                ]
                            ),
                            norm_layer(embed_dim),
                        ]
                    )
                self.head_cls = nn.Linear(embed_dim, num_classes)
                self.apply(self._init_weights)
                print("drop path rate", dpr[-1])
            self.apply_head_cls_to_all_tokens = cfg.MODEL.APPLY_HEAD_CLS_TO_ALL_TOKENS
            if cfg.MODEL.EXTRA_POS:
                self.head_pos_embed = nn.Parameter(torch.zeros(1, num_seg, embed_dim))
            if cfg.MODEL.MASK_RATIO > 0:
                self.mask_token = nn.Parameter(torch.zeros(embed_dim))
                trunc_normal_(self.mask_token, std=0.02)

        if cfg.TRAIN.DATASET == "Epickitchens":
            self.head_n = nn.Linear(embed_dim, 300)
            self.head_v = nn.Linear(embed_dim, 97)

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)

        ## initialization of temporal attention weights
        if self.attention_type == "divided_space_time":
            i = 0
            for m in self.blocks.modules():
                m_str = str(m)
                if "Block" in m_str:
                    if i > 0:
                        nn.init.constant_(m.temporal_fc.weight, 0)
                        nn.init.constant_(m.temporal_fc.bias, 0)
                    i += 1

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "pos_embed",
            "cls_token",
            "time_embed",
            "head_pos_embed",
            "head_ret_pos_embed",
        }

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x, cls=True):
        B = x.shape[0]
        x, T, W = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        ## resizing the positional embeddings in case they don't match the input at inference
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed
            cls_pos_embed = pos_embed[0, 0, :].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0, 1:, :].unsqueeze(0).transpose(1, 2)
            P = int(other_pos_embed.size(2) ** 0.5)
            H = x.size(1) // W
            other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
            new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode="nearest")
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        ## Time Embeddings
        if self.attention_type != "space_only":
            cls_tokens = x[:B, 0, :].unsqueeze(1)
            x = x[:, 1:]
            x = rearrange(x, "(b t) n m -> (b n) t m", b=B, t=T)
            ## Resizing time embeddings in case they don't match
            if T != self.time_embed.size(1):
                time_embed = self.time_embed.transpose(1, 2)
                new_time_embed = F.interpolate(time_embed, size=(T), mode="nearest")
                new_time_embed = new_time_embed.transpose(1, 2)
                x = x + new_time_embed
            else:
                x = x + self.time_embed
            x = self.time_drop(x)
            x = rearrange(x, "(b n) t m -> b (n t) m", b=B, t=T)
            x = torch.cat((cls_tokens, x), dim=1)

        ## Attention blocks
        for blk in self.blocks:
            x = blk(x, B, T, W)

        ### Predictions for space-only baseline
        if self.attention_type == "space_only":
            x = rearrange(x, "(b t) n m -> b t n m", b=B, t=T)
            x = torch.mean(x, 1)  # averaging predictions for every frame

        x = self.norm(x)
        if cls:
            return x[:, 0]
        else:
            return x

    def forward(self, x, seg_mask=None):
        if len(self.text) > 0 and (
            (hasattr(self, "step2task") and self.training)
            or not hasattr(self, "step2task")
        ):
            x, text = x
        if hasattr(self, "num_seg") and self.num_seg > 0:
            x = rearrange(
                x,
                "b c (m t) h w -> (b m) c t h w",
                m=self.num_seg,
                t=x.shape[2] // self.num_seg,
            )

        x = self.forward_features(x)

        if hasattr(self, "num_seg") and self.num_seg > 0:

            if hasattr(self, "ret") and self.ret > 0:
                ts = self.head(x)

            if seg_mask is not None:
                x = rearrange(x, "(b m) e -> b m e", m=self.num_seg, e=768)
                # Where seg_mask == 0, replace with mask_token
                x[~seg_mask.bool()] = self.mask_token
                x = rearrange(x, "b m e -> (b m) e")

            if hasattr(self, "head_pos_embed"):
                x = x.view(x.shape[0] // self.num_seg, -1, 768) + self.head_pos_embed
            if hasattr(self, "drope"):
                x = self.drope(x)
            if hasattr(self, "ret") and self.ret > 0:
                if self.training:
                    video_f = F.gumbel_softmax(ts, dim=1, tau=0.5)
                else:
                    video_f = F.softmax(ts, dim=1)
                if hasattr(self, "next") and self.next:
                    step_emb = F.embedding(
                        (video_f.topk(k=self.ret, dim=1)[1] + 1).clip(
                            max=video_f.shape[-1] - 1
                        ),
                        weight=self.head.weight,
                    )
                else:
                    step_emb = F.embedding(
                        video_f.topk(k=self.ret, dim=1)[1], weight=self.head.weight
                    )
                num_s = step_emb.shape[0]
                if hasattr(self, "head_ret_pos_embed"):
                    if hasattr(self, "ret_pos_mul") and self.ret_pos_mul:
                        step_emb = (
                            step_emb.view(step_emb.shape[0] // self.num_seg, -1, 768)
                            * self.head_ret_pos_embed
                        )
                    else:
                        step_emb = (
                            step_emb.view(step_emb.shape[0] // self.num_seg, -1, 768)
                            + self.head_ret_pos_embed
                        )
                x = torch.cat(
                    (
                        x.view(-1, self.num_seg, 768),
                        step_emb.view(num_s // self.num_seg, -1, 768),
                    ),
                    1,
                )
                if self.extra_tr == 1:
                    x = self.head_tr[0](x, x.shape[0], self.num_seg * (1 + self.ret), 1)
                else:
                    for i in range(self.extra_tr):
                        x = self.head_tr[0][i](
                            x, x.shape[0], self.num_seg * (1 + self.ret), 1
                        )
                if self.apply_head_cls_to_all_tokens:
                    x = self.head_tr[1](x)
                else:
                    x = self.head_tr[1](x[:, 0, :])
                x = self.head_cls(x)
            else:
                if self.extra_tr == 1:
                    x = self.head[0](
                        x.view(-1, self.num_seg, 768), x.shape[0], self.num_seg, 1
                    )
                else:
                    for i in range(self.extra_tr):
                        x = self.head[0][i](
                            x.view(-1, self.num_seg, 768), x.shape[0], self.num_seg, 1
                        )
                if self.apply_head_cls_to_all_tokens:
                    x = self.head[1](x)
                else:
                    x = self.head[1](x[:, 0, :])
                x = self.head_cls(x)
        elif hasattr(self, "head_n"):
            v = self.head_v(x)
            n = self.head_n(x)
            if not self.training:
                v = self.act(v)
                n = self.act(n)
            return (v, n)
        else:
            x = self.head(x)

        if (
            isinstance(self.label_emb, torch.Tensor)
            and len(self.text) > 0
            and self.training
        ):
            if hasattr(self, "text_model"):
                text_emb = self.text_model(text)["sentence_embedding"]
                text_pred = F.linear(
                    F.normalize(text_emb, 1) * 4.5,
                    F.normalize(self.label_emb.to(x.device), 1) * 4.5,
                )
                return x, text_pred
            else:
                return x, None

        if isinstance(self.label_emb, torch.Tensor) and not len(self.text) > 0:
            if self.mlp:
                text = self.text_mlp(self.label_emb.cuda())
                x = F.linear(x, weight=text)

            else:
                x = self.head2(x)
        if len(self.text) > 0 and not isinstance(self.label_emb, torch.Tensor):
            if not type(self.text_model) == type(" "):
                text_emb = self.text_model(text)["sentence_embedding"]
            else:
                text_emb = text
            if hasattr(self, "text_lp") and self.lp:
                text_emb = self.text_lp(text_emb)
            return (x, text_emb)
        if not self.training:
            x = self.act(x)
        if hasattr(self, "step2task"):
            x = (
                x.unsqueeze(1)
                * (x.unsqueeze(1) == x.topk(k=5, dim=1)[0].unsqueeze(2)).float()
            ).sum(1)
            x = x / x.sum(1, keepdim=True)
            x = F.linear(x, self.step2task.to(x.device))
        if len(x.size()) == 3:
            x = rearrange(x, "b t c -> b (t c)")
        return x


def _conv_filter(state_dict, patch_size=16):
    """convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if "patch_embed.proj.weight" in k:
            if v.shape[-1] != patch_size:
                patch_size = v.shape[-1]
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


@MODEL_REGISTRY.register()
class vit_base_patch16_224(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(vit_base_patch16_224, self).__init__()
        self.pretrained = cfg.MODEL.PRETRAINED
        patch_size = 16
        mlp = cfg.MODEL.MLP
        emb = cfg.TRAIN.LABEL_EMB
        lp = cfg.MODEL.TEXT_LP
        num_seg = cfg.MODEL.NUM_SEG
        extra = cfg.MODEL.EXTRA_TR
        drope = cfg.MODEL.DROP_E
        dpr = cfg.MODEL.DROP_PATH
        self.model = VisionTransformer(
            img_size=cfg.DATA.TRAIN_CROP_SIZE,
            num_classes=cfg.MODEL.NUM_CLASSES,
            patch_size=patch_size,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=dpr,
            num_frames=cfg.DATA.NUM_FRAMES,
            attention_type=cfg.TIMESFORMER.ATTENTION_TYPE,
            label_emb=emb,
            mlp=mlp,
            text_model=cfg.MODEL.TEXT_MODEL,
            lp=lp,
            num_seg=num_seg,
            extra_tr=extra,
            drope=drope,
            cfg=cfg,
            **kwargs
        )

        self.attention_type = cfg.TIMESFORMER.ATTENTION_TYPE
        self.model.default_cfg = default_cfgs["vit_base_patch16_224"]
        self.num_patches = (cfg.DATA.TRAIN_CROP_SIZE // patch_size) * (
            cfg.DATA.TRAIN_CROP_SIZE // patch_size
        )
        pretrained_model = cfg.TIMESFORMER.PRETRAINED_MODEL
        if self.pretrained:
            load_pretrained(
                self.model,
                num_classes=self.model.num_classes,
                in_chans=kwargs.get("in_chans", 3),
                filter_fn=_conv_filter,
                img_size=cfg.DATA.TRAIN_CROP_SIZE,
                num_patches=self.num_patches,
                attention_type=self.attention_type,
                pretrained_model=pretrained_model,
                num_frames=cfg.DATA.NUM_FRAMES,
                pre_num=cfg.MODEL.PRE_CLASSES,
            )
        else:
            print("not loading any pretrained weights!")

        # self.pos_enc = PositionalEncoding(768, 0.2)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=6,
            dropout=0.2,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=2)
        self.fc1 = nn.Linear(768, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x, seg_mask=None):
        x = self.model(x, seg_mask=seg_mask)
        return x

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import numpy as np
import torch.nn.functional as F
from torch import Tensor, nn
import torch
from loguru import logger


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, pos=None, attn_mask=None, save_specificity_opts: dict = None) -> Tensor:
        x_in = x
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None and pos is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                attn_mask=(
                    (attn_mask)[:, None].repeat(1, self.num_heads, 1, 1)
                    if attn_mask is not None
                    else None
                ),
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        if save_specificity_opts is not None:
            mode = save_specificity_opts.get('save_mode', 'output')
            x_to_save = x_in if mode == 'input' else x
            self._save_specificity(x_to_save, k, save_specificity_opts)

        return x

    def _forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _save_specificity(self, x: Tensor, k: Tensor, opts: dict) -> None:
        save_path = opts.get('save_path')
        if not save_path:
            logger.warning("Specificity save path not provided, skipping save.")
            return

        layer_id = opts.get('layer_id', -1)
        top_k = opts.get('top_k', 10)
        patch_start_index = opts.get('patch_start_index', 1) 
        
        attn_type = opts.get('attn_type', 'local')
        num_views = opts.get('num_views', 1)
        tokens_per_view = opts.get('tokens_per_view', k.shape[2])

        # k shape: [B, num_heads, N, head_dim]
        # x shape: [B, N, C]
        
        # We focus on patches only
        if attn_type == 'global' and num_views > 1:
            # Reconstruct dimensions to handle multiple views
            B, num_heads, _, head_dim = k.shape
            
            # Reshape to separate views: [B, H, S, N_view, D]
            try:
                k_reshaped = k.view(B, num_heads, num_views, tokens_per_view, head_dim)
            except Exception as e:
                logger.error(f"Failed to reshape k for specificity: {e}")
                return

            # Select patches: [B, H, S, N_patches, D]
            k_patches = k_reshaped[:, :, :, patch_start_index:, :]
            
            # Flatten back for specificity calc: [B, H, S*N_patches, D]
            k_patches_flat = k_patches.flatten(2, 3) # flatten S and N_patches
            
            # Normalize keys
            k_norm = F.normalize(k_patches_flat, dim=-1) # [B, H, S*Np, D]
            
            # Mean key across patches
            mean_k = k_norm.mean(dim=2, keepdim=True) # [B, H, 1, D]
            
            # Cosine similarity
            similarity = (k_norm * mean_k).sum(dim=-1) # [B, H, S*Np]
            
            # Average similarity across heads
            score = similarity.mean(dim=1) # [B, S*Np]
            
            # Specificity is negative similarity
            specificity = -score 
            
            # Select top k
            # Current filtering only uses top_k; threshold is not applied in this implementation.
            k_val = min(top_k, specificity.shape[1])
            _, topk_indices = torch.topk(specificity, k=k_val, dim=-1) # [B, K]
            
            # Recover indices
            # view_indices is the per-token view id within the current sequence (0..num_views-1)
            # It tells which image/view the selected token came from, not a global dataset index.
            patches_per_view = tokens_per_view - patch_start_index
            view_indices = topk_indices // patches_per_view
            patch_indices = topk_indices % patches_per_view # Relative to patch start
            
            # To gather tokens, we need to map topk_indices back to original x indices
            # Original index = (view_index * tokens_per_view) + (patch_start_index + patch_index)
            original_indices = (view_indices * tokens_per_view) + (patch_start_index + patch_indices)
            
            # Gather tokens from x
            B_dim, N_in, C_dim = x.shape
            selected_tokens = torch.gather(
                x, 
                1, 
                original_indices.unsqueeze(-1).expand(-1, -1, C_dim)
            )
            
            # Prepare saving
            to_save = {
                "tokens": selected_tokens.detach().float().cpu().numpy(),
                "view_indices": view_indices.detach().cpu().numpy(),
                "patch_indices": patch_indices.detach().cpu().numpy(), # Relative to patch start (0-based for image patch)
                "scores": torch.gather(specificity, 1, topk_indices).detach().float().cpu().numpy()
            }
        else: # not work
            assert None, "Specificity calculation currently only supports 'global' attention with multiple views."
            k_patches = k[:, :, patch_start_index:, :] # [B, H, N_p, D_h]
            
            # Normalize keys
            k_norm = F.normalize(k_patches, dim=-1)
            
            # Mean key across patches
            mean_k = k_norm.mean(dim=2, keepdim=True) # [B, H, 1, D_h]
            
            # Cosine similarity
            similarity = (k_norm * mean_k).sum(dim=-1) # [B, H, N_p]
            
            # Average similarity across heads
            score = similarity.mean(dim=1) # [B, N_p]
            
            # Specificity is negative similarity. Smallest sim = "most special"
            specificity = -score 
            
            # Select top k
            k_val = min(top_k, specificity.shape[1])
            _, topk_indices = torch.topk(specificity, k=k_val, dim=-1) # [B, K]
            
            # Map indices back to x space
            x_patches = x[:, patch_start_index:, :] # [B, N_p, C]
            B_dim, N_p, C_dim = x_patches.shape
            
            # Gather tokens
            selected_tokens = torch.gather(
                x_patches, 
                1, 
                topk_indices.unsqueeze(-1).expand(-1, -1, C_dim)
            )

            to_save = {
                "tokens": selected_tokens.detach().float().cpu().numpy(),
                "indices": topk_indices.detach().cpu().numpy(), # Indices within the patch set
                "scores": torch.gather(specificity, 1, topk_indices).detach().float().cpu().numpy()
            }
            
        # Move to CPU for saving
        filename = f"layer_{layer_id}.npy"
        
        batch_index = opts.get('batch_index', None)
        if batch_index is not None:
             filename = f"layer_{layer_id}_seq_{batch_index}.npy"
             
        full_path = os.path.join(save_path, filename)
        os.makedirs(save_path, exist_ok=True)
        
        np.save(full_path, to_save)

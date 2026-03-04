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

    def _save_specificity_by_key(self, x: Tensor, k: Tensor, opts: dict) -> None:
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

            # Flatten tokens to B x D for saving
            _D = selected_tokens.shape[2]
            tokens_to_save = selected_tokens.reshape(-1, _D)
            
            # Prepare saving
            to_save = {
                "tokens": tokens_to_save.detach().float().cpu().numpy(),
                "view_indices": view_indices.detach().cpu().numpy(),
                "patch_indices": patch_indices.detach().cpu().numpy(), # Relative to patch start (0-based for image patch)
                "scores": torch.gather(specificity, 1, topk_indices).detach().float().cpu().numpy()
            }
        else:
            # Local / single-view attention path mirrors the global save format
            k_patches = k[:, :, patch_start_index:, :] # [B, H, N_p, D_h]

            # Normalize keys then compute cosine similarity to the per-head mean key
            k_norm = F.normalize(k_patches, dim=-1)
            mean_k = k_norm.mean(dim=2, keepdim=True) # [B, H, 1, D_h]
            similarity = (k_norm * mean_k).sum(dim=-1) # [B, H, N_p]

            # Average similarity across heads and invert to get specificity
            score = similarity.mean(dim=1) # [B, N_p]
            specificity = -score

            # Select top-k most specific patches
            k_val = min(top_k, specificity.shape[1])
            _, topk_indices = torch.topk(specificity, k=k_val, dim=-1) # [B, K]

            # Gather tokens from the input/output tensor aligned with patch indices
            x_patches = x[:, patch_start_index:, :] # [B, N_p, C]
            B_dim, _, C_dim = x_patches.shape
            selected_tokens = torch.gather(
                x_patches,
                1,
                topk_indices.unsqueeze(-1).expand(-1, -1, C_dim)
            )

            # Flatten tokens to B x D for saving
            _D = selected_tokens.shape[2]
            tokens_to_save = selected_tokens.reshape(-1, _D)

            # Align keys with global format fields for downstream consumers
            patch_indices = topk_indices
            view_indices = torch.zeros_like(patch_indices) # single-view placeholder

            to_save = {
                "tokens": tokens_to_save.detach().float().cpu().numpy(),
                "view_indices": view_indices.detach().cpu().numpy(),
                "patch_indices": patch_indices.detach().cpu().numpy(),
                "scores": torch.gather(specificity, 1, topk_indices).detach().float().cpu().numpy(),
            }
            
        # Move to CPU for saving
        filename = f"layer_{layer_id}.npy"
        
        batch_index = opts.get('batch_index', None)
        if batch_index is not None:
             filename = f"layer_{layer_id}_seq_{batch_index}.npy"
             
        full_path = os.path.join(save_path, filename)
        os.makedirs(save_path, exist_ok=True)
        
        total_candidates = specificity.shape[1] if 'specificity' in locals() else -1
        logger.debug(
            f"[specificity] layer={layer_id} attn={attn_type} select_topk={k_val}/{total_candidates} "
            f"tokens_shape={to_save['tokens'].shape} save_path={full_path}"
        )

        np.save(full_path, to_save)

    def _save_specificity(self, x: Tensor, k: Tensor, opts: dict) -> None:
        """Specificity based on token features x (no dependence on k)."""
        save_path = opts.get('save_path')
        if not save_path:
            logger.warning("Specificity save path not provided, skipping save.")
            return

        layer_id = opts.get('layer_id', -1)
        top_k = opts.get('top_k', 10)
        patch_start_index = opts.get('patch_start_index', 1)

        attn_type = opts.get('attn_type', 'local')
        num_views = opts.get('num_views', 1)
        tokens_per_view = opts.get('tokens_per_view', x.shape[1])
        prev_local_tokens = opts.get('prev_local_tokens', None)

        # Deep copy x for safety, then compute similarity over flattened tokens (batch collapsed)
        x_tokens = x.clone()
        B, N, C = x_tokens.shape

        if attn_type == 'global' and num_views > 1:
            try:
                x_view = x_tokens.view(B, num_views, tokens_per_view, C)
            except Exception as e:
                logger.error(f"Failed to reshape x for specificity: {e}")
                return

            # Drop special tokens per view
            x_patches = x_view[:, :, patch_start_index:, :]  # [B, S, N_p, C]
            x_flat = x_patches.reshape(1, -1, C)  # collapse batch+views -> [1, B*S*N_p, C]

            x_norm = F.normalize(x_flat, dim=-1)
            mean_x = x_norm.mean(dim=1, keepdim=True)  # [1, 1, C]
            similarity = (x_norm * mean_x).sum(dim=-1)  # [1, B*S*N_p]
            specificity = -similarity

            total_patches_per_sample = num_views * (tokens_per_view - patch_start_index)
            k_val = min(top_k, specificity.shape[1])
            _, topk_indices = torch.topk(specificity, k=k_val, dim=-1)  # [1, K]
            flat_idx = topk_indices.squeeze(0)  # [K]

            batch_indices = flat_idx // total_patches_per_sample
            rem = flat_idx % total_patches_per_sample
            view_indices = rem // (tokens_per_view - patch_start_index)
            patch_indices = rem % (tokens_per_view - patch_start_index)

            original_indices = (view_indices * tokens_per_view) + (patch_start_index + patch_indices)

            selected_tokens = x_tokens[batch_indices, original_indices, :]  # [K, C]

            tokens_to_save = selected_tokens.reshape(-1, C)

            scores = specificity.view(-1)[flat_idx]

            local_tokens = None
            concat_tokens = None
            if prev_local_tokens is not None:
                try:
                    if prev_local_tokens.dim() == 4 and prev_local_tokens.shape[:3] == (B, num_views, tokens_per_view):
                        local_indices = patch_start_index + patch_indices
                        local_tokens = prev_local_tokens[batch_indices, view_indices, local_indices, :]
                        concat_tokens = torch.cat([local_tokens, selected_tokens], dim=-1)
                    else:
                        logger.warning(
                            "prev_local_tokens shape mismatch, skip concat: expected (B,S,N,C)="
                            f"({B},{num_views},{tokens_per_view},C) got {tuple(prev_local_tokens.shape)}"
                        )
                except Exception as e:
                    logger.error(f"Failed to gather local tokens for concat: {e}")

            to_save = {
                "tokens": tokens_to_save.detach().float().cpu().numpy(),
                "batch_indices": batch_indices.detach().cpu().numpy(),
                "view_indices": view_indices.detach().cpu().numpy(),
                "patch_indices": patch_indices.detach().cpu().numpy(),
                "scores": scores.detach().float().cpu().numpy(),
            }
            # if local_tokens is not None:
            #     to_save["local_tokens"] = local_tokens.detach().float().cpu().numpy()
            if concat_tokens is not None:
                to_save["tokens"] = concat_tokens.detach().float().cpu().numpy()
        else:
            x_patches = x_tokens[:, patch_start_index:, :]  # [B, N_p, C]
            x_flat = x_patches.reshape(1, -1, C)  # collapse batch -> [1, B*N_p, C]

            x_norm = F.normalize(x_flat, dim=-1)
            mean_x = x_norm.mean(dim=1, keepdim=True)  # [1, 1, C]
            similarity = (x_norm * mean_x).sum(dim=-1)  # [1, B*N_p]
            specificity = -similarity

            tokens_per_sample = tokens_per_view - patch_start_index
            k_val = min(top_k, specificity.shape[1])
            _, topk_indices = torch.topk(specificity, k=k_val, dim=-1)  # [1, K]
            flat_idx = topk_indices.squeeze(0)  # [K]

            batch_indices = flat_idx // tokens_per_sample
            patch_indices = flat_idx % tokens_per_sample
            original_indices = patch_start_index + patch_indices

            selected_tokens = x_tokens[batch_indices, original_indices, :]  # [K, C]

            tokens_to_save = selected_tokens.reshape(-1, C)

            scores = specificity.view(-1)[flat_idx]

            view_indices = torch.zeros_like(patch_indices)

            to_save = {
                "tokens": tokens_to_save.detach().float().cpu().numpy(),
                "batch_indices": batch_indices.detach().cpu().numpy(),
                "view_indices": view_indices.detach().cpu().numpy(),
                "patch_indices": patch_indices.detach().cpu().numpy(),
                "scores": scores.detach().float().cpu().numpy(),
            }

        filename = f"layer_{layer_id}.npy"
        batch_index = opts.get('batch_index', None)
        if batch_index is not None:
            filename = f"layer_{layer_id}_seq_{batch_index}.npy"

        full_path = os.path.join(save_path, filename)
        os.makedirs(save_path, exist_ok=True)

        total_candidates = specificity.shape[1] if 'specificity' in locals() else -1
        logger.debug(
            f"[specificity] layer={layer_id} attn={attn_type} select_topk={k_val}/{total_candidates} \n"
            f"tokens_shape={to_save['tokens'].shape} save_path={full_path}\n"
            f"batch_indices={to_save['batch_indices'].shape}\n"
            f"view_indices={to_save['view_indices'].shape}\n"
            f"patch_indices={to_save['patch_indices'].shape}\n"
            f"scores={to_save['scores'].shape}"
        )

        np.save(full_path, to_save)

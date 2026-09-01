#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseLoss(nn.Module):
    def __init__(self):
        super(BaseLoss, self).__init__()

    def forward(self, preds, targets, weight=None):
        if isinstance(preds, list):
            N = len(preds)
            if weight is None:
                weight = preds[0].new_ones(1)

            errs = [self._forward(preds[n], targets[n], weight[n])
                    for n in range(N)]
            err = torch.mean(torch.stack(errs))

        elif isinstance(preds, torch.Tensor):
            if weight is None:
                weight = preds.new_ones(1)
            err = self._forward(preds, targets, weight)

        return err


class L1Loss(BaseLoss):
    def __init__(self):
        super(L1Loss, self).__init__()

    def _forward(self, pred, target, weight):
        return torch.mean(weight * torch.abs(pred - target))


class L2Loss(BaseLoss):
    def __init__(self):
        super(L2Loss, self).__init__()

    def _forward(self, pred, target, weight):
        return torch.mean(weight * torch.pow(pred - target, 2))


class MSELoss(BaseLoss):
    def __init__(self):
        super(MSELoss, self).__init__()

    def _forward(self, pred, target):
        return F.mse_loss(pred, target)


class BCELoss(BaseLoss):
    def __init__(self):
        super(BCELoss, self).__init__()

    def _forward(self, pred, target, weight):
        return F.binary_cross_entropy(pred, target, weight=weight)


class BCEWithLogitsLoss(BaseLoss):
    def __init__(self):
        super(BCEWithLogitsLoss, self).__init__()

    def _forward(self, pred, target, weight):
        return F.binary_cross_entropy_with_logits(pred, target, weight=weight)


class DenseAVContrastiveLoss(nn.Module):
    """
    Computes contrastive loss identical to DenseAV but expects separately 
    passed visual and audio features. Handles different batch sizes for
    Semantic (B_sem) and Spatial (B_spa).
    """
    def __init__(self, loss_type="nce", semantic_temperature=0.07, spatial_temperature=0.2):
        super(DenseAVContrastiveLoss, self).__init__()
        self.loss_type = loss_type
        self.semantic_temperature = semantic_temperature
        self.spatial_temperature = spatial_temperature
        
        if self.loss_type == "nce":
            self.log_temp_sem = nn.Parameter(torch.tensor(semantic_temperature).log())
            self.log_temp_spa = nn.Parameter(torch.tensor(spatial_temperature).log())
    
    @property
    def temp_sem(self):
        return self.log_temp_sem.exp().clamp(min=1e-4, max=100.0)

    @property
    def temp_spa(self):
        return self.log_temp_spa.exp().clamp(min=1e-4, max=100.0)

    def compute_sim_matrix(self, v_feat: torch.Tensor, a_feat: torch.Tensor, normalize: bool = False, mode: str = "semantic") -> torch.Tensor:
        """
        Calculates similarity matrix between visual and audio features.
        v_feat: [B, D, H, W]
        a_feat: [B, D, F, T]
        normalize: boolean to apply L2 normalization before dot product
        mode: "semantic" or "spatial"
        Returns: [B, B] similarity matrix
        """
        B_v, D_v, H, W = v_feat.shape
        B_a, D_a, F_dim, T_dim = a_feat.shape
        
        assert D_v == D_a, "Feature dimensions must match"
        
        # Flatten spatial and temporal dimensions
        v_flat = v_feat.reshape(B_v, D_v, H * W)       # [B_v, D, HW]
        a_flat = a_feat.reshape(B_a, D_a, F_dim * T_dim) # [B_a, D, FT]

        if mode == "semantic":
            if normalize:
                v_flat = F.normalize(v_flat, p=2, dim=1)
                a_flat = F.normalize(a_flat, p=2, dim=1)
                
            # Cross-batch patch-wise similarity: [B_v, D, HW] x [B_a, D, FT] -> [B_v, B_a, HW, FT]
            sim = torch.einsum("bdp, cdq -> bcpq", v_flat, a_flat)
            # For each audio bin, find the best matching visual patch (localization)
            sim_hw_max = sim.max(dim=2).values # [B_v, B_a, FT]
            sim_ft = sim_hw_max.mean(dim=2)    # [B_v, B_a]
            return sim_ft
            
        elif mode == "spatial":
            if normalize:
                v_flat = F.normalize(v_flat, p=2, dim=1)
                a_flat = F.normalize(a_flat, p=2, dim=1)
                
            # Cross-batch patch-wise similarity: [B_v, D, HW] x [B_a, D, FT] -> [B_v, B_a, HW, FT]
            sim = torch.einsum("bdp, cdq -> bcpq", v_flat, a_flat)
            # For each audio bin, find the best matching visual patch (localization)
            sim_hw_max = sim.max(dim=2).values # [B_v, B_a, FT]
            sim_ft = sim_hw_max.mean(dim=2)    # [B_v, B_a]
            return sim_ft
            
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
    def contrast_loss(self, sims: torch.Tensor, temp: torch.Tensor) -> torch.Tensor:
        """
        Calculates infoNCE loss for a batch similarity matrix
        sims: [B, B]
        """
        b = sims.shape[0]
        # Avoid penalizing the exact same instance in positive matrix if margin used
        # For NCE, we scale by temperature
        sims = sims / temp
        
        sims_1 = sims
        sims_2 = sims.permute(1, 0)
        
        labels = torch.arange(0, b, device=sims.device)
        
        if self.loss_type == "nce":
            nce_loss = 0.5 * F.cross_entropy(sims_1, labels) + \
                       0.5 * F.cross_entropy(sims_2, labels)
            return nce_loss
        else:
            raise ValueError(f"Unknown loss type {self.loss_type}")

    def forward(self, 
                v_sem: torch.Tensor, a_sem: torch.Tensor, 
                v_spa: torch.Tensor, a_spa: torch.Tensor,
                loss_weights: dict = {"semantic": 1.0, "spatial": 1.0},
                s: int = 0,
                norm_semantic: bool = False,
                norm_spatial: bool = False):
        """
        v_sem: [B_sem, D, H, W]
        a_sem: [B_sem, D, F, T]
        v_spa: [B_spa, D, H, W]
        a_spa: [B_spa, D, F, T]
        """
        # Semantic Loss
        loss_sem = torch.tensor(0.0, device=v_sem.device)
        sim_sem = None
        if v_sem.shape[0] > 0:
            sim_sem = self.compute_sim_matrix(v_sem, a_sem, normalize=norm_semantic, mode="semantic")
            loss_sem = self.contrast_loss(sim_sem, self.temp_sem)
            
        # Spatial Loss
        loss_spa = torch.tensor(0.0, device=v_sem.device)
        sim_spa = None
        if v_spa.shape[0] > 0:
            if s > 0 and v_spa.shape[0] > s:
                b = v_spa.shape[0] - s
                spatial_indices = [0] + list(range(b, b + s))
                v_spa_selected = v_spa[spatial_indices]
                a_spa_selected = a_spa[spatial_indices]
                sim_spa = self.compute_sim_matrix(v_spa_selected, a_spa_selected, normalize=norm_spatial, mode="spatial")
            else:
                sim_spa = self.compute_sim_matrix(v_spa, a_spa, normalize=norm_spatial, mode="spatial")
            loss_spa = self.contrast_loss(sim_spa, self.temp_spa)
            
        total_loss = loss_weights["semantic"] * loss_sem + loss_weights["spatial"] * loss_spa
        
        return {
            "loss": total_loss,
            "loss_semantic": loss_sem,
            "loss_spatial": loss_spa,
            "sim_sem": sim_sem,
            "sim_spa": sim_spa
        }

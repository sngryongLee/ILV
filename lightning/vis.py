import torch
import numpy as np


def vis_slice(output, batch):
    outputs = {}

    pred_vol = output["vol"].detach().cpu()
    gt_vol = batch["volume"].detach().cpu()

    # Remove channel dim if exists
    if pred_vol.ndim == 5:
        pred_vol = pred_vol[:, 0]
    if gt_vol.ndim == 5:
        gt_vol = gt_vol[:, 0]

    B, D, H, W = pred_vol.shape

    # --- Z-slice: shape (B, H, W, 1)
    pred_z = pred_vol[:, D // 2, :, :].unsqueeze(-1)
    gt_z   = gt_vol[:, D // 2, :, :].unsqueeze(-1)

    # --- Y-slice: shape (B, D, W) → reshape to (B, H, W, 1)
    pred_y = pred_vol[:, :, H // 2, :].permute(0, 2, 1).unsqueeze(-1)  # (B, W, D, 1) → (B, H, W, 1)
    gt_y   = gt_vol[:, :, H // 2, :].permute(0, 2, 1).unsqueeze(-1)

    # --- X-slice: shape (B, D, H) → reshape to (B, H, W, 1)
    pred_x = pred_vol[:, :, :, W // 2].permute(0, 2, 1).unsqueeze(-1)  # (B, H, D, 1)
    gt_x   = gt_vol[:, :, :, W // 2].permute(0, 2, 1).unsqueeze(-1)

    # --- Concatenate along width axis
    pred_slice = torch.cat([pred_z, pred_y, pred_x], dim=2)  # (B, H, W * 3, 1)
    gt_slice   = torch.cat([gt_z, gt_y, gt_x], dim=2)

    outputs.update({f"gt_rgb":gt_slice.float().numpy(), f"pred_rgb":pred_slice.float().numpy()})
    return outputs


def vis_images(output, batch):
    # if 'image' in output:
    #     return vis_appearance_depth(output, batch)
    # else:
    return vis_slice(output, batch)

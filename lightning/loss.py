import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from pytorch_msssim import MS_SSIM

# ===========================================================
#                    BASE LOSS (PRETRAIN)
# ===========================================================

class Losses(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.log10 = torch.log(torch.tensor(10.))
        self.ssim = MS_SSIM(data_range=1.0, size_average=True, channel=1)

    def forward(self, batch, output, iter):
        scalar_stats = {}
        loss_total = 0.0

        B, V, H, W = batch["tar_xray"].shape[:-1]
        tar_xray = batch["tar_xray"].permute(0, 2, 1, 3, 4).reshape(B, H, V * W, 1)

        # -----------------------
        # Image loss 
        # -----------------------
        if "image" in output:
            pred_img = torch.nan_to_num(output["image"], nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
            tar_xray = torch.nan_to_num(tar_xray, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)

            l1 = torch.abs(pred_img - tar_xray).mean()
            mse = (pred_img - tar_xray).pow(2).mean()
            psnr = -10.0 * torch.log(mse.detach()) / self.log10.to(mse.device)

            scalar_stats.update({
                "L1": l1.detach(),
                "psnr": psnr.detach(),
            })

            with autocast(enabled=False):
                ssim_val = self.ssim(pred_img.permute(0, 3, 1, 2), tar_xray.permute(0, 3, 1, 2))
                loss_ssim = self.cfg.loss.lambda_ssim * (1 - ssim_val)
                scalar_stats["ssim"] = ssim_val.detach()

            loss_color = l1 * self.cfg.loss.lambda_color
            loss_total += loss_color + loss_ssim

        # -----------------------
        # Volume loss
        # -----------------------
        if "vol" in output:
            pred_vol = torch.nan_to_num(output["vol"], nan=0.0, posinf=1.0, neginf=0.0)
            gt_vol = torch.nan_to_num(batch["volume"], nan=0.0, posinf=1.0, neginf=0.0)

            mse = (pred_vol - gt_vol).pow(2).mean()
            loss_vol = mse * self.cfg.loss.lambda_vol
            psnr3d = -10.0 * torch.log(mse.detach()) / self.log10.to(pred_vol.device)
            scalar_stats.update({
                "volume_mse": mse.detach(),
                "vol_psnr": psnr3d.detach(),
            })
            loss_total += loss_vol

        # -----------------------
        # Refine volume loss
        # -----------------------
        if "vol_refine" in output:
            pred_refine = torch.nan_to_num(output["vol_refine"], nan=0.0, posinf=1.0, neginf=0.0)
            gt_vol = torch.nan_to_num(batch["volume"], nan=0.0, posinf=1.0, neginf=0.0)

            mse_refine = (pred_refine - gt_vol).pow(2).mean()
            loss_refine = mse_refine * self.cfg.loss.lambda_refine_vol
            psnr_refine = -10.0 * torch.log(mse_refine.detach()) / self.log10.to(pred_refine.device)
            scalar_stats.update({
                "volume_mse_refine": mse_refine.detach(),
                "vol_psnr_refine": psnr_refine.detach(),
            })
            loss_total += loss_refine



        return loss_total, scalar_stats
    




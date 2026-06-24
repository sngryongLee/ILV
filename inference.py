import os
import argparse
import cv2
import csv
import json
import torch
import imageio
import nibabel as nib
import numpy as np
from contextlib import nullcontext
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from pytorch_msssim import ssim
from skimage.metrics import structural_similarity

from dataLoader import dataset_dict
from lightning.system import system


def setup_cpu_threads(n_thread=1):
    os.environ["MKL_NUM_THREADS"] = str(n_thread)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_thread)
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_thread)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_thread)

def normalize_uint8(vol):
    return np.clip(vol * 255, 0, 255).astype(np.uint8)


def normalize_tensor(img):
    return img / img.max().clamp_min(1e-8)


def save_slice_image(gt_vol, pred_vol, save_dir, name):
    gt, pred = normalize_uint8(gt_vol), normalize_uint8(pred_vol)

    D, H, W = gt.shape
    z_idx, y_idx, x_idx = D // 2, H // 2, W // 2

    slices_gt = [gt[z_idx], gt[:, y_idx], gt[:, :, x_idx]]
    slices_pred = [pred[z_idx], pred[:, y_idx], pred[:, :, x_idx]]

    merged = [np.concatenate([g, p], axis=1) for g, p in zip(slices_gt, slices_pred)]
    final_img = np.concatenate(merged, axis=0)

    os.makedirs(os.path.join(save_dir, "slices"), exist_ok=True)
    imageio.imwrite(os.path.join(save_dir, "slices", f"{name}.png"), final_img)


def save_volume(vol, save_dir, name):
    volume_np = vol.squeeze(0).detach().cpu().numpy().astype(np.float32)
    nii = nib.Nifti1Image(volume_np, affine=np.eye(4))
    nib.save(nii, os.path.join(save_dir, f"{name}.nii.gz"))


def as_tensor(value):
    if torch.is_tensor(value):
        return value
    return torch.from_numpy(value.copy())


def metric_vol(img1, img2, metric="psnr", pixel_max=1.0):
    """Metrics for volume. img1 must be GT."""
    assert metric in ["psnr", "ssim"]
    img1 = as_tensor(img1).float()
    img2 = as_tensor(img2).float()

    if metric == "psnr":
        if pixel_max is None:
            pixel_max = img1.max()
        mse_out = torch.mean((img1 - img2) ** 2)
        psnr_out = 10 * torch.log10(pixel_max**2 / mse_out.float().clamp_min(1e-12))
        return psnr_out.item(), None

    elif metric == "ssim":
        ssims = []
        for axis in [0, 1, 2]:
            results = []
            count = 0
            for i in range(img1.shape[axis]):
                if axis == 0:
                    slice1, slice2 = img1[i, :, :], img2[i, :, :]
                elif axis == 1:
                    slice1, slice2 = img1[:, i, :], img2[:, i, :]
                elif axis == 2:
                    slice1, slice2 = img1[:, :, i], img2[:, :, i]
                else:
                    raise NotImplementedError

                if slice1.max() > 0:
                    result = ssim(slice1[None, None], slice2[None, None], data_range=1)
                    count += 1
                else:
                    result = 0
                results.append(result)

            results = torch.stack([r.detach().cpu() if torch.is_tensor(r) else torch.tensor(r) for r in results])
            mean_results = torch.sum(results) / max(count, 1)
            ssims.append(mean_results.item())

        return float(np.mean(ssims)), ssims


def real_3d_ssim(img1, img2, data_range=1.0):
    """Compute skimage structural_similarity directly on the full 3D volume."""
    if torch.is_tensor(img1):
        img1 = img1.detach().float().cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.detach().float().cpu().numpy()

    img1 = np.squeeze(img1).astype(np.float32, copy=False)
    img2 = np.squeeze(img2).astype(np.float32, copy=False)
    return float(structural_similarity(img1, img2, data_range=data_range))


def load_system_for_inference(cfg, device):
    checkpoint = torch.load(cfg.infer.ckpt_path, map_location=device)

    is_weight_only = (
        isinstance(checkpoint, dict)
        and len(checkpoint) > 0
        and all(torch.is_tensor(value) for value in checkpoint.values())
    )

    if not is_weight_only:
        return system.load_from_checkpoint(
            cfg.infer.ckpt_path,
            cfg=cfg,
            map_location=device,
            strict=False,
        )
    my_system = system(cfg=cfg)
    net_state_dict = {k.replace("net.", "", 1): v for k, v in checkpoint.items()}
    my_system.net.load_state_dict(net_state_dict, strict=False)
    return my_system.to(device)


def save_metrics(result_path, names, psnr3ds, ssim3ds, real_3d_ssims, psnrs, ssims):
    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    with open(result_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "psnr3d", "ssim3d", "real_3d_ssim", "psnr2d", "ssim2d"])

        for n, p3, s3, r3s, p2, s2 in zip(names, psnr3ds, ssim3ds, real_3d_ssims, psnrs, ssims):
            writer.writerow([n, p3, s3, r3s, p2, s2])

    print(f"Saved CSV metrics -> {result_path}")

    metric_summary = {
        "num_samples": len(names),
        "average_psnr3d": float(np.mean(psnr3ds)) if psnr3ds else 0.0,
        "average_ssim3d": float(np.mean(ssim3ds)) if ssim3ds else 0.0,
        "average_real_3d_ssim": float(np.mean(real_3d_ssims)) if real_3d_ssims else 0.0,
        "average_psnr2d": float(np.mean(psnrs)) if psnrs else 0.0,
        "average_ssim2d": float(np.mean(ssims)) if ssims else 0.0,
    }
    summary_path = os.path.splitext(result_path)[0] + "_summary.json"
    with open(summary_path, "w") as f:
        json.dump(metric_summary, f, indent=2)

    print(f"Saved metric summary -> {summary_path}")


@torch.no_grad()
def main(cfg):

    torch.set_float32_matmul_precision('medium')

    dataset = dataset_dict[cfg.infer.dataset.dataset_name]
    loader = DataLoader(dataset(cfg.infer.dataset), 
                              batch_size=cfg.infer.dataset.batch_size,
                              num_workers=cfg.infer.dataset.num_workers, 
                              shuffle=False,
                              pin_memory=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    my_system = load_system_for_inference(cfg, device)
    my_system.net.eval()
    names = []
    psnrs,ssims = [], []
    psnr3ds, ssim3ds = [], []
    real_3d_ssims = []

    os.makedirs(cfg.infer.save_folder, exist_ok=True)
    os.makedirs(os.path.join(cfg.infer.save_folder, 'slices'), exist_ok=True)

    for step, sample in enumerate(tqdm(loader, desc="Infer")):
        sample = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in sample.items()}

        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
        with autocast_ctx:
            output = my_system.net(sample, with_fine=cfg.with_fine)

        name = sample['meta']['scene'][0]
        img_pred = output['image'][0] 
        img_gt = sample['tar_xray'][0].permute(1,0,2,3).reshape(img_pred.shape)
        vol_gt = sample['volume']

        if "vol_refine" in output:
            pred_vol = torch.clip(output['vol_refine'], min=0.0, max=1.0)
        else:
            pred_vol = torch.clip(output["vol"], min=0.0, max=1.0)

        psnr_3d, _ = metric_vol(vol_gt, pred_vol, "psnr")
        ssim_3d, ssim_3d_axis = metric_vol(vol_gt, pred_vol, "ssim")
        real_3d_ssim_val = real_3d_ssim(vol_gt, pred_vol, data_range=1.0)
        psnr3ds.append(psnr_3d)
        ssim3ds.append(ssim_3d)
        real_3d_ssims.append(real_3d_ssim_val)
        
        if cfg.save_volume:
            save_dir = os.path.join(cfg.infer.save_folder, "volumes")
            os.makedirs(save_dir, exist_ok=True)
            save_volume(pred_vol, save_dir, name)

        if cfg.save_slice:
            vol_gt = vol_gt[0].cpu().numpy()  
            pred_vol = pred_vol[0].cpu().numpy()    
            save_slice_image(vol_gt, pred_vol, cfg.infer.save_folder, name)

        img_pred = normalize_tensor(img_pred)
        img_gt = normalize_tensor(img_gt)

        if cfg.save_proj:
            os.makedirs(os.path.join(cfg.infer.save_folder, "projs"), exist_ok=True)
            proj = torch.cat((img_gt, img_pred), dim=0).detach().cpu().numpy()[..., ::-1]
            cv2.imwrite(os.path.join(cfg.infer.save_folder, 'projs', name + '.jpg'), normalize_uint8(proj))

        if cfg.infer.eval_novel_view_only:
            width = 256
            img_pred = img_pred.permute(2,0,1)[None][...,width*cfg.n_views:]
            img_gt = img_gt.permute(2,0,1)[None][...,width*cfg.n_views:]
        else:
            img_pred = img_pred.permute(2,0,1)[None]
            img_gt = img_gt.permute(2,0,1)[None]
        
        if img_pred.shape[-1] > 0:
            color_loss_all = (img_pred-img_gt)**2
            psnr = -10. * torch.log(color_loss_all.mean()) / torch.log(torch.tensor([10.]).to(device))
            ssim_val = ssim(img_pred, img_gt, data_range=1.0, size_average=False).detach()
            psnrs.append(psnr.item())
            ssims.append(ssim_val.item())
            print(
                f"{name}: psnr3d={psnr_3d:.4f}, ssim3d={ssim_3d:.4f}, "
                f"real_3d_ssim={real_3d_ssim_val:.4f}, psnr2d={psnr.item():.4f}, "
                f"ssim2d={ssim_val.item():.4f}"
            )
            
        names.append(name)

        del output
        if device == "cuda":
            torch.cuda.empty_cache()

    if len(psnrs) and cfg.infer.metric_path is not None:
        save_metrics(cfg.infer.metric_path, names, psnr3ds, ssim3ds, real_3d_ssims, psnrs, ssims)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("base_config", type=str)
    parser.add_argument("infer_config", type=str)
    args, unknown = parser.parse_known_args()

    base_conf = OmegaConf.load(args.base_config)
    infer_conf = OmegaConf.load(args.infer_config)
    cli_conf = OmegaConf.from_dotlist(unknown)

    cfg = OmegaConf.merge(base_conf, infer_conf, cli_conf)
    main(cfg)

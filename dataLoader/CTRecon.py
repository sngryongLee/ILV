import os
import json
import torch
import numpy as np
from pathlib import Path
from dataLoader.utils import (
    build_rays,
    even_sample,
    load_projector_cfg,
    load_volume,
    load_projections,
    normalize_projection,
    compute_fov
)


def _resolve_path(path):
    path = Path(path)
    if path.is_absolute() or path.exists():
        return str(path)
    repo_root = Path(__file__).resolve().parents[1]
    return str(repo_root / path)


class CTRecon(torch.utils.data.Dataset):
    def __init__(self, cfg):
        super().__init__()

        self.cfg       = cfg
        self.data_root = _resolve_path(cfg.data_root)
        self.split     = cfg.split           
        self.img_size  = tuple(cfg.img_size) 
        self.n_group   = cfg.n_group
        self.n_scenes  = getattr(cfg, 'n_scenes', None)

        self.proj_cfg = load_projector_cfg(_resolve_path(cfg.projector_cfg_path))

        with open(_resolve_path(cfg.split_json), 'r') as f:
            splits = json.load(f)

        scenes = splits[self.split]
        if self.n_scenes:
            scenes = scenes[: self.n_scenes]
        self.scenes_name = scenes

        self.novel_views = 6 if self.split == "train" else 50 - self.n_group
        print(f"[CT RECON] split={self.split}, scenes={len(scenes)}")


    def __len__(self):
        return len(self.scenes_name)

    def __getitem__(self, idx):
        scene = self.scenes_name[idx]
        scene_dir = os.path.join(self.data_root, scene['path'])

        camera_path = Path(__file__).with_name("camera_params.npz")
        cams = np.load(camera_path)
        K = cams['K']
        if K.ndim == 3:
            K = K[0]
        c2ws = cams['c2ws']  # (N,4,4)
        all_view = c2ws.shape[0]

        volume = load_volume(os.path.join(scene_dir, 'volume.nii.gz'))
        projs = load_projections(os.path.join(scene_dir, 'projections.npy'))  # (50, H, W)

        src_ids = even_sample(all_view, self.n_group)

        all_ids = set(range(all_view))
        remaining_ids = list(all_ids - set(src_ids))
        tar_ids = sorted(np.random.choice(remaining_ids, self.novel_views, replace=False).tolist())
        view_ids = src_ids + tar_ids


        H, W = self.img_size
        imgs, ixts, exts, w2cs, bg = [], [], [], [], []

        for vid in view_ids:
            imgs.append(normalize_projection(projs[vid]))
            ixts.append(K.astype(np.float32))
            c2w = c2ws[vid].astype(np.float32)
            exts.append(c2w)
            w2cs.append(np.linalg.inv(c2w))
            bg.append(np.ones(3, dtype=np.float32))


        imgs = np.stack(imgs, axis=0)      # (V,H,W,3)
        ixts = np.stack(ixts, axis=0)      # (V,3,3)
        c2ws_sel = np.stack(exts, axis=0)  # (V,4,4)
        w2cs_sel = np.stack(w2cs, axis=0)  # (V,4,4)
        bg_cols = np.stack(bg, axis=0)     # (V,3)
        rays = build_rays(c2ws_sel, ixts.copy(), H, W, 1.0)
        rays_down = build_rays(c2ws_sel, ixts.copy(), H, W, 1.0/16)
        FovX, FovY = compute_fov(self.proj_cfg)
        
        return {
            'fovx': FovX,
            'fovy': FovY,
            'volume' : volume,
            'tar_xray' : imgs,
            'tar_ixt' : ixts,
            'tar_c2w' : c2ws_sel,
            'tar_w2c' : w2cs_sel,
            'tar_rays': rays,
            'tar_rays_down': rays_down,
            'bg_color': bg_cols,
            'meta'    : {'scene': scene['name'], 'src_ids': src_ids, 'tar_ids': tar_ids, 
                         'nVoxel':self.proj_cfg['nVoxel'], 'sVoxel':self.proj_cfg['sVoxel'],
                         "scene_dir": scene_dir}
        }

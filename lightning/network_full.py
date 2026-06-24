from lightning.utils import MiniCam
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as L
from einops import rearrange

from lightning.renderer_xray import Renderer

from .modules.attention import SelfAttnBlock
from .modules.encoder import DinoWrapper
from .modules.refinement import Refine3D, Refine3DLegacy
from .modules.decoder import Decoder
from .modules.xray_volume_transformer import VolTransformer




def projection(grid, w2cs, ixts):

    points = grid.reshape(1,-1,3) @ w2cs[:,:3,:3].permute(0,2,1) + w2cs[:,:3,3][:,None]
    points = points @ ixts.permute(0,2,1)
    points_xy = points[...,:2]/points[...,-1:]
    return points_xy, points[...,-1:]


class ModLN(L.LightningModule):
    """
    Modulation with adaLN.
    
    References:
    DiT: https://github.com/facebookresearch/DiT/blob/main/models.py#L101
    """
    def __init__(self, inner_dim: int, mod_dim: int, eps: float):
        super().__init__()
        self.norm = nn.LayerNorm(inner_dim, eps=eps)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(mod_dim, inner_dim * 2),
        )

    @staticmethod
    def modulate(x, shift, scale):
        # x: [N, L, D]
        # shift, scale: [N, D]
        return x * (1 + scale) + shift

    def forward(self, x, cond):
        shift, scale = self.mlp(cond).chunk(2, dim=-1)  # [N, D]
        return self.modulate(self.norm(x), shift, scale)  # [N, L, D]
    

class Network(L.LightningModule):
    def __init__(self, cfg, white_bkgd=True):
        super(Network, self).__init__()

        self.cfg = cfg
        self.scene_size = 0.5
        self.white_bkgd = white_bkgd

        self.lowlevel_proj = nn.Linear(
            1 * self.cfg.model.patch_size ** 2,
            self.cfg.model.dim,
            bias=False,
        )
        self.lowlevel_proj.apply(self._init_weights)
        self.lowlevel_norm = nn.LayerNorm(self.cfg.model.dim, bias=False)
       
        self.img_encoder = DinoWrapper(
            model_name=cfg.model.encoder_backbone,
            is_train=True,
        )
       
        encoder_feat_dim = self.img_encoder.model.num_features
        self.dir_norm = ModLN(encoder_feat_dim + self.cfg.model.dim, 6, eps=1e-6)

        # 3D volume grid for embedding
        self.grid_reso = cfg.model.vol_embedding_reso
        self.register_buffer("dense_grid", self.build_dense_grid(self.grid_reso))
        self.register_buffer("centers", self.build_dense_grid(self.grid_reso * 2))

        # view embedding
        if cfg.model.view_embed_dim > 0:
            self.view_embed = nn.Parameter(torch.randn(1, self.cfg.n_views, cfg.model.view_embed_dim, 1, 1, 1) * (1. / cfg.model.view_embed_dim) ** 0.5)
        
 
        self.n_groups = cfg.model.n_groups
        vol_embedding_dim = cfg.model.embedding_dim
        self.xvt = VolTransformer(
            embed_dim=vol_embedding_dim,
            image_feat_dim=encoder_feat_dim
            + cfg.model.view_embed_dim
            + self.cfg.model.dim,
            vol_low_res=self.grid_reso,
            vol_high_res=self.grid_reso * 2,
            out_dim=cfg.model.vol_embedding_out_dim,
            n_groups=self.n_groups,
            num_layers=cfg.model.num_layers,
            num_heads=cfg.model.num_heads,
        )

        self.feat_vol_reso = cfg.model.vol_feat_reso
        self.register_buffer("volume_grid", self.build_dense_grid(self.feat_vol_reso))

        self.n_offset_groups = cfg.model.n_offset_groups
        self.register_buffer("group_centers", self.build_dense_grid(64*2))
        self.group_centers = self.group_centers.reshape(1,-1,3)

        # R2GS model
        self.scaling_dim, self.rotation_dim = 3, 4
        self.opacity_dim = 1
        self.K = cfg.model.K

        vol_embedding_out_dim = cfg.model.vol_embedding_out_dim
        self.decoder = Decoder(vol_embedding_out_dim, self.scaling_dim, self.rotation_dim, self.opacity_dim, self.K)
        self.gs_render = Renderer(white_background=white_bkgd, radius=1)

        # parameters initialization
        self.opacity_shift = cfg.model.opacity_shift
        self.voxel_size = 2.0 / (self.grid_reso * 2)
        self.scaling_shift = np.log(
            0.5 * self.voxel_size / cfg.model.scale_shift
        )

        # Keep the module name as vol_refine so checkpoint keys still match.
        use_legacy_refine = bool(
            getattr(cfg, "n_views", None) == 8 and hasattr(cfg, "infer")
        )
        self.vol_refine = Refine3DLegacy() if use_legacy_refine else Refine3D()

    def build_dense_grid(self, reso: int):
        array = torch.arange(reso, device=self.device)
        grid = torch.stack(
            torch.meshgrid(array, array, array, indexing="ij"),
            dim=-1,
        )
        grid = (grid + 0.5) / reso * 2 - 1
        return grid.reshape(reso, reso, reso, 3) * self.scene_size
    
    def build_feat_vol(self, src_inps, img_feats, n_views_sel, batch):
  
        h, w = src_inps.shape[-2:]
        src_ixts = batch["tar_ixt"][:, :n_views_sel].reshape(-1, 3, 3)
        src_w2cs = batch["tar_w2c"][:, :n_views_sel].reshape(-1, 4, 4)

        img_wh = torch.tensor([w, h], device=self.device)
        point_img, _ = projection(self.volume_grid, src_w2cs, src_ixts)
        point_img = (point_img + 0.5) / img_wh * 2 - 1.0

        # viewing direction → plücker coords
        rays = batch["tar_rays_down"][:, :n_views_sel]
        feats_dir = self.ray_to_plucker(rays).reshape(-1, *rays.shape[2:])

        # channel-last → modulation → channel-first
        img_feats = torch.einsum("bchw->bhwc", img_feats)
        img_feats = self.dir_norm(img_feats, feats_dir)
        img_feats = torch.einsum("bhwc->bchw", img_feats)

        n_channel = img_feats.shape[1]
        feats_vol = F.grid_sample(
            img_feats.float(),
            point_img.unsqueeze(1),
            align_corners=False,
        ).to(img_feats)

        feats_vol = feats_vol.view(
            -1,
            n_views_sel,
            n_channel,
            self.feat_vol_reso,
            self.feat_vol_reso,
            self.feat_vol_reso,
        )

        return feats_vol

    def ray_to_plucker(self, rays):
        origin, direction = rays[..., :3], rays[..., 3:6]
        direction = F.normalize(direction, p=2.0, dim=-1)
        moment = torch.cross(origin, direction, dim=-1)
        return torch.cat((direction, moment), dim=-1)

    def get_offseted_pt(self, offset, K):
        B = offset.shape[0]
        half_cell_size = 0.5 * self.scene_size / self.n_offset_groups
        centers = (
            self.group_centers.unsqueeze(-2)
            .expand(B, -1, K, -1)
            .reshape(offset.shape)
            + offset * half_cell_size
        )
        return centers

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward(self, batch, with_fine=False):
        
        B, N, H, W, C = batch["tar_xray"].shape
        n_views_sel = self.cfg.n_views
        render_views = getattr(self.cfg, "render_views", True)

        src = batch["tar_xray"][:, :n_views_sel].reshape(
            B * n_views_sel, H, W, C
        )
        src = torch.einsum("bhwc->bchw", src)  # [B*V, C, H, W]

        img_feats = torch.einsum("blc->bcl", self.img_encoder(src))
        token_size = int(np.sqrt(H * W / img_feats.shape[-1]))
        img_feats = img_feats.reshape( *img_feats.shape[:2], H // token_size, W // token_size)

        lowlevel_patches = (src.view(B, n_views_sel, C, -1).permute(0, 1, 3, 2).contiguous() * 2 - 1)
        hh, ww = (H // self.cfg.model.patch_size, W // self.cfg.model.patch_size)
        lowlevel_patches = rearrange(lowlevel_patches, 
                                "b s (hh ph ww pw) d -> b (s hh ww) (ph pw d)", 
                                hh=hh, ww=ww, 
                                ph=self.cfg.model.patch_size, pw=self.cfg.model.patch_size)
        
        lowlevel_patches = self.lowlevel_norm(
            self.lowlevel_proj(lowlevel_patches)
        )

        low_img_feats = rearrange(lowlevel_patches, 
                                "b (s hh ww) d -> (b s) d hh ww", 
                                hh=hh, ww=ww)

        feat_vol = self.build_feat_vol(src, torch.cat([img_feats, low_img_feats], dim=1), n_views_sel, batch) # B n_views_sel C D H W

        if self.cfg.model.view_embed_dim > 0:
            feat_vol = torch.cat((feat_vol, self.view_embed[:,:n_views_sel].expand(B,-1,-1,self.feat_vol_reso, self.feat_vol_reso, self.feat_vol_reso)), dim = 2)

        latent_volume = self.xvt(feat_vol)
        _offset_coarse, _scaling_coarse, _rotation_coarse, _opacity_coarse = self.decoder(latent_volume, self.opacity_shift, self.scaling_shift)

        _centers_coarse = self.get_offseted_pt(_offset_coarse, self.K)
        render_img_scale = batch.get('render_img_scale', 1.0)
        
        nVoxel, sVoxel = batch['meta']['nVoxel'][0], batch['meta']['sVoxel'][0]
        outputs, vol_preds = [], []

        for i in range(B):
            
            fovx,fovy = batch['fovx'][i], batch['fovy'][i]
            height, width = int(H*render_img_scale), int(W*render_img_scale)
            _centers = _centers_coarse[i]

            if render_views:
                outputs_view = []
                tar_c2ws = batch['tar_c2w'][i]

                for j, c2w in enumerate(tar_c2ws):
                    bg_color = batch['bg_color'][i,j]
                    self.gs_render.set_bg_color(bg_color)
                
                    cam = MiniCam(c2w, width, height, fovy, fovx, self.device)
                    frame = self.gs_render.render_img(cam, _centers, _opacity_coarse[i], _scaling_coarse[i], _rotation_coarse[i], self.device)
                    frame['image'] = frame['image'] / (frame['image'].max()+ 1e-8)
                    outputs_view.append(frame)
        
                outputs.append({k: torch.cat([d[k] for d in outputs_view], dim=1) for k in outputs_view[0]})

            vol_pred = self.gs_render.query_voxel(nVoxel, sVoxel, _centers, _opacity_coarse[i], _scaling_coarse[i], _rotation_coarse[i], self.device)['vol']
            vol_preds.append(vol_pred)


        outputs = {k: torch.stack([d[k] for d in outputs]) for k in outputs[0]} if render_views else {}
        outputs["vol"] = torch.stack(vol_preds) 
        
        if with_fine:
            vol_input = outputs["vol"].unsqueeze(1)              # [B, 1, D, H, W]
            vol_refine_pred = self.vol_refine(vol_input)         # [B, 1, D, H, W]
            outputs["vol_refine"] = vol_refine_pred.squeeze(1)   # [B, D, H, W]

        return outputs

import math

import torch
from torch import nn

from xray_gaussian_rasterization_voxelization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
    GaussianVoxelizationSettings,
    GaussianVoxelizer,
)


def inverse_softplus(x, beta=1):
    return torch.log(torch.exp(beta * x) - 1) / beta


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=L.dtype, device=L.device)

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]

    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])
    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), dtype=q.dtype, device=q.device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)

    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=s.dtype, device=s.device)
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def covariance_from_scaling_rotation(scaling, rotation, c2ws):
    L = build_scaling_rotation(scaling, rotation)
    actual_cov = L @ L.transpose(1, 2)
    symm = strip_symmetric(actual_cov)
    return symm


class Renderer(nn.Module):
    def __init__(self, white_background=False, radius=1):
        super(Renderer, self).__init__()
        
        self.white_background = white_background
        self.radius = radius

        self.setup_functions()
        
        self.bg_color = torch.tensor(
            [1, 1, 1] if self.white_background else [0, 0, 0],
            dtype=torch.float32,
        )

    def setup_functions(self):
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.density_activation = torch.nn.Softplus() 
        self.density_inverse_activation = inverse_softplus
        self.rotation_activation = torch.nn.functional.normalize

    def set_bg_color(self, bg):
        self.bg_color = bg
        
    def set_rasterizer(self, viewpoint_camera, scaling_modifier=1.0, device="cuda"):
        # Set up rasterization configuration

        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            mode=1, # cone beam
            debug=False,
        )
        return GaussianRasterizer(raster_settings=raster_settings)
    
    def get_scaling(self, _scaling):
        return self.scaling_activation(_scaling)
    
    def get_rotation(self, _rotation):
        return self.rotation_activation(_rotation)
    
    def get_opacity(self, _density):
        return self.density_activation(_density)

    def get_covariance(self, _scaling, _rotation, c2ws):
        return covariance_from_scaling_rotation(self.get_scaling(_scaling), self.get_rotation(_rotation), c2ws)

    def query_voxel(self,
                nVoxel,
                sVoxel,
                centers,
                opacity,
                scales,
                rotations,
                device,
                scaling_modifier=1.0,
    ):
        """
        Query a volume with voxelization.
        """
        voxel_settings = GaussianVoxelizationSettings(
            scale_modifier=scaling_modifier,
            nVoxel_x=int(nVoxel[0]),
            nVoxel_y=int(nVoxel[0]),
            nVoxel_z=int(nVoxel[0]),
            sVoxel_x=float(sVoxel[0]),
            sVoxel_y=float(sVoxel[0]),
            sVoxel_z=float(sVoxel[0]),
            center_x=float(0),
            center_y=float(0),
            center_z=float(0),
            prefiltered=False,
            debug=False,
        )
        voxelizer = GaussianVoxelizer(voxel_settings=voxel_settings)
        if scales is not None:
            scales = self.get_scaling(scales) #+ 0.0003
        if rotations is not None:
            rotations = self.get_rotation(rotations)

        means3D = centers
        density = self.get_opacity(opacity)
        cov3D_precomp = None

        vol_pred, _ = voxelizer(
            means3D=means3D,
            opacities=density,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

        return {"vol": vol_pred,}

    def render_img(
                self,
                cam,
                centers,
                density,
                scales,
                rotations,
                device,
                cov3D_precomp=None,
                ):

        """
        Render an X-ray projection with rasterization.
        """
        scaling_modifier = 1.0

        rasterizer = self.set_rasterizer(cam, device=device)
         # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        density = self.get_opacity(density)

        if scales is not None:
            scales = self.get_scaling(scales) #+ 0.0003
        if rotations is not None:
            rotations = self.get_rotation(rotations)
        
        centers = centers
        screenspace_points = (
            torch.zeros_like(
                centers,
                dtype=centers.dtype,
                requires_grad=True,
                device=device,
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        means2D = screenspace_points
        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        # scales = None
        # rotations = None
        cov3D_precomp = None
        # if pipe.compute_cov3D_python:
        #     cov3D_precomp = pc.get_covariance(scaling_modifier)
        # else:
        #     scales = pc.get_scaling
        #     rotations = pc.get_rotation
        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, _ = rasterizer(
            means3D=centers,
            means2D=means2D,
            opacities=density,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )
        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {"image": rendered_image.permute(1,2,0),}

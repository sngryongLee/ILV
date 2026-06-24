import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .group_attention import GroupAttBlock


class VolTransformer(nn.Module):
    """
    Transformer for constructing 3D latent volume from multi-view image features.
    """

    def __init__(
        self,
        embed_dim: int,
        image_feat_dim: int,
        n_groups: list,
        vol_low_res: int,
        vol_high_res: int,
        out_dim: int,
        num_layers: int,
        num_heads: int,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.vol_low_res = vol_low_res
        self.vol_high_res = vol_high_res
        self.out_dim = out_dim
        self.n_groups = n_groups
        self.block_size = [vol_low_res // g for g in n_groups]
        self.embed_dim = embed_dim


        self.pos_embed = nn.Parameter(
            torch.randn(
                1, embed_dim,
                vol_low_res,
                vol_low_res,
                vol_low_res
            ) * (1.0 / embed_dim) ** 0.5
        )

        self.layers = nn.ModuleList([
            GroupAttBlock(
                inner_dim=embed_dim,
                cond_dim=image_feat_dim,
                num_heads=num_heads,
                eps=eps,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=eps)

        # upsample 3D
        self.deconv = nn.ConvTranspose3d(
            embed_dim, out_dim,
            kernel_size=4,
            stride=4
        )
        
        # projection for max view and mean view
        self.maxview_proj = nn.Conv3d(image_feat_dim, embed_dim // 4, 1, bias=False)
        self.meanview_proj = nn.Conv3d(image_feat_dim, embed_dim // 4, 1, bias=False)

    def forward(self, image_feats):
        """
        image_feats: [B, V, C, D, H, W]
        """

        B, V, C, D, H, W = image_feats.shape
        # max/mean over views
        max_view = self.maxview_proj(image_feats[:, :V].max(dim=1).values)
        mean_view = self.meanview_proj(image_feats[:, :V].mean(dim=1))

        # resize to match latent size
        max_view = F.interpolate(max_view, size=(32, 32, 32), mode='trilinear', align_corners=False)
        mean_view = F.interpolate(mean_view, size=(32, 32, 32), mode='trilinear', align_corners=False)

        # build grouped feature tokens
        group_tokens = []
        for n_group in self.n_groups:
            block_size = D // n_group
            blocks = (
                image_feats.unfold(3, block_size, block_size)
                           .unfold(4, block_size, block_size)
                           .unfold(5, block_size, block_size)
            )
            blocks = blocks.contiguous().view(
                B, V, C, n_group ** 3, block_size ** 3
            )
            blocks = torch.einsum('bvcgl->bgvlc',blocks).reshape(B*n_group**3,block_size**3*V,C)
            group_tokens.append(blocks)

        # base volume with positional embeddings
        x = self.pos_embed.repeat(B, 1, 1, 1, 1)

        # pass through layers
        for i, layer in enumerate(self.layers):
            idx = i % len(self.block_size)
            x = layer(
                x,
                group_tokens[idx],
                self.n_groups[idx],
                self.block_size[idx],
                max_view=torch.cat([max_view, mean_view], dim=1)
            )
            # x = layer(
            #     x,
            #     group_tokens[idx],
            #     self.n_groups[idx],
            #     self.block_size[idx],
            #     max_view=None
            # )
        x = self.norm(rearrange(x, "b c d h w -> b d h w c"))
        x = rearrange(x, "b d h w c -> b c d h w")

        # upsample
        x_up = self.deconv(x)
        x_up = torch.einsum('bcdhw->bdhwc',x_up).contiguous()
        return x_up

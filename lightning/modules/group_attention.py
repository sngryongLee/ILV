import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .attention import SelfAttnBlock  


class GroupAttBlock(nn.Module):

    def __init__(
        self,
        inner_dim: int,
        cond_dim: int,
        num_heads: int,
        eps: float = 1e-6,
        attn_drop: float = 0.,
        attn_bias: bool = False,
        mlp_ratio: float = 2.,
        mlp_drop: float = 0.,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.norm1 = norm_layer(inner_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=inner_dim,
            num_heads=num_heads,
            kdim=cond_dim,
            vdim=cond_dim,
            dropout=attn_drop,
            bias=attn_bias,
            batch_first=True,
        )

        self.cnn = nn.Conv3d(
            inner_dim + inner_dim // 2,
            inner_dim,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.self_attn = SelfAttnBlock(
            dim=inner_dim,
            head_dim=32,
            sr_ratio=2,
        )

        with torch.no_grad():
            self.cnn.weight.zero_()
            if self.cnn.bias is not None:
                self.cnn.bias.zero_()

    def forward(self, x, cond, group_axis, block_size, max_view=None):
        """
        x: [B, C, D, H, W]
        cond: condition tokens
        max_view: [B, embed_dim/2, 32, 32, 32]
        """

        B, C, D, H, W = x.shape
        G = group_axis ** 3
        L = block_size ** 3

        # chunk volume into patches
        patches = (
            x.unfold(2, block_size, block_size)
             .unfold(3, block_size, block_size)
             .unfold(4, block_size, block_size)
        )  

        patches = patches.reshape(B, C, -1, L)      
        patches = rearrange(patches, "b c g l -> b g l c")
        patches = patches.reshape(B * G, L, C)

        # Cross-attention
        patches = patches + self.cross_attn(
            self.norm1(patches),
            cond,
            cond,
            need_weights=False
        )[0]

        # reshape for self-attention
        x_tokens = patches.view(B, G * L, C)
        x_tokens = self.self_attn(x_tokens)
        patches = x_tokens.view(B * G, L, C)

        # reshape back to volume
        patches = patches.view(
            B, group_axis, group_axis, group_axis,
            block_size, block_size, block_size, C
        )

        patches = torch.einsum('bdhwzyxc->bcdzhywx', patches).reshape(x.shape)

        # CNN + view conditioning
        if max_view is None:
            max_view = torch.zeros(
                B,
                C // 2,
                D,
                H,
                W,
                dtype=patches.dtype,
                device=patches.device,
            )
        out = patches + self.cnn(torch.cat([patches, max_view], dim=1))
        return out

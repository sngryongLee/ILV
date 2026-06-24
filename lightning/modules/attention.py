import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    import xformers.ops as xops
except ImportError:
    xops = None


def cube_root(n: int) -> int:
    return round(n ** (1 / 3))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._norm(x.float()).type_as(x)
        return out * self.weight.type_as(x)


class MLP(nn.Module):
    def __init__(
            self,
            in_features: int,
            mlp_ratio: float = 4.0,
            mlp_bias: bool = False,
            out_features = None,
            act_layer=nn.GELU,
            norm_layer=None,
        ):
            super().__init__()
            self.use_norm = norm_layer is not None
            if self.use_norm:
                self.norm = norm_layer(in_features, bias=False)

            out_features = out_features or in_features
            hidden_features = int(in_features * mlp_ratio)

            self.fc1 = nn.Linear(in_features, hidden_features, bias=mlp_bias)
            self.act = act_layer()
            self.fc2 = nn.Linear(hidden_features, out_features, bias=mlp_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_norm:
            x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class SelfAttention(nn.Module):
    """
    3D-aware self-attention with optional spatial reduction (sr_ratio).

    Input:  (B, N, C)
    Output: (B, N, C)
    """
    def __init__(
        self,
        embed_dim: int = 256,
        head_dim: int = 32,
        sr_ratio: int = 2,
        qkv_bias: bool = False,
        qk_scale = None,
        qk_norm: bool = True,
        norm_layer=None,
    ):
        super().__init__()

        assert embed_dim % head_dim == 0, "embed_dim must be divisible by head_dim"
        self.num_heads = embed_dim // head_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.use_norm = norm_layer is not None
        if self.use_norm:
            self.norm = norm_layer(embed_dim, bias=False)

        self.query = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.key_value = nn.Linear(embed_dim, 2 * embed_dim, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            # 3D spatial reduction
            self.sr = nn.Conv3d(embed_dim, embed_dim,
                                kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, C)
        """
        if self.use_norm:
            x = self.norm(x)

        B, N, C = x.shape

        # (B, N, H, Dh)
        q = self.query(x).reshape(B, N, self.num_heads, C // self.num_heads)

        # spatial reduction on keys/values
        if self.sr_ratio > 1:
            n = cube_root(N)
            x_ = x.permute(0, 2, 1).reshape(B, C, n, n, n)          # (B, C, D, H, W)
            x_ = self.sr(x_)                                        # (B, C, D', H', W')
            x_ = x_.reshape(B, C, -1).permute(0, 2, 1).contiguous() # (B, Nsr, C)
            x_ = self.sr_norm(x_)
            kv = self.key_value(x_)                                 # (B, Nsr, 2C)
        else:
            kv = self.key_value(x)                                  # (B, N, 2C)

        kv = kv.reshape(B, -1, 2, self.num_heads, C // self.num_heads)
        kv = kv.permute(2, 0, 1, 3, 4)  # (2, B, Nk, H, Dh)
        k, v = kv[0], kv[1]

        q, k = self.q_norm(q), self.k_norm(k)

        if xops is not None:
            out = xops.memory_efficient_attention(
                q.contiguous(), k.contiguous(), v.contiguous(),
                op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp),
            )
        else:
            q_t = q.permute(0, 2, 1, 3)  # (B, H, Nq, Dh)
            k_t = k.permute(0, 2, 1, 3)  # (B, H, Nk, Dh)
            v_t = v.permute(0, 2, 1, 3)  # (B, H, Nk, Dh)
            try:
                out = F.scaled_dot_product_attention(q_t, k_t, v_t)  # (B, H, Nq, Dh)
                out = out.permute(0, 2, 1, 3)                        # (B, Nq, H, Dh)
            except TypeError:
                attn = (q @ k.transpose(-2, -1)) * (
                    1.0 / math.sqrt(C // self.num_heads)
                )  # (B, Nq, Nk)
                attn = attn.softmax(dim=-1)
                out = attn @ v  # (B, Nq, H, Dh)

        out = rearrange(out, "b n h d -> b n (h d)")  # (B, N, C)
        out = self.proj(out)
        return out


class SelfAttnBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        head_dim: int,
        sr_ratio: int = 1,
        mlp_ratio: float = 4.0,
        mlp_bias: bool = False,
        qkv_bias: bool = False,
        qk_scale = None,
        qk_norm: bool = True,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.attn = SelfAttention(
            embed_dim=dim,
            head_dim=head_dim,
            sr_ratio=sr_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            qk_norm=qk_norm,
            norm_layer=norm_layer,
        )
        self.mlp = MLP(
            in_features=dim,
            mlp_ratio=mlp_ratio,
            mlp_bias=mlp_bias,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x
import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, in_dim, scaling_dim, rotation_dim, opacity_dim, K=1, latent_dim=256):
        super().__init__()
        self.K = K
        self.opacity_dim = opacity_dim
        self.scaling_dim = scaling_dim
        self.rotation_dim = rotation_dim
        self.out_dim = 3 + opacity_dim + scaling_dim + rotation_dim

        num_layers = 2
        layers = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True)
        ]

        for _ in range(num_layers - 1):
            layers += [
                nn.Linear(in_dim, in_dim),
                nn.ReLU(inplace=True)
            ]

        layers.append(nn.Linear(in_dim, self.out_dim * K))

        self.mlp_coarse = nn.Sequential(*layers)
        self.init(self.mlp_coarse)

    def init(self, layers):

        init_method = "xavier"
        if init_method:
            for layer in layers:
                if not isinstance(layer, torch.nn.Linear):
                    continue 
                if init_method == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(layer.weight.data)
                elif init_method == "xavier":
                    torch.nn.init.xavier_uniform_(layer.weight.data)
                torch.nn.init.zeros_(layer.bias.data)

    
    def forward(self, feats, opacity_shift, scaling_shift):
        parameters = self.mlp_coarse(feats).float()
        parameters = parameters.view(*parameters.shape[:-1],self.K,-1)
        offset, opacity, scaling, rotation = torch.split(
            parameters, 
            [3, self.opacity_dim, self.scaling_dim, self.rotation_dim],
            dim=-1
            )
        opacity = opacity + opacity_shift 
        scaling = scaling + scaling_shift 
        offset = torch.sigmoid(offset)*2-1.0

        B = opacity.shape[0]
        opacity = opacity.view(B,-1,self.opacity_dim)
        scaling = scaling.view(B,-1,self.scaling_dim)
        rotation = rotation.view(B,-1,self.rotation_dim)
        offset = offset.view(B,-1,3)
        
        return offset, scaling, rotation, opacity


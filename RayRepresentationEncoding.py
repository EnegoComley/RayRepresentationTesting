import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import lightning as L
import torch

class RayRepresentationEncoder(nn.Module):
    def __init__(self, transformer_layers=1, representation_size=128, initial_dropout=0.3, n_rays=5000):
        super().__init__()
        self.ray_encoder = nn.Sequential(nn.Dropout(initial_dropout),
                                      nn.Linear(250 * 1, 256),
                                      nn.BatchNorm1d(256),
                                      nn.ReLU(),
                                      nn.Linear(256, representation_size),
                                      nn.BatchNorm1d(representation_size),
                                      nn.ReLU())

        #self.ray_encoder = ConvEncoder()



        transformer_layer = nn.TransformerEncoderLayer(d_model=representation_size, nhead=8, dropout=0.3, batch_first=True)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=transformer_layers)

        self.pos_encoding = nn.Parameter(torch.randn(n_rays, representation_size), requires_grad=True)

    def forward(self, ray_colours):

        batch_size = ray_colours.shape[0]
        n_rays = ray_colours.shape[1]
        ray_colours = ray_colours.view(batch_size, n_rays, -1, 4)
        alphas = ray_colours[:, :, :, 3]
        #ray_colours[:, :, :, :3] = alphas[..., None] * ray_colours[:, :, :, :3]
        #ray_colours = ray_colours.view(batch_size, n_rays, -1)

        ray_colours = ((alphas > 0.001) * 1).float()
        del alphas

        x = ray_colours

        del ray_colours
        #ray_ids = torch.randperm(n_rays)[:1000]
        #x = x[:, ray_ids, :]



        x = x.view(batch_size * n_rays, -1)

        x = self.ray_encoder(x)
        x = x.view(batch_size, n_rays, -1)
        x = x + self.pos_encoding.unsqueeze(0).repeat(batch_size, 1, 1)
        x = self.transformer(x)

        del batch_size
        return x

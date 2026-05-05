from DatasetLoading import RepairDatasetLoader
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger
import os

import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

torch.set_float32_matmul_precision('medium')


class ResnetBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out += residual
        out = self.bn2(out)
        out = self.relu(out)
        return out

class GridReconstructionNetwork(nn.Module):
    def __init__(self, small_bottleneck=False):
        super().__init__()
        if not small_bottleneck:
            self.encoder = nn.Sequential(
                nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(),
                nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(),
                nn.MaxPool3d(2), # 200 -> 100
                nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(128),
                nn.ReLU(),
                nn.MaxPool3d(2), # 100 -> 50
                nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(128),
                nn.ReLU(),
                nn.MaxPool3d(2), # 50 -> 25
                nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(128),
                nn.ReLU(),
                nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(128),
                nn.ReLU(),
            )

            self.decoder = nn.Sequential(
                nn.ConvTranspose3d(128, 128, kernel_size=4, stride=2, padding=1), # 25 -> 50
                nn.BatchNorm3d(128),
                nn.ReLU(),
                nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1), # 50 -> 100
                nn.BatchNorm3d(64),
                nn.ReLU(),
                nn.ConvTranspose3d(64, 64, kernel_size=4, stride=2, padding=1), # 100 -> 200
                nn.BatchNorm3d(64),
                nn.ReLU(),
                nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1)
            )
        else:
            self.encoder = nn.Sequential(
                nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=13), # 200 -> 224
                nn.BatchNorm3d(64),
                nn.ReLU(),
                nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(),
                nn.MaxPool3d(2),  # 224 -> 112
                nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(128),
                nn.ReLU(),
                nn.MaxPool3d(2),  # 112 -> 56
                nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(128),
                nn.ReLU(),
                nn.MaxPool3d(2),  # 56 -> 28
                nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(128),
                nn.ReLU(),
                nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=2),
                nn.BatchNorm3d(128),
                nn.ReLU(),
                nn.MaxPool3d(2),  # 28 -> 14
                nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(128),
                nn.ReLU(),
                nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=2),
                nn.BatchNorm3d(128),
                nn.ReLU(),
                nn.MaxPool3d(2),  # 14 -> 7
                nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(128),
                nn.ReLU(),
                nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(128),
                nn.ReLU(),
            )

            self.decoder = nn.Sequential(
                nn.ConvTranspose3d(128, 128, kernel_size=4, stride=2, padding=1),  # 7 -> 14
                nn.BatchNorm3d(128),
                nn.ReLU(),
                nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(128),
                nn.ReLU(),
                nn.ConvTranspose3d(128, 128, kernel_size=4, stride=2, padding=1),  # 14 -> 28
                nn.BatchNorm3d(128),
                nn.ReLU(),
                nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(128),
                nn.ReLU(),
                nn.ConvTranspose3d(128, 128, kernel_size=4, stride=2, padding=1),  # 28 -> 56
                nn.BatchNorm3d(128),
                nn.ReLU(),
                nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(128),
                nn.ReLU(),
                nn.ConvTranspose3d(128, 128, kernel_size=4, stride=2, padding=1),  # 56 -> 112
                nn.BatchNorm3d(128),
                nn.ReLU(),
                nn.Conv3d(128, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(),
                nn.ConvTranspose3d(64, 64, kernel_size=4, stride=2, padding=13),  # 112 -> 200
                nn.BatchNorm3d(64),
                nn.ReLU(),
                nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(),
                nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1))



    def forward(self, representation):
        x = self.encoder(representation)
        x = self.decoder(x)
        return x


class GridReconstruction(L.LightningModule):
    def __init__(self, weight_opacity=False, small_bottleneck=False):
        super().__init__()
        self.model = GridReconstructionNetwork(small_bottleneck)
        self.lr = 5e-4
        self.weight_opacity = weight_opacity
        self.loss_func = nn.MSELoss()

    def calculate_loss(self, batch, stage):
        representation = batch
        reconstruction = self.model(representation)

        loss = self.loss_func(representation, reconstruction)# * weight.repeat(1, 3))

        self.log(stage + '_loss', loss)

        opacity_loss = self.loss_func(representation[:, -1], reconstruction[:, -1])
        self.log(stage + '_opacity_loss', opacity_loss)

        colour_loss = self.loss_func(representation[:, :-1], reconstruction[:, :-1])
        self.log(stage + '_colour_loss', colour_loss)

        del representation, reconstruction
        if self.weight_opacity:
            loss = opacity_loss + colour_loss
        return loss

    def training_step(self, batch, batch_idx):

        return self.calculate_loss(batch, stage='train')

    def test_step(self, batch, batch_idx):
        self.calculate_loss(batch, stage='test')

    def validation_step(self, batch, batch_idx):
        self.calculate_loss(batch, stage='val')


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}




if __name__ == "__main__":
    wandb_logger = WandbLogger(project='GridReconstruction')
    dataset_loader = RepairDatasetLoader(batch_size=2, dataset_type="FixedGridDataset",
                                         representation_folder_name="grids", num_workers=2, data_dir="~/masters/datasets/")
    L.seed_everything(42)
    model = GridReconstruction(weight_opacity=True)
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(dirpath='GridReconstructionCheckpoints/weighted_loss/')
    trainer = L.Trainer(max_epochs=20, logger=wandb_logger, accelerator='gpu', accumulate_grad_batches=10, callbacks=[checkpoint_callback])
    trainer.fit(model, datamodule=dataset_loader)
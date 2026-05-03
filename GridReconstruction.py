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


class GridReconstructionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
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



    def forward(self, representation):
        x = self.encoder(representation)
        x = self.decoder(x)
        return x


class GridReconstruction(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = GridReconstructionNetwork()
        self.lr = 5e-4

    def calculate_loss(self, batch, stage):
        representation = batch
        reconstruction = self.model(representation)

        loss = nn.functional.l1_loss(representation, reconstruction)# * weight.repeat(1, 3))
        self.log(stage + '_loss', loss)
        del representation, reconstruction
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
    model = GridReconstruction()
    trainer = L.Trainer(max_epochs=20, logger=wandb_logger, accelerator='gpu', accumulate_grad_batches=10)
    trainer.fit(model, datamodule=dataset_loader)
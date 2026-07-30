from sympy import false

from DatasetLoading import RepairDatasetLoader
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from torchmetrics.segmentation import DiceScore
from pytorch_lightning.callbacks import LearningRateMonitor
import torch.nn.functional as F

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='InterpolatableGridReconstruction training')
    parser.add_argument('--small_bottleneck', action='store_true', help='Use small bottleneck architecture')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--scale', type=int, default=1, help='Scale of the latent size')
    parser.add_argument('--loss_method', type=str, default="WO", help='Loss method')
    parser.add_argument("--low_acc", action='store_true', help="Use a lower floating point precision for testing")
    parser.add_argument("--no_logger", action='store_true', help="Disable logging to Weights and Biases")



    args = parser.parse_args()
    print(args)

import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger
import os

torch.set_float32_matmul_precision('medium')

class DebugBlock(nn.Module):
    def __init__(self, debug_string):
        super().__init__()
        self.debug_string = debug_string

    def forward(self, x):
        print(self.debug_string, x.shape)
        return x

class InterpolatableGridReconstructionNetwork(nn.Module):
    def __init__(self, small_bottleneck=False, scale=1):
        super().__init__()

        self.first_encoder = nn.Sequential(
            nn.Conv3d(96 + 288, 32*scale, kernel_size=1, stride=1, padding=0),  # 96 -> 96
            nn.BatchNorm3d(32*scale),
            nn.ReLU(),
            nn.Conv3d(32 * scale, 64 * scale, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64 * scale),
            nn.ReLU(),
            nn.Conv3d(64 * scale, 64 * scale, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64 * scale),
            nn.ReLU(),
            nn.MaxPool3d(2),  # 96 -> 48
            nn.Conv3d(64 * scale, 128 * scale, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128 * scale),
            nn.ReLU(),
            nn.MaxPool3d(2),  # 48 -> 24
            nn.Conv3d(128 * scale, 128 * scale, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128 * scale),
            nn.ReLU(),
            nn.MaxPool3d(2),  # 24 -> 12
            nn.Conv3d(128 * scale, 128 * scale, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128 * scale),
            nn.ReLU(),
            nn.Conv3d(128 * scale, 128 * scale, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128 * scale),
            nn.ReLU(),
        )

        self.last_decoder = nn.Sequential(
            nn.ConvTranspose3d(128 * scale, 128 * scale, kernel_size=4, stride=2, padding=1),  # 12 -> 24
            nn.BatchNorm3d(128 * scale),
            nn.ReLU(),
            nn.Conv3d(128 * scale, 128 * scale, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128 * scale),
            nn.ReLU(),
            nn.Conv3d(128 * scale, 128 * scale, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128 * scale),
            nn.ReLU(),
            nn.ConvTranspose3d(128 * scale, 128 * scale, kernel_size=4, stride=2, padding=1),  # 24 -> 48
            nn.BatchNorm3d(128 * scale),
            nn.ReLU(),
            nn.Conv3d(128 * scale, 64 * scale, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64 * scale),
            nn.ReLU(),
            nn.Conv3d(64 * scale, 64 * scale, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64 * scale),
            nn.ReLU(),
            nn.ConvTranspose3d(64 * scale, 64 * scale, kernel_size=4, stride=2, padding=1),  # 48 -> 96
            nn.BatchNorm3d(64 * scale),
            nn.ReLU(),
            nn.Conv3d(64 * scale, 64 * scale, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64 * scale),
            nn.ReLU(),
            nn.Conv3d(64 * scale, 64 * scale, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64 * scale),
            nn.ReLU(),
            nn.Conv3d(64 * scale, 32 * scale, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32 * scale),
            nn.ReLU(),
            nn.Conv3d(32 * scale, 96+288, kernel_size=3, stride=1, padding=1),)


        if small_bottleneck:
            self.extra_encoder = nn.Sequential(
                nn.MaxPool3d(2),  # 12 -> 6
                nn.Conv3d(128 * scale, 128 * scale, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(128 * scale),
                nn.ReLU(),
                nn.Conv3d(128 * scale, 128 * scale, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(128 * scale),
                nn.ReLU())

            self.extra_decoder = nn.Sequential(
                nn.ConvTranspose3d(128 * scale, 128 * scale, kernel_size=4, stride=2, padding=1),  # 6 -> 12
                nn.BatchNorm3d(128 * scale),
                nn.ReLU(),
                nn.Conv3d(128 * scale, 128 * scale, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(128 * scale),
                nn.ReLU(),
                nn.Conv3d(128 * scale, 128 * scale, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(128 * scale),
                nn.ReLU(),
            )

        self.sigmoid = nn.Sigmoid()


    def encoder(self, x):
        x = self.first_encoder(x)
        if hasattr(self, 'extra_encoder'):
            x = self.extra_encoder(x)
        return x

    def decoder(self, x):
        if hasattr(self, 'extra_decoder'):
            x = self.extra_decoder(x)
        x = self.last_decoder(x)
        return x

    def forward(self, representation):
        x = self.encoder(representation)
        x = self.decoder(x)
        opacity = x[:, -1:]
        colour = x[:, :-1]
        opacity = self.sigmoid(opacity)
        x = torch.cat((colour, opacity), dim=1)
        #if torch.isnan(x).any():
        #    print("NaN values found in output!")
        #    print(colour.isnan().sum(), opacity.isnan().sum())
        return x


class InterpolatableGridReconstruction(L.LightningModule):
    def __init__(self, ckpt_dir, loss_method, small_bottleneck=False, scale=1, learning_rate=5e-4):
        super().__init__()
        self.model = InterpolatableGridReconstructionNetwork(small_bottleneck, scale=scale)
        self.lr = learning_rate
        self.small_bottleneck = small_bottleneck
        self.scale = scale
        self.mse_loss = nn.MSELoss()
        self.save_hyperparameters()
        self.loss_func = lambda a, b : self.mse_loss(a, b) ** 0.5
        self.loss_method = loss_method
        self.dice_loss_score = DiceScore(num_classes=2, include_background=False, input_format='index')
        self.ckpt_dir = ckpt_dir

    def get_dice_score(self, representation_opacity, reconstruction_opacity):
        representation_opacity = (representation_opacity > 0.5).float()
        dice_score = 2 * torch.sum(representation_opacity * reconstruction_opacity, dim=[2, 3, 4]) / (torch.sum(representation_opacity, dim=[2, 3, 4]) + torch.sum(reconstruction_opacity, dim=[2, 3, 4]) + 1e-8)
        dice_score = torch.mean(dice_score)

        return dice_score


    def density_to_opacity(self, density, opacity_multiplier):
        summed_density = torch.sum(density, dim=1, keepdim=True)
        density_grid = F.softplus(summed_density+10)
        return 1. - torch.exp(-density_grid * opacity_multiplier)

    def calculate_loss(self, batch, stage):
        grid, opacity_multiplier = batch
        reconstruction = self.model(grid)

        mse_loss = self.mse_loss(grid, reconstruction)

        self.log(stage + '_mse_loss', mse_loss)

        rmse_loss = mse_loss ** 0.5
        self.log(stage + '_rmse_loss', rmse_loss)

        density_grid = grid[:, :96]
        colour_grid = grid[:, 96:]

        reconstructed_density_grid = reconstruction[:, :96]
        reconstructed_colour_grid = reconstruction[:, 96:]


        opacity_grid = self.density_to_opacity(density_grid, opacity_multiplier)
        opacity_mask = (opacity_grid > 0.3).expand(-1, 288, -1, -1, -1)

        reconstructed_opacity_grid = self.density_to_opacity(reconstructed_density_grid, opacity_multiplier)

        density_loss = self.loss_func(density_grid, reconstructed_density_grid)
        self.log(stage + '_density_loss', rmse_loss)

        opacity_loss = self.loss_func(opacity_grid, reconstructed_opacity_grid)
        self.log(stage + '_opacity_loss', opacity_loss)

        mask_colour_loss = self.loss_func(colour_grid[opacity_mask], reconstructed_colour_grid[opacity_mask])
        self.log(stage + '_mask_colour_loss', mask_colour_loss)

        dice_loss = self.get_dice_score(opacity_grid, reconstructed_opacity_grid)
        del opacity_grid, reconstructed_opacity_grid, density_grid, colour_grid, reconstructed_density_grid, reconstructed_colour_grid, opacity_mask

        self.log(stage + '_dice_loss', dice_loss)
        dice_loss = (1 - dice_loss)

        if self.loss_method == "WO":
            final_loss = opacity_loss + mask_colour_loss + density_loss + density_loss
        elif self.loss_method == "RMSE":
            final_loss = rmse_loss
        elif self.loss_method == "Dice":
            final_loss = dice_loss
        elif self.loss_method == "WO+Dice":
            final_loss = opacity_loss + mask_colour_loss + dice_loss + density_loss
        elif self.loss_method == "Dice+Mask":
            final_loss = dice_loss + mask_colour_loss
        else:
            final_loss = rmse_loss

        del opacity_loss, dice_loss, mask_colour_loss, density_loss, rmse_loss, mse_loss
        return final_loss

    def training_step(self, batch, batch_idx):
        return self.calculate_loss(batch, stage='train')

    def test_step(self, batch, batch_idx):
        self.calculate_loss(batch, stage='test')

    def validation_step(self, batch, batch_idx):
        self.calculate_loss(batch, stage='val')
        if self.current_epoch % 2 == 0:
            self.trainer.save_checkpoint(os.path.join(self.ckpt_dir, f"epoch_{self.current_epoch}.ckpt"))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_total_loss"}




if __name__ == "__main__":


    datasets_path = data_dir = "~/masters/datasets/" if not args.low_acc else "~/Documents/masters/datasets/"

    batch_size_dict = {"BS" : 1, "acc" : 1} if args.no_logger else {1 : {"BS" : 3, "acc" : 10}, 2 : {"BS" : 2, "acc" : 15}, 3 : {"BS" : 1, "acc" : 30}}[args.scale]

    dataset_loader = RepairDatasetLoader(batch_size=batch_size_dict["BS"], dataset_type="InterpolatableGridDataset",
                                         representation_folder_name="interpolatableGrids", num_workers=3, data_dir=datasets_path)
    L.seed_everything(42)
    run_name = f"loss={args.loss_method}_scale={args.scale}"


    if args.small_bottleneck:
        run_name += "_small_bottleneck"
    if args.lr != 1e-3:
        run_name += f"_lr={args.lr}"

    wandb_logger = False if args.no_logger else WandbLogger(name=run_name, project='InterpolatableGridReconstruction')
    ckpt_dir = f"GridReconstructionCheckpoints/{run_name}/"

    model = InterpolatableGridReconstruction(ckpt_dir=ckpt_dir, loss_method=args.loss_method, small_bottleneck=args.small_bottleneck, learning_rate=args.lr, scale=args.scale)

    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(dirpath=ckpt_dir)
    epochs = 200
    precision = "16-true" if args.low_acc else "32-true"
    lr_monitor = LearningRateMonitor(logging_interval='step')
    accelerator = "cpu" if args.no_logger else "gpu"
    trainer = L.Trainer(max_epochs=epochs, accelerator=accelerator, accumulate_grad_batches=batch_size_dict["acc"], callbacks=[checkpoint_callback, lr_monitor], precision=precision, logger=wandb_logger)
    trainer.fit(model, datamodule=dataset_loader)
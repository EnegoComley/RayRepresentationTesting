from sympy import false

from DatasetLoading import RepairDatasetLoader
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from torchmetrics.segmentation import DiceScore

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GridReconstruction training')
    parser.add_argument('--small_bottleneck', action='store_true', help='Use small bottleneck architecture')
    parser.add_argument('--overfit', action='store_true', help='Overfit the model on a small subset of the data for debugging')
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


class ResnetBlock3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = residual + out
        out = self.bn2(out)
        out = self.relu2(out)
        return out

class DebugBlock(nn.Module):
    def __init__(self, debug_string):
        super().__init__()
        self.debug_string = debug_string

    def forward(self, x):
        print(self.debug_string, x.shape)
        return x

class GridReconstructionNetwork(nn.Module):
    def __init__(self, small_bottleneck=False, scale=1):
        super().__init__()

        self.first_encoder = nn.Sequential(
            nn.Conv3d(32, 64 * scale, kernel_size=3, stride=1, padding=1),  # 96 -> 96
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
            nn.Conv3d(64 * scale, 32, kernel_size=3, stride=1, padding=1))


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


class GridReconstruction(L.LightningModule):
    def __init__(self, ckpt_dir, loss_method, small_bottleneck=False, scale=1, learning_rate=5e-4):
        super().__init__()
        self.model = GridReconstructionNetwork(small_bottleneck, scale=scale)
        self.lr = learning_rate
        self.small_bottleneck = small_bottleneck
        self.scale = scale
        self.mse_loss = nn.MSELoss()
        self.loss_func = lambda a, b: self.mse_loss(a, b) ** 0.5
        self.save_hyperparameters()
        self.loss_method = loss_method
        self.dice_loss_score = DiceScore(num_classes=2, include_background=False, input_format='index')
        self.ckpt_dir = ckpt_dir

    def get_dice_score(self, representation, reconstruction):
        representation_opacity = representation[:, -1:]
        representation_opacity = (representation_opacity > 0.5).float()
        reconstruction_opacity = reconstruction[:, -1:]
        dice_score = 2 * torch.sum(representation_opacity * reconstruction_opacity, dim=[2, 3, 4]) / (torch.sum(representation_opacity, dim=[2, 3, 4]) + torch.sum(reconstruction_opacity, dim=[2, 3, 4]) + 1e-8)
        dice_score = torch.mean(dice_score)

        return dice_score

    def calculate_loss(self, batch, stage):
        representation = batch
        reconstruction = self.model(representation)

        loss = self.loss_func(representation, reconstruction)

        self.log(stage + '_rmse_loss', loss)

        opacity_loss = self.loss_func(representation[:, -1], reconstruction[:, -1])
        self.log(stage + '_opacity_loss', opacity_loss)

        colour_loss = self.loss_func(representation[:, :-1], reconstruction[:, :-1])
        self.log(stage + '_colour_loss', colour_loss)

        mask_colour_loss = self.loss_func(representation[:, :-1], reconstruction[:, :-1] * representation[:, -1:].repeat(1, reconstruction.size(1) - 1, 1, 1, 1))
        self.log(stage + '_mask_colour_loss', mask_colour_loss)

        dice_loss = self.get_dice_score(representation, reconstruction)
        del representation, reconstruction

        self.log(stage + '_dice_loss', dice_loss)
        dice_loss = (1 - dice_loss)

        total_loss = loss + opacity_loss + colour_loss + dice_loss + mask_colour_loss
        self.log(stage + '_total_loss', total_loss)

        if self.loss_method == "WO":
            final_loss = opacity_loss + colour_loss
        elif self.loss_method == "MSE":
            final_loss = loss
        elif self.loss_method == "Dice":
            final_loss = dice_loss
        elif self.loss_method == "WO+Dice":
            final_loss = opacity_loss + colour_loss + dice_loss
        elif self.loss_method == "Dice+Mask":
            final_loss = dice_loss + opacity_loss + mask_colour_loss
        elif self.loss_method == "WO+Dice+Mask":
            final_loss = opacity_loss + colour_loss + dice_loss + mask_colour_loss
        else:
            final_loss = loss

        del opacity_loss, colour_loss, dice_loss
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

    batch_size_dict = {1 : {"BS" : 20, "acc" : 1}, 2 : {"BS" : 10, "acc" : 2}, 3 : {"BS" : 7, "acc" : 3}}[args.scale]

    dataset_loader = RepairDatasetLoader(batch_size=batch_size_dict["BS"], dataset_type="FixedGridDataset",
                                         representation_folder_name="gridswithRepresentation", num_workers=3, data_dir=datasets_path, overfit=args.overfit)
    L.seed_everything(42)
    run_name = f"loss={args.loss_method}_scale={args.scale}"



    if args.overfit:
        run_name += "_overfit"
    if args.small_bottleneck:
        run_name += "_small_bottleneck"
    if args.lr != 1e-3:
        run_name += f"_lr={args.lr}"

    wandb_logger = False if args.no_logger else WandbLogger(name=run_name, project='GridReconstruction96')
    ckpt_dir = f"GridReconstructionCheckpoints/"

    model = GridReconstruction(ckpt_dir=ckpt_dir, loss_method=args.loss_method, small_bottleneck=args.small_bottleneck, learning_rate=args.lr, scale=args.scale)

    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(dirpath=ckpt_dir)
    epochs = 200 if args.overfit else 20
    precision = "16-true" if args.low_acc else "32-true"
    trainer = L.Trainer(max_epochs=epochs, accelerator='gpu', accumulate_grad_batches=batch_size_dict["acc"], callbacks=[checkpoint_callback], precision=precision, logger=wandb_logger)
    trainer.fit(model, datamodule=dataset_loader)
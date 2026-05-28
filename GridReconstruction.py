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
    parser.add_argument('--double_channels', action='store_true', help='Double the number of channels')
    parser.add_argument('--overfit', action='store_true', help='Double the number of channels')
    parser.add_argument('--res_net', action='store_true', help='Use a res net like structure')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--loss_method', type=str, default="WO", help='Initial learning rate')
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
    def __init__(self, small_bottleneck=False, double_channels=False, res_net=False):
        super().__init__()
        scale = 2 if double_channels else 1
        if not res_net:
            if not small_bottleneck:
                self.encoder = nn.Sequential(
                    nn.Conv3d(32, 64 * scale, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm3d(64 * scale),
                    nn.ReLU(),
                    nn.Conv3d(64 * scale, 64 * scale, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm3d(64 * scale),
                    nn.ReLU(),
                    nn.MaxPool3d(2), # 200 -> 100
                    nn.Conv3d(64 * scale, 128 * scale, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm3d(128 * scale),
                    nn.ReLU(),
                    nn.MaxPool3d(2), # 100 -> 50
                    nn.Conv3d(128 * scale, 128 * scale, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm3d(128 * scale),
                    nn.ReLU(),
                    nn.MaxPool3d(2), # 50 -> 25
                    nn.Conv3d(128 * scale, 128 * scale, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm3d(128 * scale),
                    nn.ReLU(),
                    nn.Conv3d(128 * scale, 128 * scale, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm3d(128 * scale),
                    nn.ReLU(),
                )

                self.decoder = nn.Sequential(
                    nn.ConvTranspose3d(128 * scale, 128 * scale, kernel_size=4, stride=2, padding=1), # 25 -> 50
                    nn.BatchNorm3d(128 * scale),
                    nn.ReLU(),
                    nn.ConvTranspose3d(128 * scale, 64 * scale, kernel_size=4, stride=2, padding=1), # 50 -> 100
                    nn.BatchNorm3d(64 * scale),
                    nn.ReLU(),
                    nn.ConvTranspose3d(64 * scale, 64 * scale, kernel_size=4, stride=2, padding=1), # 100 -> 200
                    nn.BatchNorm3d(64 * scale),
                    nn.ReLU(),
                    nn.Conv3d(64 * scale, 32, kernel_size=3, stride=1, padding=1)
                )
            else:
                self.encoder = nn.Sequential(
                    nn.Conv3d(32, 64 * scale, kernel_size=3, stride=1, padding=13), # 200 -> 224
                    nn.BatchNorm3d(64 * scale),
                    nn.ReLU(),
                    nn.Conv3d(64 * scale, 64 * scale, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm3d(64 * scale),
                    nn.ReLU(),
                    nn.MaxPool3d(2),  # 224 -> 112
                    nn.Conv3d(64 * scale, 128 * scale, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm3d(128 * scale),
                    nn.ReLU(),
                    nn.MaxPool3d(2),  # 112 -> 56
                    nn.Conv3d(128 * scale, 128 * scale, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm3d(128 * scale),
                    nn.ReLU(),
                    nn.MaxPool3d(2),  # 56 -> 28
                    nn.Conv3d(128 * scale, 128 * scale, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm3d(128 * scale),
                    nn.ReLU(),
                    nn.Conv3d(128 * scale, 128 * scale, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm3d(128 * scale),
                    nn.ReLU(),
                    nn.MaxPool3d(2),  # 28 -> 14
                    nn.Conv3d(128 * scale, 128 * scale, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm3d(128 * scale),
                    nn.ReLU(),
                    nn.Conv3d(128 * scale, 128 * scale, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm3d(128 * scale),
                    nn.ReLU(),
                    nn.MaxPool3d(2),  # 14 -> 7
                    nn.Conv3d(128 * scale, 128 * scale, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm3d(128 * scale),
                    nn.ReLU(),
                    nn.Conv3d(128 * scale, 128 * scale, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm3d(128 * scale),
                    nn.ReLU(),
                )

                self.decoder = nn.Sequential(
                    nn.ConvTranspose3d(128 * scale, 128 * scale, kernel_size=4, stride=2, padding=1),  # 7 -> 14
                    nn.BatchNorm3d(128 * scale),
                    nn.ReLU(),
                    nn.Conv3d(128 * scale, 128 * scale, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm3d(128 * scale),
                    nn.ReLU(),
                    nn.ConvTranspose3d(128 * scale, 128 * scale, kernel_size=4, stride=2, padding=1),  # 14 -> 28
                    nn.BatchNorm3d(128 * scale),
                    nn.ReLU(),
                    nn.Conv3d(128 * scale, 128 * scale, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm3d(128 * scale),
                    nn.ReLU(),
                    nn.ConvTranspose3d(128 * scale, 128 * scale, kernel_size=4, stride=2, padding=1),  # 28 -> 56
                    nn.BatchNorm3d(128 * scale),
                    nn.ReLU(),
                    nn.Conv3d(128 * scale, 128 * scale, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm3d(128 * scale),
                    nn.ReLU(),
                    nn.ConvTranspose3d(128 * scale, 128 * scale, kernel_size=4, stride=2, padding=1),  # 56 -> 112
                    nn.BatchNorm3d(128 * scale),
                    nn.ReLU(),
                    nn.Conv3d(128 * scale, 64 * scale, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm3d(64 * scale),
                    nn.ReLU(),
                    nn.ConvTranspose3d(64 * scale, 64 * scale, kernel_size=4, stride=2, padding=13),  # 112 -> 200
                    nn.BatchNorm3d(64 * scale),
                    nn.ReLU(),
                    nn.Conv3d(64 * scale, 64 * scale, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm3d(64 * scale),
                    nn.ReLU(),
                    nn.Conv3d(64 * scale, 32, kernel_size=3, stride=1, padding=1))
        else:
            self.encoder = nn.Sequential(
                nn.Conv3d(32, 128 * scale, kernel_size=3, stride=1, padding=1+4),  # 200 -> 208
                nn.BatchNorm3d(128 * scale),
                nn.ReLU(),
                ResnetBlock3D(128 * scale),
                nn.MaxPool3d(2),  # 208 -> 104
                ResnetBlock3D(128 * scale),
                nn.MaxPool3d(2),  # 104 -> 52
                ResnetBlock3D(128 * scale),
                nn.MaxPool3d(2),  # 52 -> 26
                ResnetBlock3D(128 * scale),
                ResnetBlock3D(128 * scale),
                nn.MaxPool3d(2),  # 26 -> 13
                ResnetBlock3D(128 * scale),
                ResnetBlock3D(128 * scale),
            )

            self.decoder = nn.Sequential(
                nn.ConvTranspose3d(128 * scale, 128 * scale, kernel_size=4, stride=2, padding=1),  # 14 -> 28
                nn.BatchNorm3d(128 * scale),
                nn.ReLU(),
                ResnetBlock3D(128 * scale),
                nn.ConvTranspose3d(128 * scale, 128 * scale, kernel_size=4, stride=2, padding=1),  # 28 -> 56
                nn.BatchNorm3d(128 * scale),
                nn.ReLU(),
                ResnetBlock3D(128 * scale),
                nn.ConvTranspose3d(128 * scale, 128 * scale, kernel_size=4, stride=2, padding=1),  # 56 -> 112
                nn.BatchNorm3d(128 * scale),
                nn.ReLU(),
                ResnetBlock3D(128 * scale),
                nn.ConvTranspose3d(128 * scale, 128 * scale, kernel_size=4, stride=2, padding=1 + 4),  # 112 -> 200
                nn.BatchNorm3d(128 * scale),
                nn.ReLU(),
                ResnetBlock3D(128 * scale),
                nn.Conv3d(128 * scale, 32, kernel_size=3, stride=1, padding=1))

        self.sigmoid = nn.Sigmoid()



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
    def __init__(self, ckpt_dir, loss_method, small_bottleneck=False, double_channels=False, res_net=False, learning_rate=5e-4):
        super().__init__()
        self.model = GridReconstructionNetwork(small_bottleneck, double_channels, res_net)
        self.lr = learning_rate
        self.small_bottleneck = small_bottleneck
        self.double_channels = double_channels
        self.res_net = res_net
        self.mse_loss = nn.MSELoss()
        self.loss_func = lambda a, b: self.mse_loss(a, b) ** 0.5
        self.save_hyperparameters()
        self.loss_method = loss_method
        self.dice_loss_score = DiceScore(num_classes=2, include_background=False, input_format='index')
        self.ckpt_dir = ckpt_dir

    def get_dice_score(self, representation, reconstruction):
        representation_opacity = representation[:, -1:]
        reconstruction_opacity = reconstruction[:, -1:]
        dice_score = 2 * torch.sum(representation_opacity * reconstruction_opacity, dim=[2, 3, 4]) / (torch.sum(representation_opacity, dim=[2, 3, 4]) + torch.sum(reconstruction_opacity, dim=[2, 3, 4]) + 1e-8)
        dice_score = torch.mean(dice_score)

        return dice_score

    def calculate_loss(self, batch, stage):
        representation = batch
        reconstruction = self.model(representation)

        loss = self.loss_func(representation, reconstruction)

        self.log(stage + '_loss', loss)
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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}




if __name__ == "__main__":

    wandb_logger = False if args.no_logger else WandbLogger(project='GridReconstruction')

    datasets_path = data_dir = "~/masters/datasets/" if not args.low_acc else "~/Documents/masters/datasets/"
    dataset_loader = RepairDatasetLoader(batch_size=1, dataset_type="FixedGridDataset",
                                         representation_folder_name="gridswithRepresentation", num_workers=2, data_dir=datasets_path, overfit=args.overfit)
    L.seed_everything(42)
    ckpt_dir = f"GridReconstructionCheckpoints/loss={args.loss_method}_small_bottleneck={args.small_bottleneck}_double_channels={args.double_channels}_learning_rate={args.lr}"

    if args.overfit:
        ckpt_dir += "_overfit"
    if args.res_net:
        ckpt_dir += "_resnet"

    model = GridReconstruction(ckpt_dir=ckpt_dir, loss_method=args.loss_method, small_bottleneck=args.small_bottleneck, double_channels=args.double_channels, res_net=args.res_net, learning_rate=args.lr)

    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(dirpath=ckpt_dir)
    epochs = 200 if args.overfit else 20
    precision = "16-true" if args.low_acc else "32-true"
    trainer = L.Trainer(max_epochs=epochs, accelerator='gpu', accumulate_grad_batches=20, callbacks=[checkpoint_callback], precision=precision, logger=wandb_logger)
    trainer.fit(model, datamodule=dataset_loader)
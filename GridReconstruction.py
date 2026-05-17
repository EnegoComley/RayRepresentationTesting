from sympy import false

from DatasetLoading import RepairDatasetLoader
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from torchmetrics.segmentation import DiceScore

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GridReconstruction training')
    parser.add_argument('--weight_opacity', action='store_true', help='Weight opacity separately in loss')
    parser.add_argument('--small_bottleneck', action='store_true', help='Use small bottleneck architecture')
    parser.add_argument('--double_channels', action='store_true', help='Double the number of channels')
    parser.add_argument('--overfit', action='store_true', help='Double the number of channels')
    parser.add_argument('--res_net', action='store_true', help='Use a res net like structure')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument("--low_acc", action='store_true', help="Use a lower floating point precision for testing")
    parser.add_argument("--dice_loss", action='store_true', help="Use dice loss")



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



    def forward(self, representation):
        x = self.encoder(representation)
        x = self.decoder(x)
        return x


class GridReconstruction(L.LightningModule):
    def __init__(self, weight_opacity=False, small_bottleneck=False, double_channels=False, res_net=False, dice_loss=false, learning_rate=5e-4):
        super().__init__()
        self.model = GridReconstructionNetwork(small_bottleneck, double_channels, res_net)
        self.lr = learning_rate
        self.weight_opacity = weight_opacity
        self.small_bottleneck = small_bottleneck
        self.double_channels = double_channels
        self.res_net = res_net
        self.loss_func = nn.MSELoss()
        self.save_hyperparameters()
        self.dice_loss_score = DiceScore(num_classes=2, include_background=False, input_format='index')
        self.use_dice_loss = dice_loss


    def categorise_representation(self, representation, threshold=0.3):
        # Representation is shape (batch_size, 32, 200, 200, 200)
        categories = (representation[:, -1:] > threshold).long()
        return categories

    def get_dice_loss(self, representation, reconstruction):
        dice_loss = self.dice_loss_score(self.categorise_representation(representation), self.categorise_representation(reconstruction))
        return dice_loss

    def calculate_loss(self, batch, stage):
        representation = batch
        reconstruction = self.model(representation)

        loss = self.loss_func(representation, reconstruction)# * weight.repeat(1, 3))

        self.log(stage + '_loss', loss)

        opacity_loss = self.loss_func(representation[:, -1], reconstruction[:, -1])
        self.log(stage + '_opacity_loss', opacity_loss)

        colour_loss = self.loss_func(representation[:, :-1], reconstruction[:, :-1])
        self.log(stage + '_colour_loss', colour_loss)



        if self.weight_opacity:
            loss = opacity_loss + colour_loss

        del opacity_loss, colour_loss
        dice_loss = self.get_dice_loss(representation, reconstruction)
        del representation, reconstruction

        self.log(stage + '_dice_loss', dice_loss)

        if self.use_dice_loss:
            loss = loss + dice_loss
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
    datasets_path = data_dir = "~/masters/datasets/" if not args.low_acc else "~/Documents/masters/datasets/"
    dataset_loader = RepairDatasetLoader(batch_size=2, dataset_type="FixedGridDataset",
                                         representation_folder_name="gridswithRepresentation", num_workers=2, data_dir=datasets_path, overfit=args.overfit)
    L.seed_everything(42)
    model = GridReconstruction(weight_opacity=args.weight_opacity, small_bottleneck=args.small_bottleneck, double_channels=args.double_channels, res_net=args.res_net, learning_rate=args.lr, dice_loss=args.dice_loss)

    ckpt_dir = f"GridReconstructionCheckpoints/BFLOATweight_opacity={args.weight_opacity}_small_bottleneck={args.small_bottleneck}_double_channels={args.double_channels}_learning_rate={args.lr}_res_net={args.res_net}"
    if args.overfit:
        ckpt_dir += "_overfit"
    if args.dice_loss:
        ckpt_dir += "_dice_loss"
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(dirpath=ckpt_dir)
    epochs = 200 if args.overfit else 20
    precision = "16-true" if args.low_acc else "bf16-mixed"
    trainer = L.Trainer(max_epochs=epochs, logger=wandb_logger, accelerator='gpu', accumulate_grad_batches=20, callbacks=[checkpoint_callback], precision=precision)
    trainer.fit(model, datamodule=dataset_loader)
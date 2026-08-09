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



class RayManager:
    def __init__(self, device = None):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device

        self.aabb =  torch.tensor([[-1.7541, -1.7541, -1.7541],[ 1.7541,  1.7541,  1.7541]], device=device)
        self.near_far = [0.01, 6.0]
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.aabbDiag = torch.sqrt(torch.sum(torch.square(self.aabbSize)))
        self.gridSize = torch.tensor([96, 96, 96], device=device)
        self.units= self.aabbSize / (self.gridSize-1)
        self.step_ratio = 0.5
        self.stepSize=torch.mean(self.units)* self.step_ratio
        self.nSamples = int((self.aabbDiag / self.stepSize).item()) + 1
        self.invaabbSize = 2.0/self.aabbSize


    def sample_ray(self, rays_o, rays_d, device = torch.device("cpu"), vecMode = [0, 1, 2]):

        near, far = self.near_far
        vec = torch.where(rays_d==0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.aabb[1] - rays_o) / vec
        rate_b = (self.aabb[0] - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)

        rng = torch.arange(self.nSamples)[None].float()

        step = self.stepSize * rng.to(rays_o.device)
        interpx = (t_min[...,None] + step)

        rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * interpx[...,None]
        mask_outbbox = ((self.aabb[0]>rays_pts) | (rays_pts>self.aabb[1])).any(dim=-1)

        xyz_sampled = rays_pts
        z_vals = interpx
        ray_valid = ~mask_outbbox
        del rays_pts, interpx, mask_outbbox

        # Normalise the coordinates for interpolation.
        xyz_sampled = (xyz_sampled-self.aabb[0]) * self.invaabbSize - 1

        # Change to the correct axis mode
        xyz_sampled = xyz_sampled[:, :, :, vecMode]

        return xyz_sampled, z_vals, ray_valid



    def raw2weight(self, sigma, dist):
        # Function taken from TensoRF
        # sigma, dist  [N_rays, N_samples]
        alpha = 1. - torch.exp(-sigma * dist)  # Percentage of colour absorbed at each point, [N_rays, N_samples]
        # When sigma*dist is 0 (sigma is transparent) it is 1 - 1 = 0, when sigma*dist is large (sigma is opaque) it is 1 - 0 = 1. E^(-x) approaches 0

        T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], alpha.shape[1], 1).to(alpha.device), 1. - alpha + 1e-10], -1), -1)
        # T is the percentage of light at each point along the ray. It starts at 1 and get's successively multiplied by the previous amount of light taken.

        weights = alpha * T[:, :, :-1]  # [N_rays, N_samples]
        # The weight is then the amount of light absorbed at each point multiplied by the amount of light remaining.
        return weights

    def get_opacity_weight(self, xyz_sampled, density_grid, ray_valid, z_vals):
        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)

        B, _, _, _ = xyz_sampled.shape

        # xyz_sampled: (N,3) in [-1,1]
        grid = xyz_sampled[ray_valid].view(B, 1, -1, 1, 3)

        sigma_feature = F.grid_sample(
            density_grid,  # (B,C,D,H,W)
            grid,
            mode="bilinear",
            align_corners=True,
        )  # torch.Size([1, 96, 1, N, 1])

        sigma_feature = torch.sum(sigma_feature, dim=1).squeeze()

        validsigma = F.softplus(sigma_feature - 10)
        sigma[ray_valid] = validsigma

        dists = torch.cat((z_vals[:, :, 1:] - z_vals[:, :, :-1], torch.zeros_like(z_vals[:, :, :1])), dim=-1)

        weight = self.raw2weight(sigma, dists * 25)  # 25 is the normal distance scale in the tensoRF

        return weight

    def NoBasisMatRender(self, features):
        b, _ = features.shape
        features = features.view(b, 3, -1)
        rgb = torch.sum(features, dim=-1)
        return rgb

    def get_colour(self, xyz_sampled, colour_grid, weight, white_bg):

        B, _, _, _ = xyz_sampled.shape

        rgb = torch.zeros((*xyz_sampled.shape[:3], 3), device=xyz_sampled.device)

        app_mask = weight > 0.0001

        if app_mask.any():
            grid = xyz_sampled[app_mask].view(B, 1, -1, 1, 3)

            app_features = F.grid_sample(
                colour_grid,      # (B,C,D,H,W)
                grid,
                mode="bilinear",
                align_corners=True,
            ).squeeze().T

            valid_rgbs = self.NoBasisMatRender(app_features)
            rgb[app_mask] = valid_rgbs

        acc_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[..., None] * rgb, -2)

        if white_bg:
            rgb_map = rgb_map + (1. - acc_map[..., None])

        return rgb_map

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
        self.RayManager = RayManager()

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
        grid, opacity_multiplier, blank_edge_rays_o, blank_edge_rays_d, rgb_rays_o, rgb_rays_d, rgb_rays_c = batch
        reconstruction = self.model(grid)

        edge_xyz_sampled, edge_z_vals, edge_ray_valid = self.RayManager.sample_ray(blank_edge_rays_o, blank_edge_rays_d)
        rgb_xyz_sampled, rgb_z_vals, rgb_ray_valid = self.RayManager.sample_ray(rgb_rays_o, rgb_rays_d)
        density_reconstruction = reconstruction[:, 96]
        colour_reconstruction = reconstruction[:, 96:]

        edge_opacity = torch.sum(self.RayManager.get_opacity_weight(edge_xyz_sampled, density_reconstruction, edge_ray_valid, edge_z_vals), dim=-1)
        rgb_weight = self.RayManager.get_opacity_weight(rgb_xyz_sampled, density_reconstruction, rgb_ray_valid, rgb_z_vals)

        rgbs = self.RayManager.get_colour(rgb_xyz_sampled, colour_reconstruction, rgb_weight, False)
        del edge_xyz_sampled, edge_ray_valid, edge_z_vals, rgb_xyz_sampled, rgb_weight, rgb_z_vals , rgb_ray_valid

        mse_loss = self.mse_loss(grid, reconstruction)

        self.log(stage + '_mse_loss', mse_loss)

        rmse_loss = mse_loss ** 0.5
        self.log(stage + '_rmse_loss', rmse_loss)

        density_grid = grid[:, :96]
        colour_grid = grid[:, 96:]


        opacity_grid = self.density_to_opacity(density_grid, opacity_multiplier)
        opacity_mask = (opacity_grid > 0.3).expand(-1, 288, -1, -1, -1)

        reconstructed_opacity_grid = self.density_to_opacity(density_reconstruction, opacity_multiplier)

        density_loss = self.loss_func(density_grid, density_reconstruction)
        self.log(stage + '_density_loss', density_loss)

        opacity_loss = self.loss_func(opacity_grid, reconstructed_opacity_grid)
        self.log(stage + '_opacity_loss', opacity_loss)

        mask_colour_loss = self.loss_func(colour_grid[opacity_mask], colour_reconstruction[opacity_mask])
        self.log(stage + '_mask_colour_loss', mask_colour_loss)

        dice_loss = self.get_dice_score(opacity_grid, reconstructed_opacity_grid)
        del opacity_grid, reconstructed_opacity_grid, density_grid, colour_grid, density_reconstruction, colour_reconstruction, opacity_mask

        self.log(stage + '_dice_loss', dice_loss)
        dice_loss = (1 - dice_loss)

        edge_loss = self.loss_func(edge_opacity, torch.zeros_like(edge_opacity))
        self.log(stage + '_edge_loss', edge_loss)

        ray_colour_loss = self.loss_func(rgbs, rgb_rays_c)
        self.log(stage + '_ray_colour_loss', ray_colour_loss)

        if self.loss_method == "WO":
            final_loss = opacity_loss + mask_colour_loss
        elif self.loss_method == "RMSE":
            final_loss = rmse_loss
        elif self.loss_method == "Dice":
            final_loss = dice_loss
        elif self.loss_method == "WO+Dice":
            final_loss = opacity_loss + mask_colour_loss + dice_loss
        elif self.loss_method == "Dice+Mask":
            final_loss = dice_loss + mask_colour_loss
        elif self.loss_method == "Ray":
            final_loss = edge_loss + ray_colour_loss
        elif self.loss_method == "WO+Ray":
            final_loss = edge_loss + ray_colour_loss + opacity_loss + mask_colour_loss
        elif self.loss_method == "WO+Dice+Ray":
            final_loss = edge_loss + ray_colour_loss + opacity_loss + mask_colour_loss + dice_loss
        else:
            final_loss = rmse_loss

        del opacity_loss, dice_loss, mask_colour_loss, density_loss, rmse_loss, mse_loss
        self.log(stage + '_final_loss', final_loss)

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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_final_loss"}




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
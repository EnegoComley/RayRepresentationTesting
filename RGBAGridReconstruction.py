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
    parser = argparse.ArgumentParser(description='RGBAGridReconstruction training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--scale', type=int, default=1, help='Scale of the latent size')
    parser.add_argument('--downsamples', type=int, default=3, help='The number of times to half the size')
    parser.add_argument('--loss_method', type=str, default="WO", help='Loss method')
    parser.add_argument("--low_acc", action='store_true', help="Use a lower floating point precision for testing")
    parser.add_argument("--no_logger", action='store_true', help="Disable logging to Weights and Biases")
    parser.add_argument('--overfit', action='store_true', help='Overfit the model on a small subset of the data for debugging')
    parser.add_argument('--no_batch_norm', action='store_true', help="Don't use batch normalization")
    parser.add_argument('--split_model', action='store_true', help='Use a split model for training')
    parser.add_argument('--no_lr_reduce', action='store_true', help='Don\'t reduce the learning rate on plateau')



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



class RGBARayManager(nn.Module):
    def __init__(self, device = None, dtype=torch.float32):
        super().__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device

        self.aabb =  torch.tensor([[-1.7541, -1.7541, -1.7541],[ 1.7541,  1.7541,  1.7541]], device=device, dtype=dtype)
        self.near_far = [0.01, 6.0]
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.aabbDiag = torch.sqrt(torch.sum(torch.square(self.aabbSize)))
        self.gridSize = torch.tensor([96, 96, 96], device=device, dtype=dtype)
        self.units= self.aabbSize / (self.gridSize-1)
        self.step_ratio = 0.5
        self.stepSize=torch.mean(self.units)* self.step_ratio
        self.nSamples = int((self.aabbDiag / self.stepSize).item()) + 1
        self.invaabbSize = 2.0/self.aabbSize
        self.dtype = dtype


    def sample_ray(self, rays_o, rays_d, device = torch.device("cpu"), vecMode = [0, 1, 2]):
        near, far = self.near_far
        vec = torch.where(rays_d==0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.aabb[1] - rays_o) / vec
        rate_b = (self.aabb[0] - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)

        rng = torch.arange(self.nSamples)[None].float()
        if self.dtype == torch.float16:
            rng = rng.half()

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
        B = xyz_sampled.shape[0]

        sigma = torch.zeros(
            xyz_sampled.shape[:-1],
            device=xyz_sampled.device,
            dtype=density_grid.dtype,
        )

        xyz_flat = xyz_sampled.reshape(B, -1, 3)
        valid_flat = ray_valid.reshape(B, -1)

        n_valid = valid_flat.sum(dim=1)
        max_valid = int(n_valid.max())

        if max_valid > 0:
            # Positions of the valid entries within each batch
            batch_idx, point_idx = torch.where(valid_flat)

            # Position of each valid point within its batch's packed array
            packed_idx = (
                    torch.arange(batch_idx.numel(), device=xyz_sampled.device)
                    - torch.cat([
                torch.zeros(
                    1,
                    device=xyz_sampled.device,
                    dtype=torch.long,
                ),
                n_valid.cumsum(0)[:-1],
            ])[batch_idx]
            )

            # Pack into padded tensor
            grid = torch.zeros(
                B, max_valid, 3,
                device=xyz_sampled.device,
                dtype=xyz_sampled.dtype,
            )

            grid[batch_idx, packed_idx] = xyz_flat[batch_idx, point_idx]

            sigma_feature = F.grid_sample(
                density_grid,
                grid[:, :, None, None, :],
                mode="bilinear",
                align_corners=True,
            )

            sigma_feature = (
                sigma_feature
                .squeeze(-1)
                .squeeze(-1)
            )

            validsigma = F.softplus(sigma_feature - 10)

            sigma_flat = sigma.reshape(B, -1)
            for b in range(B):
                mask = valid_flat[b]
                n = n_valid[b]

                if n > 0:
                    sigma_flat[b, mask] = validsigma[b, :n]
            sigma = sigma_flat.reshape(*xyz_sampled.shape[:-1])

        dists = torch.cat((z_vals[:, :, 1:] - z_vals[:, :, :-1], torch.zeros_like(z_vals[:, :, :1])), dim=-1)

        weight = self.raw2weight(sigma, dists * 25)  # 25 is the normal distance scale in the tensoRF

        return weight

    def get_colour(self, xyz_sampled, colour_grid, weight, white_bg):

        B = xyz_sampled.shape[0]

        rgb = torch.zeros(
            (*xyz_sampled.shape[:3], 3),
            device=xyz_sampled.device,
            dtype=colour_grid.dtype,
        )

        app_mask = weight > 0.0001

        # Flatten all non-batch, non-coordinate dimensions
        xyz_flat = xyz_sampled.reshape(B, -1, 3)
        mask_flat = app_mask.reshape(B, -1)

        # Number of valid points for each batch element
        n_valid = mask_flat.sum(dim=1)
        max_valid = int(n_valid.max())

        if max_valid > 0:
            # Find all valid positions
            batch_idx, point_idx = torch.where(mask_flat)

            # Compute each point's position within its batch's packed array
            offsets = torch.cat([
                torch.zeros(
                    1,
                    device=xyz_sampled.device,
                    dtype=torch.long,
                ),
                n_valid.cumsum(dim=0)[:-1],
            ])

            packed_idx = (
                    torch.arange(
                        batch_idx.numel(),
                        device=xyz_sampled.device,
                    )
                    - offsets[batch_idx]
            )

            # Pack valid coordinates into a padded tensor
            # (B, max_valid, 3)
            grid = torch.zeros(
                B,
                max_valid,
                3,
                device=xyz_sampled.device,
                dtype=xyz_sampled.dtype,
            )

            grid[batch_idx, packed_idx] = xyz_flat[batch_idx, point_idx]

            # grid_sample expects:
            # input: (B, C, D, H, W)
            # grid:  (B, D_out, H_out, W_out, 3)
            grid = grid[:, :, None, None, :]

            valid_rgbs = F.grid_sample(
                colour_grid,
                grid,
                mode="bilinear",
                align_corners=True,
            )
            # (B, C, max_valid, 1, 1)

            valid_rgbs = (
                valid_rgbs
                .squeeze(-1)
                .squeeze(-1)
                .permute(0, 2, 1)
            )
            # (B, max_valid, C)

            # Scatter sampled RGB values back to their original positions
            rgb_flat = rgb.reshape(B, -1, 3)

            rgb_flat[batch_idx, point_idx] = valid_rgbs[
                batch_idx,
                packed_idx,
            ]
            rgb = rgb_flat.reshape(*xyz_sampled.shape[:-1], 3)

        acc_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[..., None] * rgb, -2)

        if white_bg:
            rgb_map = rgb_map + (1. - acc_map[..., None])

        return rgb_map

class RGBAGridReconstructionNetwork(nn.Module):
    def __init__(self, scale=1, downsamples = 3, no_batch_norm = False, channel_size= 4):
        super().__init__()

        class ConvBlock(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
                super().__init__()
                if not no_batch_norm:
                    self.block = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=padding),
                                               nn.BatchNorm3d(out_channels),
                                               nn.ReLU())
                else:
                    self.block = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=padding),
                                               nn.ReLU())

            def forward(self, x):
                return self.block(x)

        class DownBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.block = nn.Sequential(nn.MaxPool3d(2),
                                           ConvBlock(in_channels, out_channels, 3, 1, 1))

            def forward(self, x):
                return self.block(x)

        class UpBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.block = nn.Sequential(
                    nn.ConvTranspose3d(in_channels, in_channels, kernel_size=4, stride=2, padding=1),
                    # 24 -> 48
                    nn.BatchNorm3d(in_channels),
                    nn.ReLU(),
                    nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm3d(out_channels),
                    nn.ReLU(),
                    nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm3d(out_channels),
                    nn.ReLU())

            def forward(self, x):
                return self.block(x)



        encoder_blocks = [
            [
                ConvBlock(channel_size, 32 * scale, kernel_size=3, stride=1, padding=1),  # 96 -> 96
                ConvBlock(32 * scale, 64 * scale, kernel_size=3, stride=1, padding=1),
                ConvBlock(64 * scale, 64 * scale, kernel_size=3, stride=1, padding=1),
            ],
            DownBlock(64 * scale, 128 * scale),  # 96 -> 48
            DownBlock(128 * scale, 128 * scale),  # 46 -> 24
            [
                DownBlock(128 * scale, 128 * scale),  # 24 -> 12
                ConvBlock(128 * scale, 128 * scale, kernel_size=3, stride=1, padding=1),
            ],

            #The following downsample is not done by default
            [
                DownBlock(128 * scale, 128 * scale), # 12 -> 6
                ConvBlock(128 * scale, 128 * scale, kernel_size=3, stride=1, padding=1),
            ]

        ][:(downsamples + 1)]


        self.encoder = nn.Sequential(
            *[x for scale in encoder_blocks for x in (scale if type(scale) == list else [scale])]
        )

        decoder_blocks = [
            # The following downsample is not done by default
            UpBlock(128 * scale, 128 * scale),  # 6 -> 12


            # This bit is done by default.
            UpBlock(128 * scale, 128 * scale), # 12 -> 24
            UpBlock(128 * scale, 64 * scale), # 24 -> 48
            UpBlock(64 * scale, 64 * scale), # 48 -> 96
            [
                ConvBlock(64 * scale, 32 * scale, kernel_size=3, stride=1, padding=1),
                nn.Conv3d(32 * scale, channel_size, kernel_size=3, stride=1, padding=1)
            ]

        ][(-downsamples - 1):]

        self.decoder = nn.Sequential(
            # Flatten the blocks
            *[x for scale in decoder_blocks for x in (scale if type(scale) == list else [scale])]
        )


    def forward(self, representation):

        x = self.encoder(representation)
        x = self.decoder(x)
        #if torch.isnan(x).any():
        #    print("NaN values found in output!")
        #    print(colour.isnan().sum(), opacity.isnan().sum())
        return x

class SplitModel(nn.Module):
    def __init__(self, scale=1, downsamples=3, no_batch_norm=False):
        super().__init__()
        opacity_scale = scale // 2
        colour_scale = scale - opacity_scale
        self.opacity_model = RGBAGridReconstructionNetwork(opacity_scale, downsamples, no_batch_norm, channel_size=1)
        self.colour_model = RGBAGridReconstructionNetwork(colour_scale, downsamples, no_batch_norm, channel_size=3)

    def forward(self, representation):
        opacity = self.opacity_model(representation[:, :1])
        colour = self.colour_model(representation[:, 1:])
        return torch.cat([opacity, colour], dim=1)

class RGBAGridReconstruction(L.LightningModule):
    def __init__(self, ckpt_dir, loss_method, downsamples = 3, scale=1, learning_rate=5e-4, no_batch_norm=False, save_every_n_checkpoints=2, split_model=False, no_lr_reduce=False):
        super().__init__()

        if split_model:
            self.model = SplitModel(scale=scale, downsamples=downsamples, no_batch_norm=no_batch_norm)
        else:
            self.model = RGBAGridReconstructionNetwork(scale=scale, downsamples=downsamples, no_batch_norm=no_batch_norm)
        self.no_batch_norm = no_batch_norm
        self.lr = learning_rate
        self.downsamples = downsamples
        self.scale = scale
        self.split_model = split_model
        self.save_hyperparameters()
        self.loss_func = nn.L1Loss()
        self.loss_method = loss_method
        self.dice_loss_score = DiceScore(num_classes=2, include_background=False, input_format='index')
        self.ckpt_dir = ckpt_dir
        self.save_every_n_checkpoints = save_every_n_checkpoints
        self.no_lr_reduce = no_lr_reduce
        self.RayManager = RGBARayManager()

    def get_dice_score(self, representation_opacity, reconstruction_opacity):
        representation_opacity = (representation_opacity > 0.5).float()
        dice_score = 2 * torch.sum(representation_opacity * reconstruction_opacity, dim=[1, 2, 3]) / (torch.sum(representation_opacity, dim=[1, 2, 3]) + torch.sum(reconstruction_opacity, dim=[1, 2, 3]) + 1e-8)
        dice_score = torch.mean(dice_score)

        return dice_score


    def density_to_opacity(self, density, opacity_multiplier):
        density_grid = F.softplus(density - 10)
        return 1. - torch.exp(-density_grid * torch.mean(opacity_multiplier))

    def calculate_loss(self, batch, stage):
        grid, opacity_multiplier, blank_edge_rays_o, blank_edge_rays_d, rgb_rays_o, rgb_rays_d, rgb_rays_c = batch
        reconstruction = self.model(grid)

        density_reconstruction = reconstruction[:, :1]
        colour_reconstruction = reconstruction[:, 1:]

        l1_loss = self.loss_func(grid, reconstruction)

        self.log(stage + '_l1_loss', l1_loss)

        density_grid = grid[:, :1]
        colour_grid = grid[:, 1:]

        opacity_grid = self.density_to_opacity(density_grid, opacity_multiplier)
        opacity_mask = (opacity_grid > 0.3)
        expanded_opacity_mask = opacity_mask.expand(-1, 3, -1, -1, -1)


        reconstructed_opacity_grid = self.density_to_opacity(density_reconstruction, opacity_multiplier)




        edge_xyz_sampled, edge_z_vals, edge_ray_valid = self.RayManager.sample_ray(blank_edge_rays_o, blank_edge_rays_d)
        rgb_xyz_sampled, rgb_z_vals, rgb_ray_valid = self.RayManager.sample_ray(rgb_rays_o, rgb_rays_d)

        edge_opacity = torch.sum(
            self.RayManager.get_opacity_weight(edge_xyz_sampled, density_reconstruction, edge_ray_valid, edge_z_vals),
            dim=-1)
        rgb_weight = self.RayManager.get_opacity_weight(rgb_xyz_sampled, density_reconstruction, rgb_ray_valid,
                                                        rgb_z_vals)
        center_opacity = torch.sum(rgb_weight, dim=-1)

        ray_rgbs = self.RayManager.get_colour(rgb_xyz_sampled, colour_reconstruction, rgb_weight, False,
                                              give_raw_rgb=True)


        del edge_xyz_sampled, edge_ray_valid, edge_z_vals, edge_ray_valid, rgb_xyz_sampled, rgb_z_vals, rgb_weight, rgb_ray_valid

        density_loss = self.loss_func(density_grid, density_reconstruction)
        self.log(stage + '_density_loss', density_loss)

        opacity_loss = self.loss_func(opacity_grid, reconstructed_opacity_grid)
        self.log(stage + '_opacity_loss', opacity_loss)

        mask_colour_loss = self.loss_func(colour_grid[expanded_opacity_mask], colour_reconstruction[expanded_opacity_mask])
        self.log(stage + '_mask_colour_loss', mask_colour_loss)

        dice_loss = self.get_dice_score(opacity_grid, reconstructed_opacity_grid)
        del opacity_grid, reconstructed_opacity_grid, density_grid, colour_grid, density_reconstruction, colour_reconstruction, opacity_mask

        self.log(stage + '_dice_loss', dice_loss)
        dice_loss = (1 - dice_loss)





        edge_ray_loss = self.loss_func(edge_opacity, torch.zeros_like(edge_opacity))
        self.log(stage + '_edge_ray_loss', edge_ray_loss)

        center_ray_loss = self.loss_func(center_opacity, torch.ones_like(center_opacity))
        self.log(stage + '_center_ray_loss', center_ray_loss)

        center_ray_rgb_loss = self.loss_func(ray_rgbs, rgb_rays_c)
        self.log(stage + '_center_ray_rgb_loss', center_ray_rgb_loss)

        if self.loss_method == "L1":
            final_loss = l1_loss
        elif self.loss_method == "Dice":
            final_loss = dice_loss
        elif self.loss_method == "Density":
            final_loss = density_loss
        elif self.loss_method == "Opacity":
            final_loss = opacity_loss
        elif self.loss_method == "DO":
            final_loss = density_loss + opacity_loss
        elif self.loss_method == "O+RGB":
            final_loss = opacity_loss + mask_colour_loss
        elif self.loss_method == "O+RGB+Dice":
            final_loss = opacity_loss + mask_colour_loss + dice_loss
        elif self.loss_method == "DO+RGB":
            final_loss = density_loss + opacity_loss + mask_colour_loss
        elif self.loss_method == "DO+RGB+Dice":
            final_loss = density_loss + opacity_loss + mask_colour_loss + dice_loss
        elif self.loss_method == "D+RGB":
            final_loss = density_loss + mask_colour_loss
        elif self.loss_method == "DO+RGB+Ray":
            final_loss = density_loss + opacity_loss + mask_colour_loss
        elif self.loss_method == "DO+RGB+Ray+Dice":
            final_loss = density_loss + opacity_loss + mask_colour_loss + dice_loss

        else:
            raise ValueError(f"Unknown loss method: {self.loss_method}")

        del opacity_loss, dice_loss, mask_colour_loss, density_loss, l1_loss
        self.log(stage + '_final_loss', final_loss)

        return final_loss

    def training_step(self, batch, batch_idx):
        return self.calculate_loss(batch, stage='train')

    def test_step(self, batch, batch_idx):
        self.calculate_loss(batch, stage='test')

    def validation_step(self, batch, batch_idx):
        self.calculate_loss(batch, stage='val')
        if self.current_epoch % self.save_every_n_checkpoints == 0:
            self.trainer.save_checkpoint(os.path.join(self.ckpt_dir, f"epoch_{self.current_epoch}.ckpt"))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        if self.no_lr_reduce:
            return {"optimizer": optimizer}
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_final_loss"}



if __name__ == "__main__":


    datasets_path = data_dir = "~/masters/datasets/" if not args.low_acc else "~/Documents/masters/datasets/"

    dataset_loader = RepairDatasetLoader(batch_size=4 if args.no_logger else 10, dataset_type="RGBAGridDataset",
                                         representation_folder_name="RGBAGrids", num_workers=3, data_dir=datasets_path, overfit=args.overfit)
    L.seed_everything(42)
    run_name = f"loss={args.loss_method}_scale={args.scale}"


    if args.downsamples != 3:
        run_name += f"_downsamples={args.downsamples}"
    if args.lr != 1e-3:
        run_name += f"_lr={args.lr}"
    if args.overfit:
        run_name = "overfit_" + run_name
    if args.split_model:
        run_name = "split_model_" + run_name
    if args.no_batch_norm:
        run_name += "_no_batch_norm"
    if args.no_lr_reduce:
        run_name += "_no_lr_reduce"


    wandb_logger = False if args.no_logger else WandbLogger(name=run_name, project='OverfitRGBAGridReconstruction' if args.overfit else 'RGBAGridReconstruction')
    ckpt_dir = f"RGBAGridReconstructionCheckpoints/{run_name}/"

    save_every_n_checkpoints = 75 if args.overfit else 2
    ray_manager_dtype = torch.float16 if args.low_acc else torch.float32
    model = RGBAGridReconstruction(ckpt_dir=ckpt_dir, loss_method=args.loss_method, downsamples=args.downsamples, learning_rate=args.lr, scale=args.scale, no_batch_norm=args.no_batch_norm, save_every_n_checkpoints=save_every_n_checkpoints, split_model=args.split_model, no_lr_reduce=args.no_lr_reduce)

    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(dirpath=ckpt_dir, )
    epochs = 1000 if args.overfit else 200
    precision = "16-true" if args.low_acc else "32-true"
    lr_monitor = LearningRateMonitor(logging_interval='step')
    accelerator = "gpu"
    trainer = L.Trainer(max_epochs=epochs, accelerator=accelerator, callbacks=[] if args.no_logger else [checkpoint_callback, lr_monitor], precision=precision, logger=wandb_logger, num_sanity_val_steps=0, accumulate_grad_batches=6)
    trainer.fit(model, datamodule=dataset_loader)
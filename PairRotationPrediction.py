from DatasetLoading import RepairDatasetLoader
from RayRepresentationEncoding import RayRepresentationEncoder
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger
from scipy.spatial.transform import Rotation
import gc

class RotationPredictionNetwork(nn.Module):
    def __init__(self):
        super().__init__()


        representation_size = 128 * 2
        n_rays = 1000

        self.representation_encoder = RayRepresentationEncoder(transformer_layers=2, representation_size=representation_size//2, initial_dropout=0, n_rays=n_rays)

        transformer_layer = nn.TransformerEncoderLayer(d_model=representation_size, nhead=8, dropout=0, batch_first=True)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=2)


        self.head = nn.Sequential(nn.Linear(representation_size, 64),
                                  nn.BatchNorm1d(64),
                                  nn.ReLU(),
                                  nn.Linear(64, 4))

        self.pos_encoding = nn.Parameter(torch.randn(n_rays, representation_size), requires_grad=True)

        #self.rotation_token = nn.Parameter(torch.randn(1, representation_size), requires_grad=True)




    def forward(self, ray_colours_1, ray_locations_1, ray_colours_2, ray_locations_2):
        del ray_locations_1, ray_locations_2

        x = torch.cat((ray_colours_1, ray_colours_2), dim=1)
        batch_size = x.size(0)
        num_rays = ray_colours_1.size(1)
        del ray_colours_1, ray_colours_2


        x = x.view(batch_size * 2, num_rays, -1)


        x = self.representation_encoder(x)


        x = x.view(batch_size, num_rays * 2, -1)

        x = torch.cat([x[:, num_rays:, :], x[:, :num_rays, :]], dim=2)

        #print(x.shape, self.pos_encoding.shape)
        x = x  + self.pos_encoding.unsqueeze(0).repeat(batch_size, 1, 1)

        #x = torch.cat([x, self.rotation_token.unsqueeze(0).repeat(batch_size, 1, 1)], dim=1)
        x = self.transformer(x)


        #x = x[:, -1, :]
        x = torch.mean(x, dim=1)

        #x = x.view(batch_size, 2 * x.size(1))

        x = self.head(x)
        return torch.nn.functional.normalize(x)

class PairRotationPrediction(L.LightningModule):
    def __init__(self, test_addition = ""):
        super().__init__()
        self.model = RotationPredictionNetwork()
        self.lr = 1e-4

        self.test_addition = test_addition

    def calculate_loss(self, batch, batch_idx, stage):
        ray_colours_1, piece_rotation_1, ray_locations_1, ray_colours_2, piece_rotation_2, ray_locations_2, gt_rotation = batch
        predicted_rotation = self.model(ray_colours_1, ray_locations_1, ray_colours_2, ray_locations_2)

        loss = nn.functional.mse_loss(predicted_rotation, gt_rotation)
        self.log(self.test_addition + stage + '_loss', loss)


        cos_theta = torch.sum(predicted_rotation * gt_rotation, dim=-1)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        rot_error = torch.acos(cos_theta)
        rot_error = torch.rad2deg(rot_error)
        rot_rmse = torch.sqrt(rot_error.pow(2).mean())
        rot_upper_error = torch.quantile(rot_error, 0.95)
        # Calculate standard deviation of rotation error
        rot_error_std = torch.sqrt(rot_error.var())
        self.log(self.test_addition + stage + '_angular_error', rot_rmse)
        self.log(self.test_addition + stage + '_angular_error_95th_percentile', rot_upper_error)
        self.log(self.test_addition + stage + '_angular_error_std', rot_error_std)

        del ray_colours_1, ray_locations_1, ray_colours_2, ray_locations_2, gt_rotation, predicted_rotation, cos_theta, piece_rotation_1, piece_rotation_2
        return rot_rmse + loss * 3

    def training_step(self, batch, batch_idx):

        return self.calculate_loss(batch, batch_idx, stage='train')

    def test_step(self, batch, batch_idx):
        self.calculate_loss(batch, batch_idx, stage='test')

    def validation_step(self, batch, batch_idx):
        self.calculate_loss(batch, batch_idx, stage='val')

    def predict_step(self, batch, batch_idx):
        ray_colours_1, piece_rotation_1, ray_locations_1, ray_colours_2, piece_rotation_2, ray_locations_2, gt_rotation = batch
        predicted_rotation = self.model(ray_colours_1, ray_locations_1, ray_colours_2, ray_locations_2)
        return {
            "predicted_rotation": predicted_rotation.detach().cpu(),
            "ray_colours_1": ray_colours_1.detach().cpu(),
            "ray_locations_1": ray_locations_1.detach().cpu(),
            "ray_colours_2": ray_colours_2.detach().cpu(),
            "ray_locations_2": ray_locations_2.detach().cpu(),
            "gt_rotation": gt_rotation.detach().cpu()
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_angular_error"}


def get_model(test_addition = ""):
    L.seed_everything(42)
    return PairRotationPrediction(test_addition)

def get_dataset(rays_folder_name, batch_size=12, data_dir="~/Documents/masters/datasets/"):
    return RepairDatasetLoader(batch_size=batch_size, dataset_type="RotatedRayPairsDataset", rays_folder_name=rays_folder_name, data_dir=data_dir)


def train(model, dataset, accumulate_grad_batches = 5, wandb_logger=None):
    if wandb_logger is None:
        wandb_logger = WandbLogger()
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(dirpath='pieceRotationCheckpoints/')
    trainer = L.Trainer(max_epochs=20, logger=wandb_logger, accelerator='gpu', accumulate_grad_batches=accumulate_grad_batches,
                        callbacks=[checkpoint_callback])
    trainer.fit(model, dataset)
    return model, trainer

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    tests = ["RayPairs1k30D", "RayPairs1k60D", "SRayPairs"]
    logger = WandbLogger(project="RotationPrediction")
    for test in tests:
        print("Test " + test)

        model = get_model(test + "_")
        dataset = get_dataset(test, batch_size=20, data_dir="~/masters/datasets/")
        model, trainer = train(model, dataset, accumulate_grad_batches=3)
        del model, dataset, trainer
        torch.cuda.empty_cache()
        gc.collect()

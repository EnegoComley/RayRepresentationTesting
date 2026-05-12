import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to the trained GridReconstruction Model")
    args = parser.parse_args()

from DatasetLoading import RepairDatasetLoader
from RayRepresentationEncoding import RayRepresentationEncoder
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger
from GridReconstruction import GridReconstruction


class NormalPredictionNetwork(nn.Module):
    def __init__(self, conv_encoder_path):
        super().__init__()
        lightning_model = GridReconstruction.load_from_checkpoint(conv_encoder_path)
        self.conv_encoder = lightning_model.model
        representation_size = 256 if lightning_model.double_channels else 128

        self.representation_encoder = RayRepresentationEncoder(transformer_layers=1, representation_size=representation_size)


        self.head = nn.Sequential(nn.Linear(representation_size, 64),
                                  nn.BatchNorm1d(64),
                                  nn.ReLU(),
                                  nn.Linear(64, 6))


    def forward(self, representation):
        self.conv_encoder.eval()
        with torch.no_grad():
            representation = self.conv_encoder(representation)

        b, c, w, h, d = representation.shape
        representation = representation.view(b, c, -1).permute(0, 2, 1)

        x = self.representation_encoder(representation)

        x = torch.max(x, dim=1)
        x = self.head(x)

        return x

class RepairPatternNormalPrediction(L.LightningModule):
    def __init__(self, conv_encoder_path):
        super().__init__()
        self.model = NormalPredictionNetwork(conv_encoder_path)
        self.lr = 1e-3

    def get_normal_category(self, rotations):
        #This is a very badly written function
        swaps, flips = zip(*rotations)
        category = torch.zeros(len(rotations), 6)
        swaps = torch.tensor(swaps)

        y_flips = [flip[1] for flip in flips]
        for i, x in enumerate(y_flips):
            if x == 1:
                category[i, :3] = (swaps[i] == 1).float()
            else:
                category[i, -3:] = (swaps[i] == 1).float()


        return torch.argmax(category, dim=1)


    def calculate_loss(self, batch, stage):
        representations, rotations = batch
        predicted_normal = self.model(representations)

        true_normals = self.get_normals(rotations)

        loss = torch.nn.functional.cross_entropy(predicted_normal, true_normals)

        self.log(stage + '_loss', loss)

        # Calculate accuracy
        predicted_labels = torch.argmax(predicted_normal, dim=1)
        accuracy = (predicted_labels == true_normals).float().mean()
        self.log(stage + '_accuracy', accuracy)

        del representations, rotations, predicted_normal, true_normals, predicted_labels, accuracy
        return loss

    def training_step(self, batch, batch_idx):
        return self.calculate_loss(batch, stage='train')

    def test_step(self, batch, batch_idx):
        self.calculate_loss(batch, stage='test')

    def validation_step(self, batch, batch_idx):
        self.calculate_loss(batch, stage='val')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        ray_colours, piece_rotation, ray_locations = batch
        predicted_normals = self.model(ray_colours, ray_locations)
        return predicted_normals




if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    wandb_logger = WandbLogger(project='SimpleNormalGridPrediction')
    datasets_path = data_dir = "~/masters/datasets/"
    dataset_loader = RepairDatasetLoader(batch_size=2, dataset_type="SimpleRotatedFixedGridDataset",
                                         representation_folder_name="gridswithRepresentation",
                                         num_workers=2,
                                         data_dir=datasets_path)



    L.seed_everything(42)
    model = RepairPatternNormalPrediction(args.model_path + ".ckpt")
    
    epochs = 200 if args.overfit else 20
    trainer = L.Trainer(max_epochs=epochs, logger=wandb_logger, accelerator='gpu', accumulate_grad_batches=20)
    trainer.fit(model, datamodule=dataset_loader)
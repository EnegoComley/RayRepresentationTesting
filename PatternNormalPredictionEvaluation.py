import torch
from lightning.pytorch.loggers import WandbLogger

torch.set_float32_matmul_precision('medium')

from sympy import false

from DatasetLoading import RepairDatasetLoader
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from torchmetrics.segmentation import DiceScore
from pytorch_lightning.callbacks import LearningRateMonitor
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import save_image


import argparse

import lightning as L
import os

import json

torch.set_float32_matmul_precision('medium')

from EvaluationUtils import encoders
from EvaluationUtils import TransformerEncoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Normal Prediction Evaluation')
    parser.add_argument('--model', type=str, default="PTV3", help='The model to be evaluated')
    parser.add_argument('--ncc', action='store_true', help='Running on the NCC?')
    parser.add_argument("--no_logger", action='store_true', help="Disable logging to Weights and Biases")




    args = parser.parse_args()
    print(args)

class NormalPredictionNetwork(nn.Module):
    def __init__(self, encoder_model):
        super().__init__()


        representation_size = encoder_model.embedding_size

        self.encoder_model = encoder_model
        self.transformer = TransformerEncoder(transformer_layers=2, representation_size=representation_size)



        self.head = nn.Sequential(nn.Linear(512, 64),
                                  nn.BatchNorm1d(64),
                                  nn.ReLU(),
                                  nn.Linear(64, 3))


    def forward(self, batch):
        x = self.encoder_model(batch)

        x = self.transformer(x)

        x = torch.mean(x, dim=1)
        x = self.head(x)

        return x



class PatternNormalPrediction(L.LightningModule):
    def __init__(self, encoder_model):
        super().__init__()
        self.model = NormalPredictionNetwork(encoder_model)
        self.lr = 1e-4

    def get_normals(self, rotations):
        batch_size = rotations.shape[0]
        normals = torch.tensor([0, 1, 0], dtype=torch.float32).repeat(batch_size, 1).unsqueeze(2).to(rotations.get_device())
        return torch.matmul(rotations, normals).squeeze(2)

    def calculate_loss(self, batch, stage):
        rotation = batch[-1]
        predicted_normals = self.model(batch)
        true_normals = self.get_normals(rotation)
        loss = nn.functional.mse_loss(predicted_normals, true_normals)
        self.log(f'{stage}_loss', loss)

        # Calculate angular error
        predicted_normals = nn.functional.normalize(predicted_normals, dim=1)
        true_normals = nn.functional.normalize(true_normals, dim=1)
        cos_angles = torch.clamp(torch.sum(predicted_normals * true_normals, dim=1), -1.0, 1.0)
        angles = torch.acos(cos_angles)  # in radians
        angular_error = torch.mean(angles) * (180.0 / np.pi)  # convert to degrees
        self.log(f'{stage}_angular_error', angular_error)

        del rotation, predicted_normals, true_normals, cos_angles, angles, angular_error
        return loss

    def training_step(self, batch, batch_idx):
        return self.calculate_loss(batch, stage='train')

    def test_step(self, batch, batch_idx):
        self.calculate_loss(batch, stage='test')


    def validation_step(self, batch, batch_idx):
        self.calculate_loss(batch, stage='val')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {"optimizer": optimizer}





if __name__ == "__main__":


    datasets_path = data_dir = "~/masters/datasets/" if args.ncc else "~/Documents/masters/datasets/"

    encoder = encoders[args.model.split("_")[0]](args.model.split("_")[1]) if "_" in args.model else encoders[args.model]()

    dataset_loader = RepairDatasetLoader(batch_size=encoder.batch_size if args.no_logger else 8, dataset_type=encoder.dataloader["rotated"],
                                         representation_folder_name=encoder.representation_folder_name, num_workers=3, data_dir=datasets_path)
    L.seed_everything(42)

    run_name = f"{args.model}"


    wandb_logger = False if args.no_logger else WandbLogger(name=run_name, project='PatternNormalPredictionEvaluation')
    ckpt_dir = f"PatternNormalPredictionEvaluationCheckpoints/{run_name}/"
    test_output_dir = f"PatternNormalPredictionEvaluationResults/{run_name}/"

    model = PatternNormalPrediction(encoder_model=encoder)

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(dirpath=ckpt_dir, )
    epochs = 100
    precision = "32-true"#"16-true" if args.low_acc else "32-true"
    #lr_monitor = LearningRateMonitor(logging_interval='step')
    accelerator = "gpu"
    trainer = L.Trainer(max_epochs=epochs, accelerator=accelerator, callbacks=[] if args.no_logger else [checkpoint_callback], precision=precision, logger=wandb_logger, accumulate_grad_batches=encoder.accumulate_grad_batches)
    trainer.fit(model, datamodule=dataset_loader)

    results = trainer.test(model, datamodule=dataset_loader)

    with open(f"{test_output_dir}/test_results.json", "w") as f:
        json.dump(results, f, indent=4)







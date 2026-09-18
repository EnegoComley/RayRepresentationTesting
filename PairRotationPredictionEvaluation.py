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
    parser = argparse.ArgumentParser(description='Pair Rotation Prediction Evaluation')
    parser.add_argument('--model', type=str, default="PTV3", help='The model to be evaluated')
    parser.add_argument('--ncc', action='store_true', help='Running on the NCC?')
    parser.add_argument("--no_logger", action='store_true', help="Disable logging to Weights and Biases")
    parser.add_argument("--angle_divider", type=float, default=1, help="The divider for the random angle")




    args = parser.parse_args()
    print(args)

class PairRotationPredictionNetwork(nn.Module):
    def __init__(self, encoder_model):
        super().__init__()


        representation_size = encoder_model.embedding_size

        self.encoder_model = encoder_model
        self.transformer = TransformerEncoder(transformer_layers=2, representation_size=representation_size)
        self.combined_transformer = TransformerEncoder(transformer_layers=1, representation_size=512)

        self.pos_encoder = nn.Parameter(torch.randn(2, 1, 1, 512))


        self.head = nn.Sequential(nn.Linear(512, 64),
                                  nn.BatchNorm1d(64),
                                  nn.ReLU(),
                                  nn.Linear(64, 4))


    def forward(self, batch):
        x1, x2, _ = batch
        x1 = self.encoder_model(x1)
        x2 = self.encoder_model(x2)

        x1 = self.transformer(x1)
        x2 = self.transformer(x2)
        x = self.combined_transformer(torch.cat([x1 + self.pos_encoder[0].expand_as(x1), x2 + self.pos_encoder[1].expand_as(x2)], dim=1))

        x = torch.mean(x, dim=1)
        x = self.head(x)

        return x



class PairRotationPrediction(L.LightningModule):
    def __init__(self, encoder_model):
        super().__init__()
        self.model = PairRotationPredictionNetwork(encoder_model)
        self.lr = 1e-4

    def calculate_loss(self, batch, stage):
        gt_rotation = batch[-1]
        predicted_rotation = self.model(batch)
        predicted_rotation = nn.functional.normalize(predicted_rotation, dim=-1)

        loss = nn.functional.mse_loss(predicted_rotation, gt_rotation)
        self.log(stage + '_loss', loss)

        cos_theta = torch.sum(predicted_rotation * gt_rotation, dim=-1)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        rot_error = torch.acos(cos_theta)
        rot_error = torch.rad2deg(rot_error)
        rot_rmse = torch.sqrt(rot_error.pow(2).mean())
        rot_upper_error = torch.quantile(rot_error, 0.95)
        # Calculate standard deviation of rotation error
        rot_error_std = torch.sqrt(rot_error.var())
        self.log(stage + '_angular_error', rot_rmse)
        self.log(stage + '_angular_error_95th_percentile', rot_upper_error)
        self.log(stage + '_angular_error_std', rot_error_std)

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

    dataset_loader = RepairDatasetLoader(batch_size=encoder.batch_size, dataset_type=encoder.dataloader["dualRotated"],
                                         representation_folder_name=encoder.representation_folder_name, num_workers=3, data_dir=datasets_path, angle_divider=args.angle_divider)
    L.seed_everything(42)

    run_name = f"{args.model}"


    project_name = f"PairRotationPredictionEvaluation{90/args.angle_divider}" if args.angle_divider != 1 else "PairRotationPredictionEvaluation"
    wandb_logger = False if args.no_logger else WandbLogger(name=run_name, project=project_name)
    ckpt_dir = f"{project_name}Checkpoints/{run_name}/"
    test_output_dir = f"{project_name}Results/{run_name}/"

    model = PairRotationPrediction(encoder_model=encoder)

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







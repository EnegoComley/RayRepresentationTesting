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
    parser = argparse.ArgumentParser(description='Pos Encoding Evaluation')
    parser.add_argument('--model', type=str, default="PTV3", help='The model to be evaluated')
    parser.add_argument('--ncc', action='store_true', help='Running on the NCC?')
    parser.add_argument("--no_logger", action='store_true', help="Disable logging to Weights and Biases")




    args = parser.parse_args()
    print(args)

class PosEncodingEvaluationNetwork(nn.Module):
    def __init__(self, encoder_model):
        super().__init__()


        representation_size = encoder_model.embedding_size

        self.encoder_model = encoder_model
        self.transformer = TransformerEncoder(transformer_layers=2, representation_size=representation_size)



        self.head = nn.Sequential(nn.Linear(512, 350),
                                  nn.BatchNorm1d(350),
                                  nn.ReLU(),
                                  nn.Linear(350, representation_size))


    def forward(self, x):
        #with torch.autocast(device_type="cuda", dtype=torch.float16):

        x = self.transformer(x)

        x = torch.mean(x, dim=1)
        x = self.head(x)

        return x

    def encode(self, batch):
        x = self.encoder_model(batch)
        return x



class PosEncodingEvaluationPrediction(L.LightningModule):
    def __init__(self, encoder_model):
        super().__init__()
        self.model = PosEncodingEvaluationNetwork(encoder_model)
        self.lr = 1e-4


    def calculate_loss(self, batch, stage):
        encoding = self.model.encode(batch)
        prediction = self.model(encoding)
        b, n, embed_size = encoding.shape
        encoding = encoding.view(b, 12, 12, 12, embed_size)
        target = encoding[:, 6, 6, 6]

        loss = F.mse_loss(prediction, target)
        self.log(stage + '_loss', loss)

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
    dataset_loader = RepairDatasetLoader(batch_size=encoder.batch_size, dataset_type=encoder.dataloader["rotated"],
                                         representation_folder_name=encoder.representation_folder_name, num_workers=3, data_dir=datasets_path)
    L.seed_everything(42)

    run_name = f"{args.model}"


    wandb_logger = False if args.no_logger else WandbLogger(name=run_name, project='PosEncodingEvaluation')
    ckpt_dir = f"PosEncodingEvaluationCheckpoints/{run_name}/"
    test_output_dir = f"PosEncodingEvaluationResults/{run_name}/"

    model = PosEncodingEvaluationPrediction(encoder_model=encoder)

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







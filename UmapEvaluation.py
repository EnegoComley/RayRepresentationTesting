from xml.sax.handler import all_features

import torch
from lightning.pytorch.loggers import WandbLogger
from pyg_lib.ops import nearest

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

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


import argparse

import lightning as L
import os

import json

torch.set_float32_matmul_precision('medium')

from EvaluationUtils import encoders
from EvaluationUtils import TransformerEncoder

import umap

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Normal Prediction Evaluation')
    parser.add_argument('--model', type=str, default="PTV3", help='The model to be evaluated')
    parser.add_argument('--ncc', action='store_true', help='Running on the NCC?')
    parser.add_argument("--no_logger", action='store_true', help="Disable logging to Weights and Biases")




    args = parser.parse_args()
    print(args)


    datasets_path = data_dir = "~/masters/datasets/" if args.ncc else "~/Documents/masters/datasets/"

    encoder = encoders[args.model.split("_")[0]](args.model.split("_")[1]) if "_" in args.model else encoders[args.model]().to(device)
    dataset_loader = RepairDatasetLoader(batch_size=encoder.batch_size, dataset_type=encoder.dataloader["plain"],
                                         representation_folder_name=encoder.representation_folder_name, num_workers=3, data_dir=datasets_path)
    test_dataloader = dataset_loader.test_dataloader()
    L.seed_everything(42)

    run_name = f"{args.model}"


    test_output_dir = f"UMAPEvaluationResults/{run_name}/"

    all_features = None

    visualise = []
    test_vis_pieces = ["RPf_00347",
                       "RPf_00204",
                       'RPf_00892',
                       'RPf_00822',
                       'RPf_00586',
                       'RPf_00708',
                       'RPf_00059',
                       'RPf_00925',
                       'RPf_00017',
                       'RPf_00280',
                       'RPf_00030',
                       'RPf_00363']

    piece_names = []

    os.makedirs(test_output_dir, exist_ok=True)
    for batch in iter(test_dataloader):
        with torch.inference_mode():
            features = encoder(batch)
        if all_features is None:
            all_features = features.detach().cpu()
        else:
            all_features = torch.cat((all_features, features.detach().cpu()), dim=0)

        piece_names = list(batch[-1])
        vis_pieces = [x in piece_names for x in test_vis_pieces]
        visualise += vis_pieces

        piece_names += piece_names

    visualise = np.array(visualise)
    piece_names = np.array(piece_names)

    all_features = all_features.view(all_features.shape[0], -1).numpy()
    all_features =StandardScaler().fit_transform(all_features)

    reducer = umap.UMAP(random_state=42, n_components=20)
    reducer.fit(all_features)

    embedding = reducer.transform(all_features)

    nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(embedding)
    distances, indices = nbrs.kneighbors(embedding[visualise])

    nearest_neighbors = piece_names[indices]

    print(nearest_neighbors)












import argparse

from DatasetLoading import RepairDatasetLoader

import os

import numpy as np
import torch
from GridReconstruction import GridReconstructionNetwork
from GridReconstruction import GridReconstruction


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to the trained GridReconstruction Model")
    args = parser.parse_args()


    lightning_model = GridReconstruction.load_from_checkpoint(args.model_path)
    model = lightning_model.model
    model.eval()
    model.to(device)


    datasets_path = data_dir="~/masters/datasets/"
    dataset_loader = RepairDatasetLoader(batch_size=2, dataset_type="FixedGridDataset",
                                            representation_folder_name="gridswithRepresentation", num_workers=2, data_dir=datasets_path)
    out_folder_path =  os.path.join(os.path.expanduser(datasets_path), f"{args.model_path.split('/')[-1].split('.')[0]}Bottlenecks")
    os.makedirs(out_folder_path, exist_ok=True)


    datasets = [dataset_loader.train_dataloader().dataset, dataset_loader.val_dataloader().dataset]
    for dataset in datasets:
        for idx, representation in enumerate(dataset):
            with torch.no_grad():
                bottleneck = model.encoder(representation)
            bottleneck_np = bottleneck.detach().cpu().numpy()
            piece_name = dataset.piece_names[idx]
            output_file = os.path.join(out_folder_path, piece_name + ".npz")
            np.savez(output_file, bottleneck=bottleneck_np)
            del bottleneck, bottleneck_np, piece_name, output_file
            torch.cuda.empty_cache()

from DatasetLoading import RepairDatasetLoader

import lightning as L


import os

import numpy as np
import torch


torch.set_float32_matmul_precision('medium')


from RGBAGridReconstruction import RGBAGridReconstruction

import json

checkpoints_folder = "RGBAGridReconstructionCheckpoints"


dataset_loader = RepairDatasetLoader(batch_size=5, dataset_type="RGBAGridDataset",
                                         representation_folder_name="RGBAGrids", num_workers=3)

trainer = L.Trainer(max_epochs=1, accelerator="gpu")
model_folders = [x for x in os.listdir(checkpoints_folder) if os.path.isdir(os.path.join(checkpoints_folder, x))]
for model_folder_name in model_folders:
    model_folder_path = os.path.join(checkpoints_folder, model_folder_name)
    checkpoints = [x for x in os.listdir(model_folder_path) if os.path.isfile(os.path.join(model_folder_path, x))]
    renamed_checkpoints = [x.replace("=", ".") for x in checkpoints]
    renamed_checkpoints = [x.replace("_", ".") for x in renamed_checkpoints]
    renamed_checkpoints = [x.replace("-", ".") for x in renamed_checkpoints]
    epochs = np.array([int(x.split(".")[1]) for x in renamed_checkpoints])
    checkpoint_name = checkpoints[np.argmax(epochs)]
    model = RGBAGridReconstruction.load_from_checkpoint(os.path.join(model_folder_path, checkpoint_name), device=torch.device("cuda"))
    model.test_output_dir = f"RGBAGridReconstructionTestOutputs/{model_folder_name}/"
    results = trainer.test(model, datamodule=dataset_loader)

    with open(f"RGBAGridReconstructionTestOutputs/{model_folder_name}/test_results.json", "w") as f:
        json.dump(results, f, indent=4)

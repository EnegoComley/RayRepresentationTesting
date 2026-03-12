import os

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from scipy.spatial.transform import Rotation

class PuzzleDatasetLoader():
    def __init__(self):
        self.original_datadir = ""
        self.puzzle_test_split = []
        self.puzzle_train_split = []

    def get_dataloaders(self, dataset, batch_size, train_data, val_data, test_data, data_dir):
        return (DataLoader(dataset(train_data, data_dir), batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8),
                DataLoader(dataset(val_data, data_dir), batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8),
                DataLoader(dataset(test_data, data_dir), batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8))


class RepairDatasetLoader(PuzzleDatasetLoader):
    def __init__(self, rays_folder_name = "Rays"):
        super().__init__()
        self.original_datadir = os.path.expanduser("~/Documents/masters/datasets/RePAIR/")


        test_split_location = os.path.join(self.original_datadir, "test.txt")
        with open(test_split_location, 'r') as f:
            self.puzzle_test_split = [f.strip() for f in f.readlines()]
        train_split_location = os.path.join(self.original_datadir, "train.txt")
        with open(train_split_location, 'r') as f:
            self.puzzle_train_split = [f.strip() for f in f.readlines()]
        val_split_location = os.path.join(self.original_datadir, "val.txt")
        with open(val_split_location, 'r') as f:
            self.puzzle_val_split = [f.strip() for f in f.readlines()]

        self.original_datadir = os.path.join(self.original_datadir, "SOLVED")

        self.piece_test_split = []
        self.piece_train_split = []
        self.piece_val_split = []
        for puzzle in self.puzzle_test_split:
            puzzle_dir = os.path.join(self.original_datadir, puzzle)
            piece_filenames = sorted(os.listdir(puzzle_dir))
            self.piece_test_split.extend([piece.split(".")[0] for piece in piece_filenames if piece.endswith('.mtl')])

        for puzzle in self.puzzle_train_split:
            puzzle_dir = os.path.join(self.original_datadir, puzzle)
            piece_filenames = sorted(os.listdir(puzzle_dir))
            self.piece_train_split.extend([piece.split(".")[0] for piece in piece_filenames if piece.endswith('.mtl')])

        for puzzle in self.puzzle_val_split:
            puzzle_dir = os.path.join(self.original_datadir, puzzle)
            piece_filenames = sorted(os.listdir(puzzle_dir))
            self.piece_val_split.extend([piece.split(".")[0] for piece in piece_filenames if piece.endswith('.mtl')])


        self.rays_data_dir = os.path.expanduser(f"~/Documents/masters/datasets/{rays_folder_name}/")

        ray_files = os.listdir(self.rays_data_dir)
        self.test_ray_pieces =  [f for x in self.piece_test_split for f in ray_files if x in f]
        self.train_ray_pieces = [f for x in self.piece_train_split for f in ray_files if x in f]
        self.val_ray_pieces = [f for x in self.piece_val_split for f in ray_files if x in f]


    def get_ray_dataloaders(self, batch_size):
        return self.get_dataloaders(RayDataset, batch_size, self.train_ray_pieces, self.val_ray_pieces, self.test_ray_pieces, self.rays_data_dir)

    def get_ray_pair_dataloaders(self, batch_size):
        return self.get_dataloaders(RotatedRayPairsDataset, batch_size, self.train_ray_pieces, self.val_ray_pieces, self.test_ray_pieces, self.rays_data_dir)

    def get_simple_ray_pair_dataloaders(self, batch_size):
        return self.get_dataloaders(SimpleRotatedRayPairsDataset, batch_size, self.train_ray_pieces, self.val_ray_pieces, self.test_ray_pieces, self.rays_data_dir)

    def setup(self, stage: str):
        return

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size)


# Used to clean-up when the run is finished



class RayDataset(Dataset):
    def __init__(self, ray_piece_names, rays_data_dir):
        self.ray_piece_names = ray_piece_names
        self.rays_data_dir = rays_data_dir

    def __len__(self):
        return len(self.ray_piece_names)

    def __getitem__(self, idx):
        ray_piece_name = self.ray_piece_names[idx]
        ray_colours, piece_rotation, ray_locations = self.get_rays_from_file(ray_piece_name)

        return ray_colours, piece_rotation, ray_locations

    def get_rays_from_file(self, ray_piece_name):
        path = os.path.join(self.rays_data_dir, ray_piece_name) + "/rays_archive.npz"
        rays_data = np.load(path)
        #ids = [x * 2 for x in range(2500)]
        ray_colours = torch.from_numpy(rays_data['ray_colours'])#[ids, :]
        piece_rotation = torch.from_numpy(rays_data['piece_rotation'])
        ray_locations = torch.from_numpy(rays_data['rays'])#[ids, :]
        del rays_data
        return ray_colours, piece_rotation, ray_locations

class RotatedRayPairsDataset(RayDataset):
    def __init__(self, ray_piece_names, rays_data_dir):
        super().__init__(ray_piece_names, rays_data_dir)
        piece_numbers = list(set([name.split("_")[1] for name in ray_piece_names]))
        grouped_pieces = [[x for x in ray_piece_names if n in x] for n in piece_numbers]
        self.ray_pair_names = [pair for group in grouped_pieces for pair in zip(group[::2], group[1::2])]

    def __len__(self):
        return len(self.ray_pair_names)
        #return len(self.ray_piece_names) ** 2

    def __getitem__(self, idx):
        ray_piece_name_1, ray_piece_name_2 = self.ray_pair_names[idx]
        #ray_piece_name_1 = self.ray_piece_names[idx%len(self.ray_piece_names)]
        #ray_piece_name_2 = self.ray_piece_names[idx//len(self.ray_piece_names)]
        ray_colours_1, piece_rotation_1, ray_locations_1 = self.get_rays_from_file(ray_piece_name_1)
        ray_colours_2, piece_rotation_2, ray_locations_2 = self.get_rays_from_file(ray_piece_name_2)

        piece_rotation_1 = piece_rotation_1.unsqueeze(0)
        piece_rotation_2 = piece_rotation_2.unsqueeze(0)

        relative_rotation = torch.matmul(piece_rotation_2, torch.linalg.inv(piece_rotation_1)).squeeze(0)
        relative_rotation = torch.from_numpy(Rotation.from_matrix(relative_rotation.numpy()).as_quat()).float()

        return (ray_colours_1, ray_locations_1,
                ray_colours_2, ray_locations_2, relative_rotation)


class SimpleRotatedRayPairsDataset(RotatedRayPairsDataset):
    def get_rays_from_file(self, ray_piece_name):
        path = os.path.join(self.rays_data_dir, ray_piece_name) + "/rays_archive.npz"
        rays_data = np.load(path)
        #ids = [x * 2 for x in range(2500)]
        ray_colours = torch.from_numpy(rays_data['ray_colours'])#[ids, :]
        piece_rotation = torch.from_numpy(rays_data['piece_rotation'])
        ray_locations = torch.from_numpy(rays_data['rays'])#[ids, :]
        rotation_num = torch.tensor(rays_data['rotation_num'])
        del rays_data
        return ray_colours, piece_rotation, ray_locations, rotation_num

    def __getitem__(self, idx):
        ray_piece_name_1, ray_piece_name_2 = self.ray_pair_names[idx]
        ray_colours_1, piece_rotation_1, ray_locations_1, rot_num_1 = self.get_rays_from_file(ray_piece_name_1)
        ray_colours_2, piece_rotation_2, ray_locations_2, rot_num_2 = self.get_rays_from_file(ray_piece_name_2)
        rot_num = (rot_num_2 - rot_num_1) % 4

        piece_rotation_1 = piece_rotation_1.unsqueeze(0)
        piece_rotation_2 = piece_rotation_2.unsqueeze(0)

        relative_rotation = torch.matmul(piece_rotation_2, torch.linalg.inv(piece_rotation_1)).squeeze(0)
        relative_rotation = torch.from_numpy(Rotation.from_matrix(relative_rotation.numpy()).as_quat()).float()

        return (ray_colours_1, ray_locations_1,
                ray_colours_2, ray_locations_2, relative_rotation, rot_num)
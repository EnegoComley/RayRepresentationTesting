import os

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from scipy.spatial.transform import Rotation
import lightning as L

class PuzzleDatasetLoader(L.LightningDataModule):
    def __init__(self, batch_size, dataset_type):
        super().__init__()
        self.original_datadir = ""
        self.puzzle_test_split = []
        self.puzzle_train_split = []
        self.dataset_info_dict = {}
        self.batch_size = batch_size
        self.dataset = eval(dataset_type)

    def get_dataloader(self, split_dict):
        return DataLoader(self.dataset(split_dict, self.dataset_info_dict), batch_size=self.batch_size, shuffle=True, drop_last=False, num_workers=14)



class RepairDatasetLoader(PuzzleDatasetLoader):
    def __init__(self, batch_size, dataset_type, data_dir="~/Documents/masters/datasets/", rays_folder_name = "Rays5K", fracture_rays_folder_name = "FractureRays5K"):
        super().__init__(batch_size=batch_size, dataset_type=dataset_type)
        self.original_datadir = os.path.expanduser(data_dir + "RePAIR/")


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

        self.test_ray_pieces = []
        self.train_ray_pieces = []
        self.val_ray_pieces = []

        self.puzzle_to_pieces = {}

        self.rays_data_dir = os.path.expanduser(data_dir + rays_folder_name)
        ray_files = os.listdir(self.rays_data_dir)

        self.piece_to_rotated_pieces = {}

        for (puzzle_split, piece_split, ray_pieces) in zip([self.puzzle_test_split, self.puzzle_train_split, self.puzzle_val_split],
                                                           [self.piece_test_split, self.piece_train_split, self.piece_val_split],
                                                           [self.test_ray_pieces, self.train_ray_pieces, self.val_ray_pieces]):
            for puzzle in puzzle_split:
                puzzle_dir = os.path.join(self.original_datadir, puzzle)
                piece_filenames = sorted(os.listdir(puzzle_dir))
                pieces = [piece.split(".")[0] for piece in piece_filenames if piece.endswith('.mtl')]
                piece_split.extend(pieces)
                self.puzzle_to_pieces.update({puzzle: pieces})

            for x in piece_split:
                self.piece_to_rotated_pieces.update({x: [f for f in ray_files if x in f]})
            ray_pieces.extend([f for x in piece_split for f in ray_files if x in f])

        self.pieces_to_puzzles = {piece: puzzle for puzzle, pieces in self.puzzle_to_pieces.items() for piece in pieces}


        self.test_split_dict = {"puzzles": self.puzzle_test_split, "ray_pieces": self.test_ray_pieces}
        self.train_split_dict = {"puzzles": self.puzzle_train_split, "ray_pieces": self.train_ray_pieces}
        self.val_split_dict = {"puzzles": self.puzzle_val_split, "ray_pieces": self.val_ray_pieces}
        self.dataset_info_dict = {
            "puzzle_to_pieces": self.puzzle_to_pieces,
            "data_dir": self.rays_data_dir,
            "fracture_rays_dir": os.path.expanduser(data_dir + fracture_rays_folder_name),
            "piece_to_rotated_pieces": self.piece_to_rotated_pieces,
            "pieces_to_puzzles": self.pieces_to_puzzles
                                  }

    def setup(self, stage: str):
        return

    def train_dataloader(self):
        return self.get_dataloader(self.train_split_dict)

    def val_dataloader(self):
        return self.get_dataloader(self.val_split_dict)

    def test_dataloader(self):
        return self.get_dataloader(self.test_split_dict)

    def predict_dataloader(self):
        return self.get_dataloader(self.test_split_dict)


# Used to clean-up when the run is finished



class RayDataset(Dataset):
    def __init__(self, split_dict, dataset_info_dict):
        self.ray_piece_names = split_dict["ray_pieces"]
        self.rays_data_dir = dataset_info_dict["data_dir"]

    def __len__(self):
        return len(self.ray_piece_names)

    def __getitem__(self, idx):
        ray_piece_name = self.ray_piece_names[idx]
        ray_colours, piece_rotation, ray_locations = self.get_rays_from_file(ray_piece_name)

        return ray_colours, piece_rotation, ray_locations

    def get_rays_from_file(self, ray_piece_name):
        path = os.path.join(self.rays_data_dir, ray_piece_name) + "/rays_archive.npz"
        rays_data = np.load(path)
        #ids = [x * 5 for x in range(1000)]
        ray_colours = torch.from_numpy(rays_data['ray_colours'])#[ids, :]
        piece_rotation = torch.from_numpy(rays_data['piece_rotation'])
        ray_locations = torch.from_numpy(rays_data['rays'])#[ids, :]
        del rays_data
        return ray_colours, piece_rotation, ray_locations

class RotatedRayPairsDataset(RayDataset):
    def __init__(self, split_dict, dataset_info_dict):
        super().__init__(split_dict, dataset_info_dict)
        #piece_numbers = list(set([name.split("_")[1] for name in self.ray_piece_names]))
        #grouped_pieces = [[x for x in self.ray_piece_names if n in x] for n in piece_numbers]
        #self.ray_pair_names = [pair for group in grouped_pieces for pair in zip(group[::2], group[1::2])]

    #def __len__(self):
    #    return len(self.ray_pair_names)
        #return len(self.ray_piece_names) ** 2

    def get_rays_from_file(self, ray_piece_name):
        path = os.path.join(self.rays_data_dir, ray_piece_name) + "/rays_archive.npz"
        rays_data = np.load(path)
        #ids = [x * 5 for x in range(1000)]
        ray_colours = torch.from_numpy(rays_data['ray_colours']).float()#[ids, :]
        piece_rotation = torch.from_numpy(rays_data['piece_rotation']).float()
        ray_locations = torch.from_numpy(rays_data['rays']).float()#[ids, :]
        ray_colours_b = torch.from_numpy(rays_data['ray_colours_b']).float()#[ids, :]
        piece_rotation_b = torch.from_numpy(rays_data['piece_rotation_b']).float()
        ray_locations_b = torch.from_numpy(rays_data['rays_b']).float()#[ids, :]rotation_transform
        rotation_transform = torch.from_numpy(rays_data['rotation_transform']).float()  # [ids, :]

        del rays_data
        return ray_colours, piece_rotation, ray_locations, ray_colours_b, piece_rotation_b, ray_locations_b, rotation_transform

    def __getitem__(self, idx):

        #ray_piece_name_1, ray_piece_name_2 = self.ray_pair_names[idx]
        #ray_piece_name_1 = self.ray_piece_names[idx%len(self.ray_piece_names)]
        #ray_piece_name_2 = self.ray_piece_names[idx//len(self.ray_piece_names)]

        piece_name = self.ray_piece_names[idx]
        ray_colours, piece_rotation, ray_locations, ray_colours_b, piece_rotation_b, ray_locations_b, rotation_transform = self.get_rays_from_file(piece_name)
        #ray_colours_1, piece_rotation_1, ray_locations_1 = self.get_rays_from_file(ray_piece_name_1)
        #ray_colours_2, piece_rotation_2, ray_locations_2 = self.get_rays_from_file(ray_piece_name_2)

        #piece_rotation_1 = piece_rotation_1.unsqueeze(0)
        #piece_rotation_2 = piece_rotation_2.unsqueeze(0)

        #relative_rotation = torch.matmul(piece_rotation_2, torch.linalg.inv(piece_rotation_1)).squeeze(0)
        #relative_rotation = torch.from_numpy(Rotation.from_matrix(relative_rotation.numpy()).as_quat()).float()

        return ray_colours, piece_rotation, ray_locations, ray_colours_b, piece_rotation_b, ray_locations_b, rotation_transform


class SimpleRotatedRayPairsDataset():  # RotatedRayPairsDataset):
    def get_rays_from_file(self, ray_piece_name):
        path = os.path.join(self.rays_data_dir, ray_piece_name) + "/rays_archive.npz"
        rays_data = np.load(path)
        # ids = [x * 2 for x in range(2500)]
        ray_colours = torch.from_numpy(rays_data['ray_colours'])  # [ids, :]
        piece_rotation = torch.from_numpy(rays_data['piece_rotation'])
        ray_locations = torch.from_numpy(rays_data['rays'])  # [ids, :]
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

class GarfCompatibleRayDataset(RayDataset):
    def __init__(self, split_dict, dataset_info_dict):
        super().__init__(split_dict, dataset_info_dict)
        self.puzzle_split = split_dict["puzzles"]
        self.puzzle_to_pieces = dataset_info_dict["puzzle_to_pieces"]
        self.fracture_rays_dir = dataset_info_dict["fracture_rays_dir"]
        self.piece_to_rotated_pieces = dataset_info_dict["piece_to_rotated_pieces"]
        self.num_rotations = min([len(v) for v in self.piece_to_rotated_pieces.values()])

    def __len__(self):
        return len(self.puzzle_split) * self.num_rotations

    def __getitem__(self, idx):
        puzzle_name = self.puzzle_split[idx//self.num_rotations]

        piece_names = self.puzzle_to_pieces[puzzle_name]
        piece_names = [self.piece_to_rotated_pieces[x][idx%self.num_rotations] for x in piece_names]

        ray_colours_list = []
        ray_locations_list = []
        fracture_rays_list = []
        for piece_name in piece_names:
            ray_piece_name = [f for f in self.ray_piece_names if piece_name in f][0]
            ray_colours, piece_rotation, ray_locations = self.get_rays_from_file(ray_piece_name)
            piece_rotation = piece_rotation[[3, 0, 1, 2]]
            ray_colours_list.append(ray_colours)
            ray_locations_list.append(ray_locations)

            fracture_rays = self.load_fracture_rays(piece_name)
            fracture_rays_list.append(fracture_rays)

        ray_colours = torch.cat(ray_colours_list, dim=0)
        ray_locations = torch.cat(ray_locations_list, dim=0)
        fracture_rays = torch.cat(fracture_rays_list, dim=0)


        data = {
            "index": idx,
            "name": puzzle_name,
            "num_parts" : len(piece_names),
            "ray_features": ray_colours,
            "ray_locations": ray_locations,
            "fracture_surface_gt" : fracture_rays,
            "pieces": ",".join(piece_names),
            "quaternions" : piece_rotation

        }

    def load_fracture_rays(self, piece):
        path = os.path.join(self.fracture_rays_dir, piece + ".Bin")
        with open(path, 'rb') as f:
            n_bytes = f.read(4)
            if len(n_bytes) < 4:
                raise EOFError("File too small to contain header")
            n = int.from_bytes(n_bytes, byteorder='little', signed=True)
            if n < 0:
                raise ValueError("Invalid length in header")
            raw = f.read(n)
            if len(raw) < n:
                raise EOFError("File truncated")
            arr = np.frombuffer(raw, dtype=np.uint8)
            return arr.astype(bool)


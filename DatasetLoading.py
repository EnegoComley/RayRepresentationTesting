import os

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from scipy.spatial.transform import Rotation
import lightning as L
from typing import Iterable, List, Tuple, Union

from NerfRepresentationUtils import ColourPredictionPredictionNetwork

import itertools


class PuzzleDatasetLoader(L.LightningDataModule):
    def __init__(self, batch_size, dataset_type, num_workers=14):
        super().__init__()
        self.original_datadir = ""
        self.puzzle_test_split = []
        self.puzzle_train_split = []
        self.dataset_info_dict = {}
        self.batch_size = batch_size
        self.dataset = eval(dataset_type)
        self.train_split_dict = []
        self.val_split_dict = []
        self.test_split_dict = []
        self.num_workers = num_workers

    def get_dataloader(self, split_dict):
        return DataLoader(self.dataset(split_dict, self.dataset_info_dict), batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers)

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

class RepairDatasetLoader(PuzzleDatasetLoader):
    def __init__(self, batch_size, dataset_type, data_dir="~/Documents/masters/datasets/", representation_folder_name = "Rays5K", fracture_representation_folder_name = "FractureRays5K", num_workers=14, overfit=False, **kwargs):
        super().__init__(batch_size=batch_size, dataset_type=dataset_type, num_workers=num_workers)
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

        self.test_pieces = []
        self.train_pieces = []
        self.val_pieces = []

        self.puzzle_to_pieces = {}

        self.representation_data_dir = os.path.expanduser(data_dir + representation_folder_name)
        representation_files = os.listdir(self.representation_data_dir)


        for (puzzle_split, piece_split, pieces) in zip([self.puzzle_test_split, self.puzzle_train_split, self.puzzle_val_split],
                                                           [self.piece_test_split, self.piece_train_split, self.piece_val_split],
                                                           [self.test_pieces, self.train_pieces, self.val_pieces]):
            for puzzle in puzzle_split:
                puzzle_dir = os.path.join(self.original_datadir, puzzle)
                piece_filenames = sorted(os.listdir(puzzle_dir))
                pieces_names = [piece.split(".")[0] for piece in piece_filenames if piece.endswith('.mtl')]
                piece_split.extend(pieces_names)
                self.puzzle_to_pieces.update({puzzle: pieces_names})


            pieces.extend([f for x in piece_split for f in representation_files if x in f])

        self.pieces_to_puzzles = {piece: puzzle for puzzle, pieces in self.puzzle_to_pieces.items() for piece in pieces}

        if overfit:
            self.puzzle_test_split = self.puzzle_test_split[:10]
            self.puzzle_train_split = self.puzzle_train_split[:10]
            self.puzzle_val_split = self.puzzle_val_split[:10]

            self.test_pieces = self.test_pieces[:10]
            self.train_pieces = self.train_pieces[:10]
            self.val_pieces = self.val_pieces[:10]

        self.test_split_dict = {"puzzles": self.puzzle_test_split, "pieces": self.test_pieces}
        self.train_split_dict = {"puzzles": self.puzzle_train_split, "pieces": self.train_pieces}
        self.val_split_dict = {"puzzles": self.puzzle_val_split, "pieces": self.val_pieces}

        self.latent2representation = ColourPredictionPredictionNetwork(latent_size=31)
        self.latent2representation.load_state_dict(torch.load("TensoRFColourPrediction.pth"))
        self.latent2representation.eval()

        self.dataset_info_dict = {
            "puzzle_to_pieces": self.puzzle_to_pieces,
            "data_dir": self.representation_data_dir,
            "pieces_to_puzzles": self.pieces_to_puzzles,
            "latent2representation": self.latent2representation,
            "kwargs": kwargs
                                  }

class RayRepairDatasetDataloader(RepairDatasetLoader):
    def __init__(self, batch_size, dataset_type, data_dir="~/Documents/masters/datasets/", representation_folder_name="Rays5K",
                 fracture_representation_folder_name="FractureRays5K"):
        super().__init__(batch_size=batch_size, dataset_type=dataset_type, data_dir=data_dir, representation_folder_name=representation_folder_name,
                         fracture_representation_folder_name=fracture_representation_folder_name)
        self.dataset_info_dict["fracture_rays_dir"] = os.path.expanduser(data_dir + fracture_representation_folder_name)

        piece_to_rotated_pieces = {}
        representation_files = os.listdir(self.representation_data_dir)
        for (puzzle_split, piece_split, pieces) in zip(
                [self.puzzle_test_split, self.puzzle_train_split, self.puzzle_val_split],
                [self.piece_test_split, self.piece_train_split, self.piece_val_split],
                [self.test_pieces, self.train_pieces, self.val_pieces]):

            for x in piece_split:
                piece_to_rotated_pieces.update({x: [f for f in representation_files if x in f]})
        self.dataset_info_dict["piece_to_rotated_pieces"] = piece_to_rotated_pieces


class GridDataset(Dataset):
    def __init__(self, split_dict, dataset_info_dict):
        self.piece_names = split_dict["pieces"]
        self.representation_data_dir = dataset_info_dict["data_dir"]
        self.latent2representation = dataset_info_dict["latent2representation"]

    def __len__(self):
        return len(self.piece_names)

    def __getitem__(self, idx):
        raise NotImplementedError()

    def rotate_grid(self, idx_tensors, representation, alphas, rotation):
        return idx_tensors, representation, alphas

    def load_grid_representation(self, file_path: str, rotation=None, load_colours_and_dirs = False, device: Union[str, torch.device] = 'cpu'):
        """
        Load the file at `file_path` and return (indices_list, values_tensor, colour_tensor).

        - indices_list: list of 3 torch tensors, each shape (N,), dtype torch.uint8, on `device`
        - values_tensor: torch tensor shape (N, 28), dtype torch.float32, on `device`
        - colour_tensor: torch tensor shape (N, 3), dtype torch.float32, on `device`
        """
        full_path = os.path.join(self.representation_data_dir, file_path)
        data = np.load(full_path)

        idx_np = data['indices']
        #vals_np = data['values']
        alphas_np = data['alphas']
        representation_np = data['representation']


        if idx_np.ndim != 2 or idx_np.shape[0] != 3:
            raise ValueError("Saved 'indices' must have shape (3, N)")


        dev = torch.device(device)

        # Convert to torch tensors and move to device
        idx_tensors = [torch.from_numpy(idx_np[i]).to(device=dev, dtype=torch.int).flatten() for i in range(3)]
        representation_tensor = torch.from_numpy(representation_np).to(device=dev, dtype=torch.float32)
        alphas_tensor = torch.from_numpy(alphas_np).to(device=dev, dtype=torch.float32)

        # Rotate indices if needed
        idx_tensors, representation_tensor, alphas_tensor = self.rotate_grid(idx_tensors, representation_tensor, alphas_tensor, rotation)

        if load_colours_and_dirs:
            colour_np = data['colour']
            directions_np = data['directions']

            if colour_np.ndim != 2 or colour_np.shape[1] != 3 or colour_np.shape[0] != idx_np.shape[1]:
                raise ValueError("Saved 'colour' must have shape (N, 3) matching indices length")

            colour_tensor = torch.from_numpy(colour_np).to(device=dev, dtype=torch.float32)
            directions = torch.from_numpy(directions_np).to(device=dev, dtype=torch.float32)

            return idx_tensors, representation_tensor, alphas_tensor, colour_tensor, directions

        return idx_tensors, representation_tensor, alphas_tensor

class RawGridDataset(GridDataset):
    def load_raw_representation(self, path: str, device: Union[str, torch.device] = 'cpu') -> Tuple[
        List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load the file at `path` and return (indices_list, values_tensor, colour_tensor).

        - indices_list: list of 3 torch tensors, each shape (N,), dtype torch.uint8, on `device`
        - values_tensor: torch tensor shape (N, 28), dtype torch.float32, on `device`
        - colour_tensor: torch tensor shape (N, 3), dtype torch.float32, on `device`
        """
        full_path = os.path.join(self.representation_data_dir, path)
        data = np.load(full_path)
        if 'indices' not in data or 'values' not in data or 'colour' not in data:
            raise ValueError("File must contain 'indices', 'values' and 'colour' arrays")

        idx_np = data['indices']
        vals_np = data['values']
        colour_np = data['colour']
        directions_np = data['directions']

        if idx_np.ndim != 2 or idx_np.shape[0] != 3:
            raise ValueError("Saved 'indices' must have shape (3, N)")
        if vals_np.ndim != 2 or vals_np.shape[1] != 28 or vals_np.shape[0] != idx_np.shape[1]:
            raise ValueError("Saved 'values' must have shape (N, 28) matching indices length")
        if colour_np.ndim != 2 or colour_np.shape[1] != 3 or colour_np.shape[0] != idx_np.shape[1]:
            raise ValueError("Saved 'colour' must have shape (N, 3) matching indices length")

        dev = torch.device(device)

        # Convert to torch tensors and move to device
        idx_tensors = [torch.from_numpy(idx_np[i]).to(device=dev, dtype=torch.int).flatten() for i in range(3)]
        vals_tensor = torch.from_numpy(vals_np).to(device=dev, dtype=torch.float32)
        colour_tensor = torch.from_numpy(colour_np).to(device=dev, dtype=torch.float32)
        directions = torch.from_numpy(directions_np).to(device=dev, dtype=torch.float32)

        return idx_tensors, vals_tensor, colour_tensor, directions

    def __getitem__(self, idx):
        piece_name = self.piece_names[idx]
        idx_tensors, vals_tensor, colour_tensor, directions = self.load_raw_representation(piece_name)

        ids = torch.randperm(directions.shape[0])[:20]

        return directions[ids], vals_tensor[ids, 1:], colour_tensor[ids], vals_tensor[ids, 0]

#class LatentGridTestingDataset(GridDataset):
#    def __getitem__(self, idx):
#        piece_name = self.piece_names[idx]
#        idx_tensors, vals_tensor, colour_tensor, directions = self.load_grid_representation(piece_name, load_colours_and_dirs=True)
#
#        ids = torch.randperm(directions.shape[0])[:20]
#
#        return directions[ids], vals_tensor[ids, 1:], colour_tensor[ids], vals_tensor[ids, 0]

class FixedGridDataset(GridDataset):
    def create_grid_representation(self, idx_tensors, vals_tensor, alphas):
        representation_values = torch.cat([vals_tensor, alphas], dim=1)

        representation = torch.zeros([200, 200, 200, representation_values.shape[1]], dtype=torch.float32)
        representation[idx_tensors[0], idx_tensors[1], idx_tensors[2]] = representation_values
        representation = torch.permute(representation, (3, 0, 1, 2))

        return representation

    def __getitem__(self, idx, rotation=None):
        piece_name = self.piece_names[idx]
        idx_tensors, vals_tensor, alphas = self.load_grid_representation(piece_name, rotation=rotation)

        representation = self.create_grid_representation(idx_tensors, vals_tensor, alphas)

        del idx_tensors, vals_tensor, alphas
        return representation


    def get_testing_items(self, idx, rotation=None):
        piece_name = self.piece_names[idx]
        idx_tensors, representation_tensor, alphas_tensor, colour_tensor, directions = self.load_grid_representation(piece_name, load_colours_and_dirs=True, rotation=rotation)

        representation = self.create_grid_representation(idx_tensors, representation_tensor, alphas_tensor)

        return representation, idx_tensors, colour_tensor, directions
        


class SimpleRotatedFixedGridDataset(FixedGridDataset):
    def __init__(self, split_dict, dataset_info_dict):
        super().__init__(split_dict, dataset_info_dict)
        axis_swaps = list(itertools.permutations([0, 1, 2]))
        #axis_flips = [(1, 1, 1)] + list(set(itertools.permutations([1, 1, -1])))
        axis_flips = list(itertools.product([ -1, 1], repeat=3))
        pairs = [list(p) for p in itertools.product(range(len(axis_swaps)), range(len(axis_flips)))]
        self.rotations = [(np.array(axis_swaps[a]), np.array(axis_flips[b])) for a, b in pairs]
        self.normal_categories = self.get_normal_category(self.rotations)

    def __len__(self):
        return len(self.piece_names)

    def __getitem__(self, idx, rotation=None):
        if rotation is None:
            rotation_num = torch.randint(low=0, high=len(self.rotations), size=(1,)).item()
            rotation = self.rotations[rotation_num]
            normal_cat = self.normal_categories[rotation_num]
        else:
            normal_cat = self.get_normal_category([rotation])[0]

        representation = super().__getitem__(idx, rotation=rotation)
        return representation, rotation, normal_cat

    def rotate_grid(self, idx_tensors, representation, alphas, rotation):
        axis_swaps, axis_flips = rotation

        for i in range(3):
            idx_tensors[i] = 199 - idx_tensors[i] if axis_flips[i] == -1 else idx_tensors[i]
        idx_tensors = [idx_tensors[i] for i in axis_swaps]

        return idx_tensors, representation, alphas

    def get_testing_items(self, idx, rotation=((0,1,2), (1,1,1))):
        return super().get_testing_items(idx, rotation=rotation)

    def get_normal_category(self, rotations):
        # This is a very badly written function
        try:
            swaps, flips = zip(*rotations)
        except ValueError as e:
            swaps, flips = rotations
            swaps = swaps.unsqueeze(0)
            flips = flips.unsqueeze(0)
        category = torch.zeros(len(rotations), 6)

        swaps = torch.tensor(swaps)
        flips = torch.tensor(flips)
        # if type(swaps) != torch.Tensor:
        #    swaps = torch.tensor(swaps)

        y_flips = [flip[1] for flip in flips]
        for i, x in enumerate(y_flips):
            if x == 1:
                category[i, :3] = (swaps[i] == 1).float()
            else:
                category[i, -3:] = (swaps[i] == 1).float()

        return torch.argmax(category, dim=1)





class RayDataset(Dataset):
    def __init__(self, split_dict, dataset_info_dict):
        self.piece_names = split_dict["pieces"]
        self.representation_data_dir = dataset_info_dict["data_dir"]

    def __len__(self):
        return len(self.piece_names)

    def __getitem__(self, idx):
        piece_name = self.piece_names[idx]
        ray_colours, piece_rotation, ray_locations = self.get_rays_from_file(piece_name)

        return ray_colours, piece_rotation, ray_locations

    def get_rays_from_file(self, piece_name):
        path = os.path.join(self.representation_data_dir, piece_name) + "/rays_archive.npz"
        rays_data = np.load(path)
        #ids = [x * 5 for x in range(1000)]
        ray_colours = torch.from_numpy(rays_data['ray_colours'])#[ids, :]
        piece_rotation = torch.from_numpy(rays_data['piece_rotation'])
        ray_locations = torch.from_numpy(rays_data['rays'])#[ids, :]
        del rays_data
        return ray_colours, piece_rotation, ray_locations

class TensorRepresentationsDataset(RayDataset):
    def __getitem__(self, idx):
        piece_name = self.piece_names[idx]
        directions, representation, colour, weight = self.get_rays_from_file(piece_name)

        ids = torch.randperm(len(directions))[:20]
        #non_zero_ids = torch.randperm(len(directions))
        #non_zero_colours = torch.amax(colour[non_zero_ids], dim=-1) > 0.1
        #non_zero_ids = non_zero_ids[non_zero_colours][:15]
        #ids = torch.cat([ids, non_zero_ids])

        return directions[ids], representation[ids], colour[ids], weight[ids]

    def get_rays_from_file(self, piece_name):
        path = os.path.join(self.representation_data_dir, piece_name) + "/rays_archive.npz"
        rays_data = np.load(path)
        directions = torch.from_numpy(rays_data['directions'])#[ids, :]
        representation = torch.from_numpy(rays_data['representation'])
        colour = torch.from_numpy(rays_data['colour'])#[ids, :]
        weight = torch.from_numpy(rays_data['weight'])#[ids, :]
        del rays_data
        return directions, representation, colour, weight


class RotatedRayPairsDataset(RayDataset):
    def __init__(self, split_dict, dataset_info_dict):
        super().__init__(split_dict, dataset_info_dict)
        #piece_numbers = list(set([name.split("_")[1] for name in self.piece_names]))
        #grouped_pieces = [[x for x in self.piece_names if n in x] for n in piece_numbers]
        #self.ray_pair_names = [pair for group in grouped_pieces for pair in zip(group[::2], group[1::2])]

    #def __len__(self):
    #    return len(self.ray_pair_names)
        #return len(self.piece_names) ** 2

    def get_rays_from_file(self, piece_name):
        path = os.path.join(self.representation_data_dir, piece_name) + "/rays_archive.npz"
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

        #piece_name_1, piece_name_2 = self.ray_pair_names[idx]
        #piece_name_1 = self.piece_names[idx%len(self.piece_names)]
        #piece_name_2 = self.piece_names[idx//len(self.piece_names)]

        piece_name = self.piece_names[idx]
        ray_colours, piece_rotation, ray_locations, ray_colours_b, piece_rotation_b, ray_locations_b, rotation_transform = self.get_rays_from_file(piece_name)
        #ray_colours_1, piece_rotation_1, ray_locations_1 = self.get_rays_from_file(piece_name_1)
        #ray_colours_2, piece_rotation_2, ray_locations_2 = self.get_rays_from_file(piece_name_2)

        #piece_rotation_1 = piece_rotation_1.unsqueeze(0)
        #piece_rotation_2 = piece_rotation_2.unsqueeze(0)

        #relative_rotation = torch.matmul(piece_rotation_2, torch.linalg.inv(piece_rotation_1)).squeeze(0)
        #relative_rotation = torch.from_numpy(Rotation.from_matrix(relative_rotation.numpy()).as_quat()).float()

        return ray_colours, piece_rotation, ray_locations, ray_colours_b, piece_rotation_b, ray_locations_b, rotation_transform


class SimpleRotatedRayPairsDataset():  # RotatedRayPairsDataset):
    def get_rays_from_file(self, piece_name):
        path = os.path.join(self.representation_data_dir, piece_name) + "/rays_archive.npz"
        rays_data = np.load(path)
        # ids = [x * 2 for x in range(2500)]
        ray_colours = torch.from_numpy(rays_data['ray_colours'])  # [ids, :]
        piece_rotation = torch.from_numpy(rays_data['piece_rotation'])
        ray_locations = torch.from_numpy(rays_data['rays'])  # [ids, :]
        rotation_num = torch.tensor(rays_data['rotation_num'])
        del rays_data
        return ray_colours, piece_rotation, ray_locations, rotation_num

    def __getitem__(self, idx):
        piece_name_1, piece_name_2 = self.ray_pair_names[idx]
        ray_colours_1, piece_rotation_1, ray_locations_1, rot_num_1 = self.get_rays_from_file(piece_name_1)
        ray_colours_2, piece_rotation_2, ray_locations_2, rot_num_2 = self.get_rays_from_file(piece_name_2)
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
            piece_name = [f for f in self.piece_names if piece_name in f][0]
            ray_colours, piece_rotation, ray_locations = self.get_rays_from_file(piece_name)
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



import os

import numpy as np
import torch
import matplotlib.pyplot as plt

import torch.nn.functional as F


from tqdm import tqdm

from torchvision.utils import save_image

from torch_pointcloud.utils.data import collate
import torch_pointcloud as tp

from torch_pointcloud.transforms import Shift

from torch import nn

from RGBAGridReconstruction import RGBAGridReconstruction

import math

import lightning as L

class TransformerEncoder(nn.Module):
    def __init__(self, transformer_layers=1, representation_size=128, nhead=16):
        super().__init__()

        self.initial_layer = nn.Sequential(nn.Linear(representation_size, 512), nn.ReLU())
        transformer_layer = nn.TransformerEncoderLayer(d_model=512, nhead=nhead, dropout=0.3, batch_first=True)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=transformer_layers)


    def forward(self, x):
        x = self.initial_layer(x)
        x = self.transformer(x)

        return x


def show_point_cloud(pos, color=None, *, ax=None, title=None, size=1, cmap="viridis", view= (34, -160)):
    """Scatter a point cloud. `pos` is (N, 3); `color` is per-point RGB, a label vector, or None."""
    if ax is None:
        ax = plt.figure(figsize=(4, 4)).add_subplot(projection="3d")

    p = pos.detach().cpu().numpy()
    c = color.detach().cpu().numpy() if torch.is_tensor(color) else color
    kw = {"cmap": cmap} if c is not None and np.ndim(c) == 1 else {}
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=c, s=size, depthshade=False, linewidths=0, **kw)
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_box_aspect(p.max(axis=0) - p.min(axis=0), zoom=1.6)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10)
    return ax

class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dataloader = {}
        self.representation_folder_name = ""
        self.embedding_size = 64
        self.batch_size = 32
        self.accumulate_grad_batches = 1
        self.use_batch_norm = True

    def forward(self, batch):
        raise NotImplementedError
        return batch

class SinusoidalPositionalEncoding3D(nn.Module):
    """
    Additive 3D sinusoidal positional encoding.

    Input:
        x: (B, Z, Y, X, D)

    Output:
        positional_encoding: (1, Z, Y, X, D)

    where:
        Z = Y = X = 12
        D = 512
    """

    def __init__(self, dim=256):
        super().__init__()

        self.dim = dim

        # Frequencies for the sinusoidal encoding
        half_dim = dim // 2

        div_term = torch.exp(
            torch.arange(0, half_dim, dtype=torch.float32)
            * (-math.log(10000.0) / half_dim)
        )

        self.register_buffer("div_term", div_term)

    def encode_axis(self, positions):
        """
        positions: (N,)
        returns:   (N, D)
        """
        pe = torch.zeros(
            positions.shape[0],
            self.dim,
            device=positions.device,
            dtype=positions.dtype,
        )

        pe[:, 0::2] = torch.sin(
            positions[:, None] * self.div_term[None, :]
        )

        pe[:, 1::2] = torch.cos(
            positions[:, None] * self.div_term[None, :]
        )

        return pe

    def forward(self, x):
        B, Z, Y, X, D = x.shape

        assert D == self.dim

        # Positions along each axis
        z = torch.arange(Z, device=x.device, dtype=x.dtype)
        y = torch.arange(Y, device=x.device, dtype=x.dtype)
        x_pos = torch.arange(X, device=x.device, dtype=x.dtype)

        # (Z, D), (Y, D), (X, D)
        pe_z = self.encode_axis(z)
        pe_y = self.encode_axis(y)
        pe_x = self.encode_axis(x_pos)

        # Broadcast across the other spatial dimensions
        # (Z, 1, 1, D)
        pe_z = pe_z[:, None, None, :]

        # (1, Y, 1, D)
        pe_y = pe_y[None, :, None, :]

        # (1, 1, X, D)
        pe_x = pe_x[None, None, :, :]

        # (Z, Y, X, D)
        pe = pe_z + pe_y + pe_x

        # Add batch dimension
        # (1, Z, Y, X, D)
        return pe

class MLPPositionalEncoding3D(nn.Module):
    """
    3D positional encoding using an MLP.

    Input:
        x: (B, Z, Y, X, D)

    Output:
        positional_encoding: (1, Z, Y, X, D)

    The 3D coordinate (z, y, x) at each grid location is
    mapped to a D-dimensional positional embedding.
    """

    def __init__(
        self,
        dim=256,
        hidden_dim=128,
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        B, Z, Y, X, D = x.shape

        assert D == self.mlp[-1].out_features

        # Coordinates in [-1, 1]
        z = torch.linspace(
            -1, 1, Z,
            device=x.device,
            dtype=x.dtype,
        )

        y = torch.linspace(
            -1, 1, Y,
            device=x.device,
            dtype=x.dtype,
        )

        x_pos = torch.linspace(
            -1, 1, X,
            device=x.device,
            dtype=x.dtype,
        )

        # Create 3D coordinate grid
        zz, yy, xx = torch.meshgrid(
            z, y, x_pos,
            indexing="ij",
        )

        # (Z, Y, X, 3)
        coords = torch.stack(
            [zz, yy, xx],
            dim=-1,
        )

        # Flatten coordinates
        # (Z*Y*X, 3)
        coords = coords.reshape(-1, 3)

        # MLP -> (Z*Y*X, 512)
        pe = self.mlp(coords)

        # Back to grid
        # (Z, Y, X, 512)
        pe = pe.reshape(Z, Y, X, D)

        # Add batch dimension
        # (1, Z, Y, X, 512)
        return pe.unsqueeze(0)

class NoPositionalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.zeros_like(x)

class RGBAGridEncoder(Encoder):
    def __init__(self, model):
        super().__init__()
        self.dataloader = {"plain" : "RGBAGridDataset", "rotated" : "RandomRotationRGBAGridDataset", "dualRotated" : "DualRandomRotationRGBAGridDataset"}
        self.representation_folder_name = "RGBAGrids"
        self.batch_size = 4
        self.accumulate_grad_batches = 16
        parameters = model.split("-")
        model = parameters[0]


        lightning_model = RGBAGridReconstruction.load_from_checkpoint(f"{model}.ckpt", device=torch.device("cuda"))




        self.embedding_size = lightning_model.scale * 128

        self.model = lightning_model.model.encoder
        self.model = self.model.eval()

        positional_encoding = parameters[1]
        if positional_encoding == "mlp":
            self.positional_encoding = MLPPositionalEncoding3D()
        elif positional_encoding == "sin":
            self.positional_encoding = SinusoidalPositionalEncoding3D()
        elif positional_encoding == "none":
            self.positional_encoding = NoPositionalEncoder()
        elif positional_encoding == "learned":
            self.encoding_param = nn.Parameter(torch.zeros(1, 12, 12, 12, self.embedding_size), requires_grad=True)
            self.positional_encoding = lambda x: self.encoding_param

        del lightning_model




    def forward(self, batch):
        grid  = batch[0]
        with torch.inference_mode():
            representation = self.model(grid)
            representation = representation.permute(0, 2, 3, 4, 1)  # (B, C, Z, Y, X) -> (B, Z, Y, X, C)
        representation = representation + self.positional_encoding(representation)
        representation = representation.view(representation.shape[0], -1, representation.shape[-1])
        del grid
        return representation


class PlainRGBAGrid(Encoder):
    def __init__(self, model):
        super().__init__()
        self.dataloader = {"plain": "RGBAGridDataset", "rotated": "RandomRotationRGBAGridDataset",
                           "dualRotated": "DualRandomRotationRGBAGridDataset"}
        self.representation_folder_name = "RGBAGrids"
        self.batch_size = 8
        self.accumulate_grad_batches = 8
        parameters = model.split("-")

        positional_encoding = parameters[1]
        if positional_encoding == "mlp":
            self.positional_encoding = MLPPositionalEncoding3D()
        elif positional_encoding == "sin":
            self.positional_encoding = SinusoidalPositionalEncoding3D()
        elif positional_encoding == "none":
            self.positional_encoding = NoPositionalEncoder()
        elif positional_encoding == "learned":
            self.encoding_param = nn.Parameter(torch.zeros(1, 12, 12, 12, 512), requires_grad=True)
            self.positional_encoding = lambda x: self.encoding_param

        self.embedding_size = 512

        self.mlp = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, 512), nn.ReLU())

    def forward(self, batch):
        grid = batch[0]
        grid = grid.view(grid.size(0), 12, 8, 12, 8, 12, 8, 4)
        grid = grid.permute(0, 1, 3, 5, 2, 4, 6, 7)
        grid =torch.reshape(grid, (grid.size(0), 12, 12, 12, -1))

        representation = self.mlp(grid)

        representation = representation + self.positional_encoding(representation)
        representation = representation.view(representation.shape[0], -1, representation.shape[-1])

        del grid
        return representation


class GridPoints(Encoder):
    def __init__(self, model):
        super().__init__()
        self.dataloader = {"plain": "RGBAGridDataset", "rotated": "RandomRotationRGBAGridDataset",
                           "dualRotated": "DualRandomRotationRGBAGridDataset"}
        self.representation_folder_name = "RGBAGrids"

        model = encoders[model]()
        self.model = model

        self.batch_size = 1
        self.accumulate_grad_batches = 64
        self.embedding_size = model.embedding_size

        from torch_pointcloud.transforms import EstimateNormals
        self.estimate_normals = EstimateNormals
        self.use_batch_norm = False


    def forward(self, batch):
        grid = batch[0]

        n = grid.shape[-1]  # grid resolution per axis
        coords = torch.linspace(-17, 17, n)  # (n,) evenly spaced values

        xx, yy, zz = torch.meshgrid(coords, coords, coords, indexing="ij")
        coord_grid = torch.stack([xx, yy, zz], dim=-1).unsqueeze(0).expand(self.batch_size, -1, -1, -1, -1).to(grid.device)

        mask = grid[:, 0] > 0.3

        assert grid.shape[0] == 1
        points = coord_grid[mask]
        colours = torch.clamp(grid[:, 1:].permute(0, 2, 3, 4, 1)[mask] * 255, 0, 255).int()
        normals = self.estimate_normals(keys="pos", normal_key="normal")({"pos":points})["normal"]

        features = self.model((points.unsqueeze(0), normals.unsqueeze(0), colours.unsqueeze(0)))

        return features

class RayEncoder(Encoder):
    def __init__(self, model):
        super().__init__()
        self.dataloader = {"plain" : "RGBAGridDataset", "rotated" : "RandomRotationRGBAGridDataset"}
        self.representation_folder_name = "RGBAGrids"
        self.batch_size = 2
        self.accumulate_grad_batches = 4
        parameters = model.split("-")
        model = parameters[0]


        positional_encoding = parameters[1]
        if positional_encoding == "mlp":
            self.positional_encoding = MLPPositionalEncoding3D()
        elif positional_encoding == "sin":
            self.positional_encoding = SinusoidalPositionalEncoding3D()
        elif positional_encoding == "none":
            self.positional_encoding = NoPositionalEncoder()

class EM3FEncoder(Encoder):
    def __init__(self):
        super().__init__()
        self.embedding_size = 64
        self.dataloader = {"plain": "PointCloudDatasetDataloader", "rotated": "RandomRotationPointCloudsDataloader",
                           "dualRotated": "RandomDualRotationPointCloudsDataloader"}
        self.representation_folder_name = "pointclouds2_5k"
        self.batch_size = 8
        self.accumulate_grad_batches = 8

        import hydra
        from hydra import initialize, compose
        with initialize(config_path="em3rf/configs"):
            cfg = compose(config_name="eval.yaml", overrides=["experiment=denoiser_flow_matching"])
            model: L.LightningModule = hydra.utils.instantiate(cfg.get("model"))

            state_dict = torch.load("E-M3RFfinal.pth", map_location="cpu", weights_only=False)["state_dict"]
            model.load_state_dict(state_dict)
            self.model = model.feature_extractor.eval()

    def forward(self, batch, visualise=False):
        points, normals, colors, *extra_args = batch

        B, n_points, _ = batch[0].shape
        features = None
        for i in range(B):
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=False):
                    # Pass the color into the encoder dictionary
                    _, encoder_out_dict = self.model.encoder(
                        {
                            "coord": points[i],
                            "offset": torch.tensor([points.shape[1]]),
                            "feat": torch.cat([points[i], normals[i]], dim=-1),
                            "color": (colors[i].float() / 127.5) - 1.0,  # Added color here
                            "grid_size": torch.tensor(self.model.grid_size).to(points.device),
                        }
                    )
                    model_out = encoder_out_dict["feat"]
            if features is None:
                features = torch.zeros(B, n_points, self.embedding_size).to(batch[0].device)
            features[i] = model_out
            del model_out
        return features

class PointEncoder(Encoder):
    def __init__(self):
        super().__init__()
        self.embedding_size = 64
        #self.model = lambda transformed_data : torch.zeros(self.embedding_size)
        self.transform = lambda x : x
        self.transformed_origin_pos_name = "NONE"
        self.dataloader = {"plain": "PointCloudDatasetDataloader", "rotated" : "RandomRotationPointCloudsDataloader", "dualRotated" : "RandomDualRotationPointCloudsDataloader"}
        self.representation_folder_name = "pointclouds2_5k"
        self.batch_size = 16
        self.accumulate_grad_batches = 4

    def package_pointcloud(self, batch):
        data =  [{"pos": batch[0][i] * 10, "normal": batch[1][i], "color": batch[2][i]} for i in range(batch[0].shape[0])] #
        data = collate(data)
        return data

    def forward(self, batch, visualise=False):
        B, n_points, _ = batch[0].shape
        features = torch.zeros(B, n_points, self.embedding_size).to(batch[0].device)
        for i in range(B):
            packaged_data = self.package_pointcloud([x[i:i+1] for x in batch])
            transformed_data = self.transform(packaged_data)
            with torch.inference_mode():
                model_out = self.model(transformed_data)
                features[i] = model_out
                if visualise:
                    rgb = self.pca_color(model_out).cpu()
                    self.show_clouds(
                        [{"pos": packaged_data["pos"].cpu(), "color": packaged_data["color"].cpu() / 255}, {"pos": transformed_data[self.transformed_origin_pos_name].cpu(), "color": rgb}],
                        ["input: color from the scanner", f"PCA of {model_out.shape[1]:,} features per point"],
                        point_size=0.4,
                    )
                del model_out
        return features

    def show_clouds(self, clouds, titles, point_size=5.0, columns=None, height=4.4):
        """Draw one cloud per panel, colored by its own `color`: RGB rows in [0, 1], or one color for the panel."""
        columns = columns or len(clouds)
        rows = -(-len(clouds) // columns)
        _, axes = plt.subplots(rows, columns, figsize=(4.4 * columns, height * rows), subplot_kw={"projection": "3d"})
        for ax, cloud, title in zip(np.ravel(axes), clouds, titles):
            pos = np.asarray(cloud["pos"])
            ax.scatter(*pos.T, c=cloud["color"], s=point_size, linewidths=0, depthshade=False)
            ax.set_box_aspect(np.ptp(pos, axis=0))
            ax.set_title(title, fontsize=10)
            ax.set_axis_off()
        plt.show()

    def pca_color(self, feat: torch.Tensor) -> torch.Tensor:
        """Map a per-point feature (N, C) to RGB in [0, 1] through its top principal components."""
        _, _, components = torch.pca_lowrank(feat, center=True, q=6, niter=5)
        projected = feat @ components
        projected = projected[:, :3] * 0.6 + projected[:, 3:6] * 0.4
        low, high = projected.min(0, keepdim=True).values, projected.max(0, keepdim=True).values
        return ((projected - low) / (high - low).clamp_min(1e-6)).clamp(0, 1)



class PointTransformerV3(PointEncoder):
    def model_func(self, transformed_data):
        features = self.tp_model(
            transformed_data["x"],
            transformed_data["pos_grid"],
            batch = torch.zeros(transformed_data["x"].shape[0], dtype=torch.int64, device = transformed_data["x"].device),
        )
        return features[transformed_data["inverse"]]

    def __init__(self):
        super().__init__()
        tp_model, info = tp.create_model(
            "ptv3-base.s3dis-area5.pointcept",
            task="segmentation",
            pretrained=True,
            num_classes=0,
            return_info=True
        )
        self.tp_model = tp_model.eval().to(torch.device("cuda:0"))
        self.model = self.model_func
        self.transform = info["transform"]
        self.embedding_size = 64
        self.transformed_origin_pos_name = "origin_pos"

    def package_pointcloud(self, batch):
        data =  [{"pos": batch[0][i] * 10, "normal": batch[1][i], "color": batch[2][i], "category" : 0} for i in range(batch[0].shape[0])] #
        data = collate(data)
        data["segment"] = data["batch"]
        return data


class PointNet2(PointEncoder):
    def model_func(self, transformed_data):
        features = self.tp_model(
            x = transformed_data["x"],
            pos = transformed_data["pos"],
            batch = torch.zeros(transformed_data["x"].shape[0], dtype=torch.int64, device=transformed_data["x"].device),
        )
        return features

    def __init__(self):
        super().__init__()
        tp_model, info = tp.create_model(
            "pointnet2.s3dis-area5.openpoints",
            task="segmentation",
            pretrained=True,
            num_classes=0,
            return_info=True
        )
        self.tp_model = tp_model.eval().to(torch.device("cuda:0"))
        self.model = self.model_func
        self.transform = info["transform"]
        self.embedding_size = 128
        self.transformed_origin_pos_name = "pos"

    def package_pointcloud(self, batch):
        data = [{"pos": batch[0][i] * 10, "normal": batch[1][i], "color": batch[2][i]} for i in
                range(batch[0].shape[0])]  #
        data = collate(data)
        return data

class PointNextxl(PointEncoder):
    def model_func(self, transformed_data):
        features = self.tp_model(
            x = transformed_data["x"],
            pos = transformed_data["pos"],
            batch = torch.zeros(transformed_data["x"].shape[0], dtype=torch.int64, device=transformed_data["x"].device),
        )
        return features[transformed_data["inverse"]]

    def __init__(self):
        super().__init__()
        tp_model, info = tp.create_model(
            "pointnext-xl.s3dis-area5.openpoints",
            task="segmentation",
            pretrained=True,
            num_classes=0,
            return_info=True
        )
        self.tp_model = tp_model.eval().to(torch.device("cuda:0"))
        self.model = self.model_func
        self.transform = info["transform"]
        self.embedding_size = 64
        self.transformed_origin_pos_name = "pos"

    def package_pointcloud(self, batch):
        data = [{"pos": batch[0][i] * 10, "normal": batch[1][i], "color": batch[2][i], "category": 0} for i in range(batch[0].shape[0])]
        data = collate(data)
        data["segment"] = data["batch"]
        return data

class Sonata(PointEncoder):
    def model_func(self, transformed_data):
        features = self.tp_model(
            x = transformed_data["x"],
            pos_grid = transformed_data["pos_grid"],
            batch = torch.zeros(transformed_data["x"].shape[0], dtype=torch.int64, device=transformed_data["x"].device),
        )
        return features[transformed_data["inverse"]]

    def __init__(self):
        super().__init__()
        tp_model, info = tp.create_model(
            "sonata-lp.scannet20.fair",
            task="segmentation",
            pretrained=True,
            num_classes=0,
            return_info=True
        )
        self.tp_model = tp_model.eval().to(torch.device("cuda:0"))
        self.model = self.model_func
        self.transform = info["transform"]
        self.embedding_size = 1232
        self.transformed_origin_pos_name = "origin_pos"


    def package_pointcloud(self, batch):
        data = [{"pos": batch[0][i] * 10, "normal": batch[1][i], "color": batch[2][i], "category": 0} for i in range(batch[0].shape[0])]
        data = collate(data)
        data["segment"] = data["batch"]
        return data


class KPConv(PointEncoder):
    def model_func(self, transformed_data):
        features = self.tp_model(
            x = transformed_data["x"],
            pos = transformed_data["pos"],
            batch = torch.zeros(transformed_data["x"].shape[0], dtype=torch.int64, device=transformed_data["x"].device),
        )
        return features[transformed_data["inverse"]]

    def __init__(self):
        super().__init__()
        tp_model, info = tp.create_model(
            "kpfcnn-base.s3dis.hugues-thomas",
            task="segmentation",
            pretrained=True,
            num_classes=0,
            return_info=True
        )
        self.tp_model = tp_model.eval().to(torch.device("cuda:0"))
        self.model = self.model_func
        self.transform = info["transform"]
        self.embedding_size = 128
        self.transformed_origin_pos_name = "origin_pos"


    def package_pointcloud(self, batch):
        data = [{"pos": batch[0][i] * 10, "normal": batch[1][i], "color": batch[2][i], "category": 0} for i in range(batch[0].shape[0])]
        data = collate(data)
        data["segment"] = data["batch"]
        return data



encoders = {"PTV3" : PointTransformerV3, "PointNet2" : PointNet2, "PointNextxl" : PointNextxl, "Sonata" : Sonata, "KPConv" : KPConv, "RGBAGridEncoder" : RGBAGridEncoder, "PlainRGBAGrid" : PlainRGBAGrid, "EM3FEncoder" : EM3FEncoder, "GridPoints" : GridPoints}

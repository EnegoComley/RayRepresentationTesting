import torch
from lightning.pytorch.loggers import WandbLogger

torch.set_float32_matmul_precision('medium')

from DatasetLoading import RepairDatasetLoader
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from torchmetrics.segmentation import DiceScore
from pytorch_lightning.callbacks import LearningRateMonitor
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from scipy.spatial.transform import Rotation


import argparse

import lightning as L
import os

import json

torch.set_float32_matmul_precision('medium')

from EvaluationUtils import encoders
from EvaluationUtils import TransformerEncoder

import math

import cv2
import numpy as np
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pair Rotation Prediction Evaluation')
    parser.add_argument('--model', type=str, default="PTV3", help='The model to be evaluated')
    parser.add_argument('--ncc', action='store_true', help='Running on the NCC?')
    parser.add_argument("--no_logger", action='store_true', help="Disable logging to Weights and Biases")
    parser.add_argument("--angle_divider", type=float, default=1, help="The divider for the random angle")
    parser.add_argument('--overfit', action='store_true', help='Overfit the model on a small subset of the data for debugging')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--nPoints', type=int, default=2500, help='Number of points to sample')
    parser.add_argument('--nOverfit', type=int, default=-1, help='Number of points to sample')
    #parser.add_argument("--param_test", action='store_true', help="Run parameter test")



    args = parser.parse_args()
    print(args)

def quat2mat(quat):
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                          2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                          2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat
class VerySimplePairRotationPredictionNetwork(nn.Module):
    def __init__(self, encoder_model):
        super().__init__()


        representation_size = encoder_model.embedding_size

        self.encoder_model = encoder_model

        self.mlp = nn.Sequential(nn.Linear(representation_size, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 128),
                                 nn.ReLU())

        self.head = nn.Sequential(nn.Linear(256, 128),
                                  nn.ReLU(),
                                  nn.Linear(128, 64),
                                  nn.ReLU(),
                                  nn.Linear(64, 4))





    def forward(self, batch):
        x1, x2, _ = batch
        #combined_rep = (torch.cat([x1[i].unsqeeze(1), x2[i].unsqeeze(1)], dim=1).view(x1[i].shape[0] * 2, *x1[i].shape[1:]) for i in range(len(x1)))

        x1 = self.encoder_model(x1)
        x2 = self.encoder_model(x2)

        x1 = self.mlp(x1)
        x2 = self.mlp(x2)

        x1 = torch.max(x1, dim=1)[0]
        x2 = torch.max(x2, dim=1)[0]

        x = self.head(torch.cat([x1, x2], dim=1))

        #if args.param_test:
        #    x = self.test_param.expand_as(x)

        return x


class NoEncoderPairRotationPredictionNetwork(nn.Module):
    def __init__(self, encoder_model):
        super().__init__()


        representation_size = encoder_model.embedding_size

        self.encoder_model = encoder_model

        self.mlp = nn.Sequential(nn.Linear(3, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 128),
                                 nn.ReLU())

        self.head = nn.Sequential(nn.Linear(256, 128),
                                  nn.ReLU(),
                                  nn.Linear(128, 64),
                                  nn.ReLU(),
                                  nn.Linear(64, 4))





    def forward(self, batch):
        x1, x2, _ = batch
        #combined_rep = (torch.cat([x1[i].unsqeeze(1), x2[i].unsqeeze(1)], dim=1).view(x1[i].shape[0] * 2, *x1[i].shape[1:]) for i in range(len(x1)))

        #x1 = torch.cat((x1[0], x1[1], x1[2]), dim=2)
        #x2 = torch.cat((x2[0], x2[1], x2[2]), dim=2)
        x1 = x1[0]
        x2 = x2[0]

        x1 = self.mlp(x1)
        x2 = self.mlp(x2)

        x1 = torch.max(x1, dim=1)[0]
        x2 = torch.max(x2, dim=1)[0]

        x = self.head(torch.cat([x1, x2], dim=1))

        #if args.param_test:
        #    x = self.test_param.expand_as(x)

        return x

class SmallPairRotationPredictionNetwork(nn.Module):
    def __init__(self, encoder_model):
        super().__init__()


        representation_size = encoder_model.embedding_size

        self.encoder_model = encoder_model
        self.combined_transformer = TransformerEncoder(transformer_layers=2, representation_size=representation_size, latent_size=128, nhead=16)

        self.pos_encoder = nn.Parameter(torch.randn(2, 1, 1, representation_size), requires_grad=True)

        #if encoder_model.use_batch_norm:
        #    self.head = nn.Sequential(nn.Linear(512, 64),
        #                              #nn.BatchNorm1d(64),
        #                              nn.ReLU(),
        #                              nn.Linear(64, 4))
        #else:
        #    self.head = nn.Sequential(nn.Linear(512, 64),
        #                              nn.ReLU(),
        #                              nn.Linear(64, 4))
        self.head = nn.Sequential(nn.Linear(128, 256),
                                  nn.ReLU(),
                                  nn.Linear(256, 64),
                                  nn.ReLU(),
                                  nn.Linear(64, 4))

        self.cls_token = nn.Parameter(torch.randn(1, 1, representation_size), requires_grad=True)
        #if args.param_test:
        #    self.test_param = nn.Parameter(torch.randn(1, 4))

    def positionalencoding1d(self, d_model, length):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                              -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe


    def forward(self, batch):
        x1, x2, _ = batch
        #combined_rep = (torch.cat([x1[i].unsqeeze(1), x2[i].unsqeeze(1)], dim=1).view(x1[i].shape[0] * 2, *x1[i].shape[1:]) for i in range(len(x1)))



        x1 = self.encoder_model(x1)
        x2 = self.encoder_model(x2)

        #x = self.encoder_model(combined_rep)

        batch_size = x1.shape[0]


        #x1 = x1.requires_grad_()
        #x2 = x2.requires_grad_()
        #x1.retain_grad()
        #x2.retain_grad()
        #print(x1.shape, x2.shape, self.pos_encoder.shape)
        #x = self.combined_transformer(torch.cat([x1 + self.pos_encoder[0].expand_as(x1), x2 + self.pos_encoder[1].expand_as(x2)], dim=1))

        x = torch.cat([x1, x2], dim=1)
        print(x.shape)
        #x = torch.cat([x1 + self.pos_encoder[0].expand_as(x1), x2 + self.pos_encoder[1].expand_as(x2)], dim=1)

        # Add the class token on to the end
        x = torch.cat([self.cls_token.repeat(batch_size, 1, 1), x], dim=1)

        x = x + self.positionalencoding1d(x.shape[-1], x.shape[1]).to(x.device)




        x = self.combined_transformer(x)



        #x = torch.mean(x, dim=1)
        x = x[:, 0]


        x = self.head(x)



        return x


class PairRotationPredictionNetwork(nn.Module):
    def __init__(self, encoder_model):
        super().__init__()


        representation_size = encoder_model.embedding_size

        self.encoder_model = encoder_model
        #self.transformer = TransformerEncoder(transformer_layers=2, representation_size=representation_size, latent_size=512, nhead=16)
        self.combined_transformer = TransformerEncoder(transformer_layers=2, representation_size=representation_size)

        self.pos_encoder = nn.Parameter(torch.randn(2, 1, 1, representation_size), requires_grad=True)

        #if encoder_model.use_batch_norm:
        #    self.head = nn.Sequential(nn.Linear(512, 64),
        #                              #nn.BatchNorm1d(64),
        #                              nn.ReLU(),
        #                              nn.Linear(64, 4))
        #else:
        #    self.head = nn.Sequential(nn.Linear(512, 64),
        #                              nn.ReLU(),
        #                              nn.Linear(64, 4))
        self.head = nn.Sequential(nn.Linear(512, 256),
                                  nn.ReLU(),
                                  nn.Linear(256, 64),
                                  nn.ReLU(),
                                  nn.Linear(64, 4))

        #self.cls_token = nn.Parameter(torch.randn(1, 1, 128), requires_grad=True)

        #if args.param_test:
        #    self.test_param = nn.Parameter(torch.randn(1, 4))

    def positionalencoding1d(self, d_model, length, device):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model, device=device)
        position = torch.arange(0, length, device=device).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float, device=device) *
                              -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe


    def forward(self, batch, use_quaternion=True):
        x1, x2, _ = batch
        #combined_rep = (torch.cat([x1[i].unsqeeze(1), x2[i].unsqeeze(1)], dim=1).view(x1[i].shape[0] * 2, *x1[i].shape[1:]) for i in range(len(x1)))



        x1 = self.encoder_model(x1)
        x2 = self.encoder_model(x2)

        #x = self.encoder_model(combined_rep)

        batch_size = x1.shape[0]

        #x1 = self.transformer(x1)
        #x2 = self.transformer(x2)

        #x1 = x1[:, 1:]
        #x2 = x2[:, 1:]

        #print(x1.shape, x2.shape, self.pos_encoder.shape)
        #x = self.combined_transformer(torch.cat([x1 + self.pos_encoder[0].expand_as(x1), x2 + self.pos_encoder[1].expand_as(x2)], dim=1))

        #x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x1 + self.pos_encoder[0].expand_as(x1), x2 + self.pos_encoder[1].expand_as(x2)], dim=1)


        # Add the class token on to the end
        #x = torch.cat([self.cls_token.repeat(batch_size, 1, 1), x], dim=1)

        #x = x + self.positionalencoding1d(128, x.shape[1], device=x.device)




        x = self.combined_transformer(x)



        x = torch.max(x, dim=1)[0]
        #x = x[:, 0]
        x = self.head(x)

        #if args.param_test:
        #    x = self.test_param.expand_as(x)

        if not use_quaternion:
            # Reshape to (B, 3, 3)
            return quat2mat(x), x

        return x

class UpWrapper(nn.Module):
    def __init__(self, encoder_model, embed_size):
        super().__init__()
        representation_size = encoder_model.embedding_size
        self.encoder_model = encoder_model

        self.mlp = nn.Sequential(nn.Linear(representation_size, embed_size//2),
                                 nn.ReLU(),
                                 nn.Linear(embed_size//2, embed_size),
                                 )

    def forward(self, x):
        x = self.encoder_model(x)

        x = self.mlp(x).transpose(1, 2)
        return x

class DCPNetwork(nn.Module):
    class model_params(nn.Module):
        def __init__(self, encoder_model):
            super().__init__()
            self.emb_nn = 'pointnet' if encoder_model is None else "own"
            self.emb_dims = 1024
            if encoder_model is not None:
                self.encoder = UpWrapper(encoder_model, embed_size=self.emb_dims)
            self.cycle = False
            self.pointer = "transformer"
            self.head = "mlp"
            self.n_blocks = 1
            self.n_heads = 4
            self.ff_dims = 1024
            self.dropout = 0.0

    def __init__(self, encoder_model=None):
        super().__init__()
        from dcpModified.model import DCP
        self.DCP = DCP(self.model_params(encoder_model))
        self.own_encoder = encoder_model is not None

    def forward(self, batch):
        x1, x2, _ = batch
        if self.own_encoder:
            return self.DCP(x1, x2)[:2]
        else:
            return self.DCP(x1[0].transpose(2, 1), x2[0].transpose(2, 1))[:2]




class PairRotationPrediction(L.LightningModule):
    def __init__(self, encoder_model, lr=1e-3):
        super().__init__()
        #self.model = PairRotationPredictionNetwork(encoder_model)
        #self.model = SmallPairRotationPredictionNetwork(encoder_model)
        #self.model = VerySimplePairRotationPredictionNetwork(encoder_model)
        #self.model = NoEncoderPairRotationPredictionNetwork(encoder_model)
        #self.model = DCPNetwork(None)
        self.model = DCPNetwork(encoder_model)
        self.lr = lr

    def calculate_loss_OLD(self, batch, stage):
        gt_rotation = batch[-1]
        predicted_rotation = self.model(batch)


        #gt_norm1 = self.get_normals(batch[0][-1])
        #gt_norm2 = self.get_normals(batch[1][-1])
        #normal_loss = 0.5 * (nn.functional.mse_loss(gt_norm1, pred_norm1) + nn.functional.mse_loss(gt_norm2, pred_norm2))
        #self.log(f'{stage}_normal_loss', normal_loss)


        # Diagnostic: do not normalize here so we can inspect raw directional signal and gradients
        #predicted_rotation = nn.functional.normalize(predicted_rotation, dim=-1)


        loss = nn.functional.mse_loss(predicted_rotation, gt_rotation)
        #loss = torch.mean(1.0 - torch.abs(dot))
        self.log(stage + '_loss', loss)


        #dot = torch.sum(predicted_rotation * gt_rotation, dim=-1)
        #dot = torch.clamp(dot.abs(), -1.0, 1.0)

        cos_theta = torch.sum(predicted_rotation * gt_rotation, dim=-1)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        rot_error = torch.acos(cos_theta) * 2.0
        rot_error = torch.rad2deg(rot_error)

        #rot_error_rad = 2.0 * torch.acos(dot)
        #rot_error = torch.rad2deg(rot_error_rad)

        rot_rmse = torch.sqrt(rot_error.pow(2).mean())
        rot_upper_error = torch.quantile(rot_error, 0.95)
        # Calculate standard deviation of rotation error
        rot_error_std = torch.sqrt(rot_error.var())

        #print("bob", gt_rotation.shape)
        #print(gt_rotation.cpu().detach().numpy(), predicted_rotation.cpu().detach().numpy(), rot_rmse.cpu().detach().numpy())
        #print(rot_rmse.cpu().detach().numpy())

        self.log(stage + '_angular_error', rot_rmse)
        self.log(stage + '_angular_error_95th_percentile', rot_upper_error)
        self.log(stage + '_angular_error_std', rot_error_std)

        #self.log(stage + '_total_loss', rot_rmse + normal_loss)
        return rot_rmse# + normal_loss * 50

    def calculate_loss_euler(self, batch, stage):
        gt_rotation = batch[-1]
        predicted_rotation = self.model(batch)

        loss = nn.functional.mse_loss(predicted_rotation, gt_rotation)
        self.log(stage + '_loss', loss)

    def calculate_loss_CLAUDE(self, batch, stage): #
        gt = batch[-1]
        pred = self.model(batch)

        #print(pred, nn.functional.normalize(pred, dim=-1), gt)

        mse_loss = F.mse_loss(pred, gt)
        self.log(f"{stage}_mse_loss", mse_loss)

        pred = F.normalize(pred, dim=-1)


        pred_rotation = Rotation.from_quat(pred[0].cpu().detach().numpy())
        rotated_pointcloud = batch[0][0][0] @ torch.tensor(pred_rotation.as_matrix().T, device=self.device, dtype=pred.dtype)

        self.visualize_pointclouds([(batch[0][0][0], batch[0][2][0]), (batch[1][0][0], batch[1][2][0]), (rotated_pointcloud, batch[0][2][0])])

        dot = (pred * gt).sum(-1)

        dot_loss = (1.0 - dot.abs()).mean()  # == min(||p-q||², ||p+q||²)/2, invariant to q -> -q
        self.log(f"{stage}_dot_loss", dot_loss)



        with torch.no_grad():
            rot_error = torch.rad2deg(2.0 * torch.acos(dot.abs().clamp(max=1.0)))  # 0–180°
            self.log(f"{stage}_angular_error", rot_error.pow(2).mean().sqrt())
            self.log(f"{stage}_angular_error_95th_percentile", torch.quantile(rot_error, 0.95))
            self.log(f"{stage}_angular_error_std", rot_error.std())
        return mse_loss #+ dot_loss



    def calculate_loss_DCP_based(self, batch, stage):  #
        gt = batch[-1]
        gt_mat = quat2mat(gt)

        pred, pred_quat = self.model(batch, use_quaternion=False)
        #pred_rotation = Rotation.from_matrix(pred.cpu().detach().numpy(), assume_valid=False)
        #pred_quat = torch.tensor(pred_rotation.as_quat(), device=self.device, dtype=pred.dtype)

        # print(pred, nn.functional.normalize(pred, dim=-1), gt)

        DCP_loss = F.mse_loss(torch.matmul(pred.transpose(2, 1), gt_mat), torch.eye(3, device=self.device, dtype=pred.dtype).unsqueeze(0).repeat(pred.shape[0], 1, 1))
        self.log(f"{stage}_DCP_loss", DCP_loss)


        #pred = F.normalize(pred, dim=-1)

        n_pred_quat = nn.functional.normalize(pred_quat, dim=-1)
        #rotated_pointcloud = batch[0][0][0] @ quat2mat(n_pred_quat)[0].T

        #self.visualize_pointclouds(
        #    [(batch[0][0][0], batch[0][2][0]), (batch[1][0][0], batch[1][2][0]), (rotated_pointcloud, batch[0][2][0])])


        dot = (n_pred_quat * gt).sum(-1)

        dot_loss = (1.0 - dot.abs()).mean()  # == min(||p-q||², ||p+q||²)/2, invariant to q -> -q
        self.log(f"{stage}_dot_loss", dot_loss)

        with torch.no_grad():
            rot_error = torch.rad2deg(2.0 * torch.acos(dot.abs().clamp(max=1.0)))  # 0–180°
            self.log(f"{stage}_angular_error", rot_error.pow(2).mean().sqrt())
            self.log(f"{stage}_angular_error_95th_percentile", torch.quantile(rot_error, 0.95))
            self.log(f"{stage}_angular_error_std", rot_error.std())

        return DCP_loss
    def calculate_loss(self, batch, stage):  #_DCP
        gt = batch[-1]
        gt_mat = quat2mat(gt)

        pred, translation_pred = self.model(batch)
        pred_rotation = Rotation.from_matrix(pred.cpu().detach().numpy(), assume_valid=False)
        pred_quat = torch.tensor(pred_rotation.as_quat(), device=self.device, dtype=pred.dtype)

        # print(pred, nn.functional.normalize(pred, dim=-1), gt)

        DCP_loss = F.mse_loss(torch.matmul(pred.transpose(2, 1), gt_mat), torch.eye(3, device=self.device, dtype=pred.dtype).unsqueeze(0).repeat(pred.shape[0], 1, 1))
        self.log(f"{stage}_DCP_loss", DCP_loss)

        #translation_loss = F.mse_loss(translation_pred, torch.zeros_like(translation_pred))
        #self.log(f"{stage}_translation_loss", translation_loss)


        #pred = F.normalize(pred, dim=-1)


        #rotated_pointcloud = batch[0][0][0] @ pred[0].T
        #self.visualize_pointclouds([(batch[0][0][0], batch[0][2][0]), (batch[1][0][0], batch[1][2][0]), (rotated_pointcloud, batch[0][2][0])])


        n_pred_quat = nn.functional.normalize(pred_quat, dim=-1)

        dot = (n_pred_quat * gt).sum(-1)

        dot_loss = (1.0 - dot.abs()).mean()  # == min(||p-q||², ||p+q||²)/2, invariant to q -> -q
        self.log(f"{stage}_dot_loss", dot_loss)

        with torch.no_grad():
            rot_error = torch.rad2deg(2.0 * torch.acos(dot.abs().clamp(max=1.0)))  # 0–180°
            self.log(f"{stage}_angular_error", rot_error.pow(2).mean().sqrt())
            self.log(f"{stage}_angular_error_95th_percentile", torch.quantile(rot_error, 0.95))
            self.log(f"{stage}_angular_error_std", rot_error.std())


        return DCP_loss# + translation_loss


    def visualize_pointclouds(self,
            pointclouds,
            image_size=600,
            point_size=2,
            azimuth=45,
            elevation=10,
            padding=0.1,
    ):
        """
        Visualise multiple 3D point clouds from the same fixed view.

        Args:
            pointclouds:
                List of (points, colour) tuples.

                points:
                    PyTorch tensor of shape (N, 3)

                colour:
                    Either:

                    1. A single colour:
                       (B, G, R)

                    2. A per-point colour tensor/array:
                       shape (N, 3)

                    OpenCV uses BGR colours.

            image_size:
                Size of each output image.

            point_size:
                Radius of each rendered point.

            view:
                "xy" -> look along Z
                "xz" -> look along Y
                "yz" -> look along X

            padding:
                Fraction of image to leave around the point clouds.

        Returns:
            combined:
                Combined OpenCV image.
        """

        clouds = []

        # ------------------------------------------------------------
        # Convert input
        # ------------------------------------------------------------

        for points, colour in pointclouds:

            # Points -> numpy
            if isinstance(points, torch.Tensor):
                points = points.detach().cpu().numpy()

            points = np.asarray(points, dtype=np.float32)

            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError(
                    f"Expected points with shape (N, 3), "
                    f"got {points.shape}"
                )

            # --------------------------------------------------------
            # Convert colours
            # --------------------------------------------------------

            if isinstance(colour, torch.Tensor):
                colour = colour.detach().cpu().numpy()

            colour = np.asarray(colour)

            # Single colour: (3,)
            if colour.ndim == 1:

                if colour.shape[0] != 3:
                    raise ValueError(
                        f"Colour must have 3 values, got {colour.shape}"
                    )

                colour = tuple(
                    int(c) for c in colour
                )

            # Per-point colours: (N, 3)
            elif colour.ndim == 2:

                if colour.shape != points.shape:
                    raise ValueError(
                        f"Per-point colour shape {colour.shape} "
                        f"does not match point shape {points.shape}"
                    )

                colour = np.asarray(
                    colour,
                    dtype=np.float32
                )

                # If colours are in [0, 1], convert to [0, 255]
                if colour.max() <= 1.0:
                    colour = colour * 255.0

                colour = np.clip(
                    colour,
                    0,
                    255
                ).astype(np.uint8)

            else:
                raise ValueError(
                    f"Colour must have shape (3,) or (N, 3), "
                    f"got {colour.shape}"
                )

            clouds.append((points, colour))

        if len(clouds) == 0:
            raise ValueError("No point clouds provided.")

        # ------------------------------------------------------------
        # Determine common bounds
        # ------------------------------------------------------------

        all_points = np.concatenate(
            [points for points, _ in clouds],
            axis=0
        )

        # ------------------------------------------------------------
        # Select projection
        # ------------------------------------------------------------

        projected_all = self.project_points(all_points, azimuth, elevation)

        min_xy = projected_all.min(axis=0)
        max_xy = projected_all.max(axis=0)

        centre = (min_xy + max_xy) / 2

        extent = np.max(max_xy - min_xy)

        if extent < 1e-8:
            extent = 1.0

        extent *= (1 + 2 * padding)

        # ------------------------------------------------------------
        # Render clouds
        # ------------------------------------------------------------

        images = []

        for points, colours in clouds:

            projected = self.project_points(points, azimuth, elevation)

            # --------------------------------------------------------
            # Project to image coordinates
            # --------------------------------------------------------

            x = (
                    (projected[:, 0] - centre[0])
                    / extent
                    * image_size
                    + image_size / 2
            )

            y = (
                    -(projected[:, 1] - centre[1])
                    / extent
                    * image_size
                    + image_size / 2
            )

            x = np.round(x).astype(np.int32)
            y = np.round(y).astype(np.int32)

            # --------------------------------------------------------
            # Create image
            # --------------------------------------------------------

            image = np.zeros(
                (image_size, image_size, 3),
                dtype=np.uint8
            )

            # --------------------------------------------------------
            # Valid points
            # --------------------------------------------------------

            valid = (
                    (x >= 0)
                    & (x < image_size)
                    & (y >= 0)
                    & (y < image_size)
            )

            x = x[valid]
            y = y[valid]

            # --------------------------------------------------------
            # Draw points
            # --------------------------------------------------------

            if isinstance(colours, tuple):

                # One colour for the whole cloud

                for px, py in zip(x, y):
                    cv2.circle(
                        image,
                        (int(px), int(py)),
                        int(point_size),
                        colours,
                        -1,
                        cv2.LINE_AA,
                    )

            else:

                # Per-point colours

                colours = colours[valid]

                for px, py, colour in zip(x, y, colours):
                    # Explicitly convert numpy values to Python ints
                    bgr = (
                        int(colour[0]),
                        int(colour[1]),
                        int(colour[2]),
                    )

                    cv2.circle(
                        image,
                        (int(px), int(py)),
                        int(point_size),
                        bgr,
                        -1,
                        cv2.LINE_AA,
                    )

            images.append(image)

        # ------------------------------------------------------------
        # Combine images
        # ------------------------------------------------------------

        combined = np.hstack(images)

        # ------------------------------------------------------------
        # Display
        # ------------------------------------------------------------

        cv2.imshow(
            "Point Clouds",
            combined
        )

        cv2.waitKey(1)
        #cv2.destroyAllWindows()

        return combined

    import numpy as np

    def project_points(self, points, azimuth, elevation):
        """
        Project 3D points onto a fixed camera view.

        azimuth:
            Rotation around Z axis, in degrees.

        elevation:
            Camera elevation above the XY plane, in degrees.
        """

        az = np.deg2rad(azimuth)
        el = np.deg2rad(elevation)

        # Camera forward direction
        forward = np.array([
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el),
        ])

        # Camera right direction
        right = np.array([
            -np.sin(az),
            np.cos(az),
            0.0,
        ])

        # Camera up direction
        up = np.cross(right, forward)

        # Project onto camera plane
        x = points @ right
        y = points @ up

        return np.stack([x, y], axis=1)

    #def calculate_loss(self, batch, stage):
    #    gt_rotation = batch[-1]
    #    predicted_rotation = self.model(batch)
    #    predicted_rotation = nn.functional.normalize(predicted_rotation, dim=-1)

    #    loss = nn.functional.mse_loss(predicted_rotation, gt_rotation)
    #    self.log(stage + '_loss', loss)

    #    cos_theta = torch.sum(predicted_rotation * gt_rotation, dim=-1)
    #    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    #    rot_error = torch.acos(cos_theta)
    #    rot_error = torch.rad2deg(rot_error)
    #    rot_rmse = torch.sqrt(rot_error.pow(2).mean())
    #    rot_upper_error = torch.quantile(rot_error, 0.95)
    #    # Calculate standard deviation of rotation error
    #    rot_error_std = torch.sqrt(rot_error.var())

        #print("bob", gt_rotation.shape)
        #print(gt_rotation.cpu().detach().numpy(), predicted_rotation.cpu().detach().numpy(), rot_rmse.cpu().detach().numpy())
        #print(rot_rmse.cpu().detach().numpy())

     #   self.log(stage + '_angular_error', rot_rmse)
     #   self.log(stage + '_angular_error_95th_percentile', rot_upper_error)
     #   self.log(stage + '_angular_error_std', rot_error_std)

     #   return loss

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

    datasets_path = "~/masters/datasets/" if args.ncc else "~/Documents/masters/datasets/"

    encoder = encoders[args.model.split("_")[0]](args.model.split("_")[1]) if "_" in args.model else encoders[args.model]()

    dataset_kwargs = {}
    if encoder.dataloader["dualRotated"] == "RandomDualRotationPointCloudsDataloader":
        dataset_kwargs["n_points"] = args.nPoints

    batch_size = encoder.batch_size
    if args.overfit:
        batch_size = 1
    if args.nOverfit != -1:
        batch_size = min(batch_size, args.nOverfit)

    dataset_loader = RepairDatasetLoader(batch_size=batch_size, dataset_type=encoder.dataloader["dualRotated"],
                                         representation_folder_name=encoder.representation_folder_name, num_workers=8, data_dir=datasets_path, angle_divider=args.angle_divider, overfit=args.overfit, nOverfit=args.nOverfit, ncc=args.ncc, **dataset_kwargs)
    L.seed_everything(42)

    run_name = f"{args.model}"
    #if args.param_test:
    #    run_name = "SPECIAL GUESS"
    if args.overfit:
        run_name = "Overfit_" + run_name
    if args.lr != 1e-3:
        run_name += f"_lr={args.lr}"
    if args.nOverfit != -1:
        run_name = f"nOverfit={args.nOverfit}_" + run_name




    project_name = f"PairRotationPredictionEvaluationUpTo{180/args.angle_divider}"# if args.angle_divider != 1 else "PairRotationPredictionEvaluation"



    wandb_logger = False if args.no_logger else WandbLogger(name=run_name, project=project_name)
    ckpt_dir = f"{project_name}Checkpoints/{run_name}/"
    test_output_dir = f"{project_name}Results/{run_name}/"



    model = PairRotationPrediction(encoder_model=encoder, lr=args.lr)

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(dirpath=ckpt_dir, )
    epochs = 1000 if args.overfit or args.nOverfit != -1 else 2000
    precision = "32-true"  # PTV3 doesn't support float16 for normal estimation on CUDA
    #lr_monitor = LearningRateMonitor(logging_interval='step')
    accelerator = "gpu"
    trainer = L.Trainer(max_epochs=epochs, accelerator=accelerator, callbacks=[] if args.no_logger else [checkpoint_callback], precision=precision, logger=wandb_logger, accumulate_grad_batches=encoder.learned_batch_size/encoder.batch_size if not args.overfit or args.nOverfit != -1 else 1, num_sanity_val_steps=0)
    trainer.fit(model, datamodule=dataset_loader)

    results = trainer.test(model, datamodule=dataset_loader)

    with open(f"{test_output_dir}/test_results.json", "w") as f:
        json.dump(results, f, indent=4)







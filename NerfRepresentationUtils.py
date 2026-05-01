import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt


class ColourPredictionPredictionNetwork(nn.Module):
    def __init__(self, latent_size):
        super().__init__()


        self.representation_encoder = nn.Sequential(nn.Linear(27, 128),
                                               nn.ReLU(),
                                               nn.Linear(128, latent_size),
                                               nn.ReLU())

        self.head = nn.Sequential(nn.Linear(latent_size + 3, 3))



    def forward(self, representation, alpha, direction):
        x = self.get_latent_representation(representation, alpha)
        x = torch.cat((x, direction), dim=1)
        x = self.head(x)

        return x

    def get_latent_representation(self, representation, alpha):
        #x = torch.cat((representation, alpha), dim=1)
        return self.representation_encoder(representation)

def plot_colored_voxels(coords, colours, axis_names=None, assume_normalized=True, figsize=(8, 8), edgecolor=None):
    """
    coords: tuple/list of three 1D torch tensors or numpy arrays (x_coords, y_coords, z_coords), length N
    colours: (N,3) torch tensor or numpy array with RGB values in 0..1
    Returns: (fig, ax)
    """

    # BGR to RGB
    colours = torch.flip(colours, dims=[1])

    x_t, y_t, z_t = (c.cpu().numpy() if isinstance(c, torch.Tensor) else np.asarray(c) for c in coords)
    cols = colours.cpu().numpy() if isinstance(colours, torch.Tensor) else np.asarray(colours)

    if not assume_normalized and cols.size and cols.max() > 1.0:
        cols = cols / 255.0

    cols = np.asarray(cols, dtype=float)
    if cols.ndim != 2 or cols.shape[1] < 3:
        raise ValueError("colours must have shape (N,3)")

    ux = np.unique(x_t)
    uy = np.unique(y_t)
    uz = np.unique(z_t)
    nx, ny, nz = ux.size, uy.size, uz.size

    ix_map = {v: i for i, v in enumerate(ux)}
    iy_map = {v: i for i, v in enumerate(uy)}
    iz_map = {v: i for i, v in enumerate(uz)}

    mask = np.zeros((nx, ny, nz), dtype=bool)
    facecolors = np.zeros((nx, ny, nz, 4), dtype=float)

    for xv, yv, zv, col in zip(x_t, y_t, z_t, cols):
        if np.isnan(xv) or np.isnan(yv) or np.isnan(zv):
            continue
        ix = ix_map.get(xv)
        iy = iy_map.get(yv)
        iz = iz_map.get(zv)
        if ix is None or iy is None or iz is None:
            continue
        mask[ix, iy, iz] = True
        rgb = np.clip(col[:3], 0.0, 1.0)
        facecolors[ix, iy, iz, :3] = rgb
        facecolors[ix, iy, iz, 3] = 1.0

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(mask, facecolors=facecolors, edgecolor=edgecolor)

    # place ticks at voxel centers
    ax.set_xticks(np.arange(nx) + 0.5)
    ax.set_yticks(np.arange(ny) + 0.5)
    ax.set_zticks(np.arange(nz) + 0.5)
    ax.set_xticklabels([str(v) for v in ux], rotation=90, fontsize=8)
    ax.set_yticklabels([str(v) for v in uy], fontsize=8)
    ax.set_zticklabels([str(v) for v in uz], fontsize=8)

    if axis_names is None:
        axis_names = ['x', 'y', 'z']
    ax.set_xlabel(axis_names[0])
    ax.set_ylabel(axis_names[1])
    ax.set_zlabel(axis_names[2])

    # Ensure voxels appear square/cubic by setting box aspect to the physical extents
    span_x = np.ptp(ux) if ux.size > 1 else 1.0
    span_y = np.ptp(uy) if uy.size > 1 else 1.0
    span_z = np.ptp(uz) if uz.size > 1 else 1.0
    dx = (span_x / (nx - 1)) if nx > 1 else 1.0
    dy = (span_y / (ny - 1)) if ny > 1 else 1.0
    dz = (span_z / (nz - 1)) if nz > 1 else 1.0
    extent_x = dx * nx
    extent_y = dy * ny
    extent_z = dz * nz
    ax.set_box_aspect((extent_x, extent_y, extent_z))

    plt.tight_layout()
    return fig, ax
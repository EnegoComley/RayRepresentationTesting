import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm



class ColourPredictionPredictionNetwork(nn.Module):
    def __init__(self, latent_size):
        super().__init__()


        self.representation_encoder = nn.Sequential(nn.Linear(27, 128),
                                               nn.ReLU(),
                                               nn.Linear(128, latent_size),
                                               nn.ReLU())

        self.head = nn.Sequential(nn.Linear(latent_size + 3, 3))



    def forward(self, representation, direction):
        x = self.get_latent_representation(representation)
        return self.get_colour_from_latent(x, direction)

    def get_latent_representation(self, representation):
        return self.representation_encoder(representation)

    def get_colour_from_latent(self, latent_representation, direction):
        x = torch.cat((latent_representation, direction), dim=1)
        return self.head(x)

def plot_colored_voxels(coords, colours, axis_names=None, assume_normalized=True, figsize=(8, 8), edgecolor=None, axis_flips = (1, -1, 1)):
    """
    coords: tuple/list of three 1D torch tensors or numpy arrays (x_coords, y_coords, z_coords), length N
    colours: (N,3) torch tensor or numpy array with RGB values in 0..1
    Returns: (fig, ax)
    """

    # BGR to RGB
    #colours = torch.flip(colours, dims=[1])

    coords = [coords[i] * axis_flips[i] for i in range(len(coords))]
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


def _crop_to_cube(arr, mask):
    """Return cropped arr, cropped mask, and slices describing the cube region.
    The cube is the smallest cube that contains the bounding box of True in mask,
    expanded (centered) to equal side lengths and clamped to array bounds.
    """
    # find bounding box
    coords = np.where(mask)
    mins = [int(c.min()) for c in coords]
    maxs = [int(c.max()) for c in coords]
    lengths = [maxs[i] - mins[i] + 1 for i in range(3)]
    # desired cube size (cannot exceed array smallest dimension)
    cube_size = min(max(lengths), min(arr.shape))
    print(cube_size)
    # compute centered starts for each axis
    starts = []
    for i in range(3):
        center = (mins[i] + maxs[i]) // 2
        start = center - cube_size // 2
        # clamp start to valid range
        start = max(0, min(start, arr.shape[i] - cube_size))
        starts.append(int(start))
    slices = tuple(slice(starts[i], starts[i] + cube_size) for i in range(3))

    return arr[slices], mask[slices], slices

def plot_opacity_tensor(tensor, threshold=0.1, cmap='viridis', figsize=(8, 8), show=True):
    """
    Crop the 3D tensor to a cube containing all values >= threshold, then plot.
    Returns (fig, ax, cube_slices) where cube_slices is a 3-tuple of slice objects.
    """
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(tensor)

    if arr.ndim != 3:
        raise ValueError("Expected a 3D tensor/array")

    mask = arr >= threshold
    if not mask.any():
        print("No voxels above threshold.")
        return

    # Crop to cube containing all masked voxels
    arr_c, mask_c, cube_slices = _crop_to_cube(arr, mask)

    # Use raw values (clipped to [0,1]) for colormap and alpha
    clipped = np.clip(arr_c, 0.0, 1.0)
    cmap_obj = cm.get_cmap(cmap)
    rgba = cmap_obj(clipped)  # shape (X,Y,Z,4)

    facecolors = np.zeros(mask_c.shape + (4,), dtype=float)
    facecolors[mask_c] = rgba[mask_c]
    facecolors[..., 3] = np.where(mask_c, clipped, 0.0)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(mask_c, facecolors=facecolors, edgecolor='k')

    sx, sy, sz = mask_c.shape
    ax.set_xlim(0, sx)
    ax.set_ylim(0, sy)
    ax.set_zlim(0, sz)
    ax.set_box_aspect((sx, sy, sz))

    ax.set_xticks(np.arange(0, sx + 1))
    ax.set_yticks(np.arange(0, sy + 1))
    ax.set_zticks(np.arange(0, sz + 1))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if show:
        plt.show()
    return fig, ax, cube_slices


class GridRepresentationEncoder(nn.Module):
    def __init__(self, transformer_layers=1, representation_size=128, initial_dropout=0.1, n_patches=7*7*7):
        super().__init__()
        self.patch_encoder = nn.Sequential(nn.Dropout(initial_dropout),
                                      nn.Linear(representation_size, representation_size),
                                      nn.BatchNorm1d(representation_size),
                                      nn.ReLU(),
                                      nn.Linear(representation_size, representation_size),
                                      nn.BatchNorm1d(representation_size),
                                      nn.ReLU())

        #self.ray_encoder = ConvEncoder()



        transformer_layer = nn.TransformerEncoderLayer(d_model=representation_size, nhead=8, dropout=0.3, batch_first=True)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=transformer_layers)

        self.pos_encoding = nn.Parameter(torch.randn(n_patches, representation_size), requires_grad=True)

    def forward(self, representation):
        batch_size, n_patches, latent_size = representation.shape

        x = n_patches.view(batch_size * n_patches, -1)

        x = self.patch_encoder(x)
        x = x.view(batch_size, n_patches, -1)
        x = x + self.pos_encoding.unsqueeze(0).repeat(batch_size, 1, 1)
        x = self.transformer(x)

        del batch_size, n_patches, latent_size
        return x
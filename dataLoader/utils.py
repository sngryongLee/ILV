import torch, math
import numpy as np
from sklearn.cluster import KMeans
import nibabel as nib
import yaml
import re


def even_sample(N, n, start=None):
    base_stride = N // n
    remainder = N % n

    strides = [base_stride] * (n - remainder) + [base_stride + 1] * remainder
    np.random.shuffle(strides)

    start = np.random.randint(0, N) if start is None else start
    indices = [start]

    for stride in strides[:-1]:
        next_idx = (indices[-1] + stride) % N
        indices.append(next_idx)

    return indices

def load_projector_cfg(path):
    
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    proj_cfg = config["projector"]
    proj_cfg["sVoxel"] = (
        np.array(proj_cfg["dVoxel"]) * np.array(proj_cfg["nVoxel"])
    )
    proj_cfg["sDetector"] = (
        np.array(proj_cfg["dDetector"]) * np.array(proj_cfg["nDetector"])
    )

    scene_scale = 1.0 / max(proj_cfg["sVoxel"])
    keys = ["dVoxel", "sVoxel", "sDetector", "dDetector",
            "offOrigin", "offDetector", "DSD", "DSO"]

    for key in keys:
        proj_cfg[key] = (np.array(proj_cfg[key]) * scene_scale).tolist()

    proj_cfg["scene_scale"] = scene_scale * 1000
    return proj_cfg


def load_volume(path):
    nii_img = nib.load(path)
    return nii_img.get_fdata(dtype=np.float32).copy()


def load_projections(path):
    return np.load(path).astype(np.float32)


def normalize_projection(im):
    im = np.array(im[..., None]).astype(np.float32)
    im /= im.max() + 1e-8
    return im


def compute_fov(proj_cfg):
    sDet = np.array(proj_cfg["sDetector"])
    DSD = proj_cfg["DSD"]
    FovX = np.arctan2(sDet[1] / 2, DSD) * 2
    FovY = np.arctan2(sDet[0] / 2, DSD) * 2
    return FovX, FovY


def build_rays_torch(c2ws, ixts, H, W, scale=1.0):

    H, W = int(H*scale), int(W*scale)
    ixts[:,:2] *= scale
    rays_o = c2ws[:,:3, 3][:,None,None]
    X, Y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
    XYZ = torch.cat((X[:, :, None] + 0.5, Y[:, :, None] + 0.5, torch.ones_like(X[:, :, None])), dim=-1).to(c2ws)
    
    i2ws = torch.inverse(ixts).permute(0,2,1) @ c2ws[:,:3, :3].permute(0,2,1)
    XYZ = torch.stack([(XYZ @ i2w) for i2w in i2ws])
    rays_o = rays_o.repeat(1,H,1,1)
    rays_o = rays_o.repeat(1,1,W,1)
    rays = torch.cat((rays_o, XYZ), dim=-1)
    return rays


def build_rays(c2ws, ixts, H, W, scale=1.0):

    H, W = int(H*scale), int(W*scale)
    ixts[:,:2] *= scale

    rays_o = c2ws[:,:3, 3][:,None,None]
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    XYZ = np.concatenate((X[:, :, None] + 0.5, Y[:, :, None] + 0.5, np.ones_like(X[:, :, None])), axis=-1)
    i2ws = np.linalg.inv(ixts).transpose(0,2,1) @ c2ws[:,:3, :3].transpose(0,2,1)
    XYZ = np.stack([(XYZ @ i2w) for i2w in i2ws])
    rays_o = rays_o.repeat(H, axis=1)
    rays_o = rays_o.repeat(W, axis=2)
    rays = np.concatenate((rays_o, XYZ), axis=-1)
    return rays.astype(np.float32)


def build_rays_ortho(c2ws, H, W, scale=1.0):

    c2ws_rot = c2ws[:,:3,:3]
    c2ws_t = c2ws[:,:3,3].reshape(-1,1,3)
            
    rays_d = torch.zeros(1,1,3).to(c2ws)
    rays_d[...,-1] = 1.0
    rays_d = rays_d @ c2ws_rot.transpose(1,2)
    rays_d = rays_d[:,None].expand(-1,H,W,-1)
            
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    X = torch.from_numpy(X[:, :, None] + 0.5).float()/W * 2 - 1.0
    Y = torch.from_numpy(Y[:, :, None] + 0.5).float()/H * 2 - 1.0
    XYZ = torch.cat((X*scale, Y*scale, torch.zeros_like(X)), dim=-1).to(c2ws)
    XYZ = XYZ.view(1,-1,3)
    rays_o = XYZ @ c2ws_rot.transpose(1,2) + c2ws_t
    rays = torch.cat((rays_o.view(rays_d.shape), rays_d), dim=-1)
    return rays


def KMean(xyz, n_clusters):
    kmeans = KMeans(n_clusters = n_clusters, n_init=10, random_state=20211202)
    kmeans.fit(xyz)
    labels = kmeans.labels_
    
    clusters = []
    for i in range(n_clusters):
        idx = np.where(labels==i)[0]
        clusters.append(idx.astype(np.uint8))

    return clusters


def fov_to_ixt(fov, reso):
    ixt = np.eye(3, dtype=np.float32)
    ixt[0][2], ixt[1][2] = reso[0]/2, reso[1]/2
    focal = .5 * reso / np.tan(.5 * fov)
    ixt[[0,1],[0,1]] = focal
    return ixt


def intrinsic_to_fov(K, w=None, h=None):
    # Extract the focal lengths from the intrinsic matrix
    fx = K[0, 0]
    fy = K[1, 1]
    
    w = K[0, 2]*2 if w is None else w
    h = K[1, 2]*2 if h is None else h
    
    # Calculating field of view
    fov_x = 2 * np.arctan2(w, 2 * fx) 
    fov_y = 2 * np.arctan2(h, 2 * fy)
    
    return fov_x, fov_y


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


def pose_sub_selete(poses, N_poses):
    # to select N_poses poses that close to the equatorial
    # Define the camera's local up-vector and the world's up-vector
    camera_up_vector = np.array([0, 1, 0])
    world_up_vector = np.array([1, 0, 0])

    # Extract the rotation matrices from the transformation matrices
    rotations = poses[:, :3, :3]  # Shape: [num_poses, 3, 3]

    # Transform the camera's up-vector to world coordinates for all poses
    camera_up_world = np.einsum('ijk,k->ij', rotations, camera_up_vector)  # Shape: [num_poses, 3]

    # Normalize the transformed up-vectors
    camera_up_world_norm = camera_up_world / np.linalg.norm(camera_up_world, axis=1)[:, np.newaxis]

    # Calculate the dot product with the world up-vector to find cosines of angles
    cos_angles = np.dot(camera_up_world_norm, world_up_vector)

    # Calculate angles in radians using arccos
    angles = np.arccos(np.clip(cos_angles, -1.0, 1.0))

    # Select the poses with the smallest angles (closest to the equatorial plane)
    indices = np.argsort(angles)[:N_poses]
    
    return indices


def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale
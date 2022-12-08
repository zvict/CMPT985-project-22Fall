import os
import numpy as np
import imageio
import json
from PIL import Image
import torch
import math


def cam_to_world(vectors, c2w):
    """
        vectors: [N, H, W, 3] or [H, W, 3] or [K, 3]
        c2w: [N, 4, 4] or [4, 4]
    """
    if vectors.ndim == 4:
        assert c2w.ndim == 3
        N = c2w.shape[0]
        rotated_vec = torch.sum(vectors.unsqueeze(-2) * c2w[:, :3, :3].reshape(N, 1, 1, 3, 3), -1) # [N, H, W, 3]
    elif vectors.ndim == 3:
        assert c2w.ndim == 2
        H, W, _ = vectors.shape
        rotated_vec = torch.sum(vectors.unsqueeze(-2) * c2w[:3, :3].reshape(1, 1, 3, 3), -1)    # [H, W, 3]
    elif vectors.ndim == 2:
        assert c2w.ndim == 2
        K, _ = vectors.shape
        rotated_vec = torch.sum(vectors.unsqueeze(-2) * c2w[:3, :3].reshape(1, 3, 3), -1)   # [K, 3]
    else:
        raise ValueError('Wrong dimension of vectors')
    return rotated_vec


def world_to_cam(vectors, c2w):
    """
        vectors: [N, H, W, 3] or [H, W, 3] or [K, 3]
        c2w: [N, 4, 4] or [4, 4]
    """
    c2w = torch.inverse(c2w)
    if vectors.ndim == 4:
        assert c2w.ndim == 3
        N = c2w.shape[0]
        rotated_vec = torch.sum(vectors.unsqueeze(-2) * c2w[:, :3, :3].reshape(N, 1, 1, 3, 3), -1) # [N, H, W, 3]
    elif vectors.ndim == 3:
        assert c2w.ndim == 2
        H, W, _ = vectors.shape
        rotated_vec = torch.sum(vectors.unsqueeze(-2) * c2w[:3, :3].reshape(1, 1, 3, 3), -1)    # [H, W, 3]
    elif vectors.ndim == 2:
        assert c2w.ndim == 2
        K, _ = vectors.shape
        rotated_vec = torch.sum(vectors.unsqueeze(-2) * c2w[:3, :3].reshape(1, 3, 3), -1)   # [K, 3]
    else:
        raise ValueError('Wrong dimension of vectors')
    return rotated_vec


def get_rays(H, W, focal, c2w, fineness=1, coord='world'):
    N = c2w.shape[0]
    width = torch.linspace(0, W / focal, steps=int(W / fineness) + 1, dtype=torch.float32)
    height = torch.linspace(0, H / focal, steps=int(H / fineness) + 1, dtype=torch.float32)
    y, x = torch.meshgrid(height, width)
    pixel_size_x = width[1] - width[0]
    pixel_size_y = height[1] - height[0]
    x = (x - W / focal / 2 + pixel_size_x / 2)[:-1, :-1]
    y = -(y - H / focal / 2 + pixel_size_y / 2)[:-1, :-1]
    dirs_d = torch.stack([x, y, -torch.ones_like(x)], -1)   # [H, W, 3], vectors, since the camera is at the origin
    if coord == 'world':
        # rays_d = torch.sum(dirs_d.reshape(1, H, W, 1, 3) * c2w[:, :3, :3].reshape(N, 1, 1, 3, 3), -1) # [N, H, W, 3]
        rays_d = cam_to_world(dirs_d.unsqueeze(0), c2w)  # [N, H, W, 3]
        rays_o = c2w[:, :3, -1]       # [N, 3]
    elif coord == 'cam':
        rays_d = dirs_d.reshape(1, H, W, 3).repeat(N, 1, 1, 1)  # [N, H, W, 3]
        rays_o = torch.zeros(N, 3, dtype=torch.float32)
    return rays_o, rays_d


def extract_patches(imgs, normals, rays_o, rays_d, args):
    patch_opt = args.patches
    N, H, W, C = normals.shape

    img_patches = None
    normal_patches = None

    if patch_opt.type == "continuous":
        num_patches_H = math.ceil((H - patch_opt.overlap) / (patch_opt.height - patch_opt.overlap))
        num_patches_W = math.ceil((W - patch_opt.overlap) / (patch_opt.width - patch_opt.overlap))
        num_patches = num_patches_H * num_patches_W
        rayd_patches = np.zeros((N, num_patches, patch_opt.height, patch_opt.width, 3), dtype=np.float32)
        rayo_patches = np.zeros((N, num_patches, 3), dtype=np.float32)
        if normals is not None:
            norm_patches = np.zeros((N, num_patches, patch_opt.height, patch_opt.width, 3), dtype=np.float32)
        if imgs is not None:
            img_patches = np.zeros((N, num_patches, patch_opt.height, patch_opt.width, C), dtype=np.float32)

        for i in range(N):
            n_patch = 0
            for start_height in range(0, H - patch_opt.overlap, patch_opt.height - patch_opt.overlap):
                for start_width in range(0, W - patch_opt.overlap, patch_opt.width - patch_opt.overlap):
                    end_height = min(start_height + patch_opt.height, H)
                    end_width = min(start_width + patch_opt.width, W)
                    start_height = end_height - patch_opt.height
                    start_width = end_width - patch_opt.width
                    rayd_patches[i, n_patch, :, :] = rays_d[i, start_height:end_height, start_width:end_width]
                    rayo_patches[i, n_patch, :] = rays_o[i, :]
                    if normals is not None:
                        norm_patches[i, n_patch, :, :] = normals[i, start_height:end_height, start_width:end_width]
                    if imgs is not None:
                        img_patches[i, n_patch, :, :] = imgs[i, start_height:end_height, start_width:end_width]
                    n_patch += 1

    elif patch_opt.type == "random":
        num_patches = patch_opt.max_patches
        rayd_patches = np.zeros((N, num_patches, patch_opt.height, patch_opt.width, 3), dtype=np.float32)
        rayo_patches = np.zeros((N, num_patches, 3), dtype=np.float32)
        if normals is not None:
            norm_patches = np.zeros((N, num_patches, patch_opt.height, patch_opt.width, 3), dtype=np.float32)
        if imgs is not None:
            img_patches = np.zeros((N, num_patches, patch_opt.height, patch_opt.width, C), dtype=np.float32)

        for i in range(N):
            for n_patch in range(num_patches):
                start_height = np.random.randint(0, H - patch_opt.height)
                start_width = np.random.randint(0, W - patch_opt.width)
                end_height = start_height + patch_opt.height
                end_width = start_width + patch_opt.width
                rayd_patches[i, n_patch, :, :] = rays_d[i, start_height:end_height, start_width:end_width]
                rayo_patches[i, n_patch, :] = rays_o[i, :]
                if normals is not None:
                    norm_patches[i, n_patch, :, :] = normals[i, start_height:end_height, start_width:end_width]
                if imgs is not None:
                    img_patches[i, n_patch, :, :] = imgs[i, start_height:end_height, start_width:end_width]

    return img_patches, rayd_patches, rayo_patches, norm_patches, num_patches
    

def load_meta_data(args):
    """
    0 -----------> W
    |
    |
    |
    ⬇
    H
    [H, W, 4]
    """

    normals = None
    
    if args.type == "synthetic":
        images, poses, hwf = load_blender_data(args.path, factor=args.factor)
        print('Loaded blender', images.shape, hwf, args.path)
        
        if args.white_bg:
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]

        normals = images

    else:
        data = np.load(args.path)
        images = data['images']
        poses = data['poses']
        focal = data['focal']
        if "200" in args.path:
            focal *= 2
        elif "800" in args.path:
            focal *= 8
        H, W = images.shape[1:3]
        i_test = np.arange(100, int(images.shape[0]))
        i_val = i_test
        hwf = [H, W, focal]

    H, W, focal = hwf

    if images is not None:
        images = torch.from_numpy(images).float()
    poses = torch.from_numpy(poses).float()
    if normals is not None:
        normals = torch.from_numpy(normals).float()
        if args.bg_norm_cam_coord:
            bg_norm = torch.tensor(args.bg_norm_cam_coord, dtype=torch.float32)
            bg_norm = torch.matmul(poses[:, :3, :3], bg_norm)
            mask = normals[:, :, :, 3].unsqueeze(-1)
            normals = mask * normals[:, :, :, :3] + (1 - mask) * bg_norm.reshape(-1, 1, 1, 3)
        normals = normals[:, :, :, :3]
        if args.normalize_norm:
            normals = normals / torch.norm(normals, dim=-1, keepdim=True)

    return images, poses, normals, H, W, focal


def load_blender_data(basedir, testskip=1, factor=1):
    with open(os.path.join(basedir, 'transforms.json'), 'r') as fp:
        meta = json.load(fp)

    poses = []
    images = []

    dir_path = os.path.dirname(os.path.realpath(__file__))

    for i, frame in enumerate(meta['frames']):
        img = imageio.imread(os.path.join(basedir, frame['file_path'] + '.png'))
        W, H = img.shape[:2]
        if factor > 1:
            img = Image.fromarray(img).resize((W//factor, H//factor))
        poses.append(np.array(frame['transform_matrix']))
        images.append((np.array(img) / 255.).astype(np.float32))
    poses = np.array(poses).astype(np.float32)
    images = np.array(images).astype(np.float32)

    H, W = images[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    return images, poses, [H, W, focal]


def rgb2norm(img):
    norm_vec = np.stack([img[..., 0] * 2.0 / 255.0 - 1.0, 
                         img[..., 1] * 2.0 / 255.0 - 1.0, 
                         img[..., 2] * 2.0 / 255.0 - 1.0, 
                         img[..., 3] / 255.0], axis=-1)
    return norm_vec

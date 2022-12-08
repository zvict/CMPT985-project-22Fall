import torch
from torch.utils.data import Dataset
import numpy as np
import imageio
import json
from PIL import Image
from .utils import load_meta_data, get_rays, extract_patches, cam_to_world, world_to_cam


class RINDataset(Dataset):
    """ Ray Image Normal Dataset """
    def __init__(self, args):
        self.args = args
        images, c2w, normals, H, W, focal = load_meta_data(args)
        focal = focal / args.rays.focal_factor
        rays_o, rays_d = get_rays(H, W, focal, c2w, coord=args.rays.cam_world)

        if args.cam_world == 'cam':
            normals = world_to_cam(normals, c2w)

        self.images = images    # (N, H, W, C) or None
        self.rayd = rays_d      # (N, H, W, 3)
        self.rayo = rays_o      # (N, 3)
        self.normals = normals  # (N, H, W, 3) or None
        self.c2w = c2w      # (N, 4, 4)

        self.num_imgs = rays_d.shape[0]

        if self.args.extract_patch == True:
            img_patches, rayd_patches, rayo_patches, norm_patches, num_patches = extract_patches(images, normals, rays_o, rays_d, args)
            self.img_patches = img_patches      # (N, n_patches, patch_height, patch_width, C) or None
            self.rayd_patches = rayd_patches    # (N, n_patches, patch_height, patch_width, 3)
            self.rayo_patches = rayo_patches    # (N, n_patches, 3)
            self.norm_patches = norm_patches    # (N, n_patches, patch_height, patch_width, 3) or None

            self.num_patches = num_patches

    def __len__(self):
        if self.args.extract_patch == True:
            return self.num_imgs * self.num_patches
        else:
            return self.num_imgs

    def __getitem__(self, idx):
        if self.args.extract_patch == True:
            img_idx = idx // self.num_patches
            patch_idx = idx % self.num_patches
            return img_idx, patch_idx, \
                    self.img_patches[img_idx, patch_idx] if self.img_patches is not None else 0, \
                    self.rayd_patches[img_idx, patch_idx], \
                    self.rayo_patches[img_idx, patch_idx], \
                    self.norm_patches[img_idx, patch_idx] if self.norm_patches is not None else 0
        else:
            return idx, self.images[idx] if self.images is not None else 0, \
                    self.rayd[idx], self.rayo[idx], \
                    self.normals[idx] if self.normals is not None else 0

    def get_full_img(self, img_idx):
        return self.images[img_idx].unsqueeze(0) if self.images is not None else None, \
                self.rayd[img_idx].unsqueeze(0), self.rayo[img_idx].unsqueeze(0), \
                self.normals[img_idx].unsqueeze(0) if self.normals is not None else None

    def get_c2w(self, img_idx):
        return self.c2w[img_idx]


class MultiSceneRINDataset(Dataset):
    """ Ray Image Normal Dataset """
    def __init__(self, args):
        self.args = args
        images, c2w, normals, H, W, focal = load_meta_data(args)
        focal = focal / args.rays.focal_factor
        rays_o, rays_d = get_rays(H, W, focal, c2w, coord=args.rays.cam_world)

        if args.cam_world == 'cam':
            normals = world_to_cam(normals, c2w)

        self.images = images    # (N, H, W, C) or None
        self.rayd = rays_d      # (N, H, W, 3)
        self.rayo = rays_o      # (N, 3)
        self.normals = normals  # (N, H, W, 3) or None
        self.c2w = c2w      # (N, 4, 4)

        self.num_imgs = rays_d.shape[0]

        if self.args.extract_patch == True:
            img_patches, rayd_patches, rayo_patches, norm_patches, num_patches = extract_patches(images, normals, rays_o, rays_d, args)
            self.img_patches = img_patches      # (N, n_patches, patch_height, patch_width, C) or None
            self.rayd_patches = rayd_patches    # (N, n_patches, patch_height, patch_width, 3)
            self.rayo_patches = rayo_patches    # (N, n_patches, 3)
            self.norm_patches = norm_patches    # (N, n_patches, patch_height, patch_width, 3) or None

            self.num_patches = num_patches

    def __len__(self):
        if self.args.extract_patch == True:
            return self.num_imgs * self.num_patches
        else:
            return self.num_imgs

    def __getitem__(self, idx):
        if self.args.extract_patch == True:
            img_idx = idx // self.num_patches
            patch_idx = idx % self.num_patches
            return img_idx, patch_idx, \
                    self.img_patches[img_idx, patch_idx] if self.img_patches is not None else 0, \
                    self.rayd_patches[img_idx, patch_idx], \
                    self.rayo_patches[img_idx, patch_idx], \
                    self.norm_patches[img_idx, patch_idx] if self.norm_patches is not None else 0
        else:
            return idx, self.images[idx] if self.images is not None else 0, \
                    self.rayd[idx], self.rayo[idx], \
                    self.normals[idx] if self.normals is not None else 0

    def get_full_img(self, img_idx):
        return self.images[img_idx].unsqueeze(0) if self.images is not None else None, \
                self.rayd[img_idx].unsqueeze(0), self.rayo[img_idx].unsqueeze(0), \
                self.normals[img_idx].unsqueeze(0) if self.normals is not None else None

    def get_c2w(self, img_idx):
        return self.c2w[img_idx]

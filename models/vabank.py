import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import math
import os
import numpy as np
from .utils import normalize_vector, posenc, create_learning_rate_fn, cam_to_world, world_to_cam
from .mlp import get_attn_mlp, get_norm_mlp
from .transformer import get_transformer, rate


class VolumetricBank(nn.Module):
    def __init__(self, args):
        super(VolumetricBank, self).__init__()
        self.args = args

        point_opt = args.geoms.points
        if point_opt.load_path:
            points = torch.load(point_opt.load_path)
        else:
            if point_opt.init_type == 'sphere':
                points = self._sphere_pc(point_opt.init_center, point_opt.num, point_opt.init_scale)
            elif point_opt.init_type == 'qube':
                points = self._qube_pc(point_opt.init_center, point_opt.num, point_opt.init_scale)
            else:
                raise NotImplementedError("Point init type [{:s}] is not found".format(point_opt.init_type))
        self.points = torch.nn.Parameter(points, requires_grad=True)

        pc_norm_opt = args.geoms.point_norms
        if pc_norm_opt.load_path:
            pc_norms = torch.load(pc_norm_opt.load_path)
        else:
            pc_norms = normalize_vector(points - torch.tensor(point_opt.init_center, device=points.device).float())
        self.pc_norms = torch.nn.Parameter(pc_norms, requires_grad=True)

        pc_feat_opt = args.geoms.point_feats
        if pc_feat_opt.load_path:
            pc_feats = torch.load(pc_feat_opt.load_path)
        else:
            pc_feats = torch.randn((point_opt.num, pc_feat_opt.dim), device=points.device)
        self.pc_feats = torch.nn.Parameter(pc_feats, requires_grad=True)

        self.tx_out_dim = args.models.transformer.dim * (int(args.models.transformer.concat) * (args.models.transformer.num_layers - 1) + 1)

        """ MLPs 
            The RGB MLP and the Attn MLP in our model are defined here
            The RGB MLP: (norm_mlp)
            The Attn MLP: (attn_mlp)
        """
        if args.models.use_attn_mlp:
            self.attn_mlp = get_attn_mlp(args.models)
        self.norm_mlp = get_norm_mlp(args.models)

        v_extra_dim = 0
        kq_extra_dim = 0
        value_type = args.models.transformer.value_type
        self.v_extra_dim = v_extra_dim
        self.kq_extra_dim = kq_extra_dim


        """ Transformer
            The transformer in our model is defined here
        """
        transformer, kq_dim, value_dim, tx_kq_dim, tx_v_dim, n_kernel = get_transformer(args.models.transformer, 
                                                                                        v_extra_dim=v_extra_dim, 
                                                                                        kq_extra_dim=kq_extra_dim,
                                                                                        N_sample=self.args.geoms.num_sample)
        self.transformer = transformer
        self.kq_dim = kq_dim
        self.value_dim = value_dim
        self.tx_kq_dim = tx_kq_dim
        self.tx_v_dim = tx_v_dim
        self.n_kernel = n_kernel

        self._init_optimizers()


    def _init_optimizers(self):
        lr_opt = self.args.training.lr
        optimizer_points = torch.optim.Adam([self.points], lr=lr_opt.points)
        optimizer_pc_norms = torch.optim.Adam([self.pc_norms], lr=lr_opt.norms)
        optimizer_pc_feats = torch.optim.Adam([self.pc_feats], lr=lr_opt.feats)
        optimizer_norm_mlp = torch.optim.Adam(self.norm_mlp.parameters(), lr=lr_opt.norm_mlp.base_lr, weight_decay=lr_opt.norm_mlp.weight_decay)
        optimizer_tx = torch.optim.Adam(self.transformer.parameters(), lr=lr_opt.transformer.base_lr, weight_decay=lr_opt.transformer.weight_decay)

        debug = False
        lr_scheduler_norm_mlp = create_learning_rate_fn(optimizer_norm_mlp, self.args.training.steps, lr_opt.norm_mlp, debug=debug)
        lr_scheduler_tx = create_learning_rate_fn(optimizer_tx, self.args.training.steps, lr_opt.transformer, debug=debug)

        self.optimizers = {
            "points": optimizer_points,
            "pc_norms": optimizer_pc_norms,
            "pc_feats": optimizer_pc_feats,
            "norm_mlp": optimizer_norm_mlp,
            "transformer": optimizer_tx,
        }

        self.schedulers = {
            "norm_mlp": lr_scheduler_norm_mlp,
            "transformer": lr_scheduler_tx,
        }

        if self.args.models.use_attn_mlp:
            optimizer_attn_mlp = torch.optim.Adam(self.attn_mlp.parameters(), lr=lr_opt.attn_mlp.base_lr, weight_decay=lr_opt.attn_mlp.weight_decay)
            lr_scheduler_attn_mlp = create_learning_rate_fn(optimizer_attn_mlp, self.args.training.steps, lr_opt.attn_mlp, debug=debug)

            self.optimizers["attn_mlp"] = optimizer_attn_mlp
            self.schedulers["attn_mlp"] = lr_scheduler_attn_mlp
        

    def _sphere_pc(self, center, num_pts, scale):
        xs, ys, zs = [], [], []
        phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians
        for i in range(num_pts):
            y = 1 - (i / float(num_pts - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at y
            theta = phi * i  # golden angle increment
            x = math.cos(theta) * radius
            z = math.sin(theta) * radius
            xs.append(x * scale[0] + center[0])
            ys.append(y * scale[1] + center[1])
            zs.append(z * scale[2] + center[2])
        points = np.stack([np.array(xs), np.array(ys), np.array(zs)], axis=-1)
        return torch.from_numpy(points).float()

    
    def _qube_pc(self, center, num_pts, scale):
        xs = np.random.uniform(-scale[0], scale[0], num_pts) + center[0]
        ys = np.random.uniform(-scale[1], scale[1], num_pts) + center[1]
        zs = np.random.uniform(-scale[2], scale[2], num_pts) + center[2]
        points = np.stack([np.array(xs), np.array(ys), np.array(zs)], axis=-1)
        return torch.from_numpy(points).float()


    def _get_points(self, c2w):
        if self.args.geoms.points.cam_world == 'cam':
            points = world_to_cam(self.points, c2w, vector=False)
        else:
            points = self.points
        return points

    
    def _get_pc_norms(self, c2w):
        if self.args.geoms.point_norms.cam_world == 'cam':
            pc_norms = world_to_cam(self.pc_norms, c2w)
        else:
            pc_norms = self.pc_norms
        return pc_norms

    
    def _get_kqv(self, rays_o, rays_d, points, pc_norms):
        N, H, W, _ = rays_d.shape
        num_pts, _ = points.shape

        """ Ray marching: The key part of this project
            Rays: defined by the origin or ray (rays_o) and the direction of the ray (rays_d)
            Sampled points along the ray: (sample_points)
            Point cloud: (points)
        """
        near = torch.norm(rays_o) - 0.5
        far = near + 1.0
        N_sample = self.args.geoms.num_sample
        z_vals = torch.linspace(near, far, N_sample, device=points.device)
        sample_points = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        # points2sample = torch.norm(points.reshape(1, 1, 1, num_pts, 1, 3) - sample_points.reshape(N, H, W, 1, -1, 3), dim=-1)
        points2sample = points.reshape(1, 1, 1, num_pts, 1, 3) - sample_points.reshape(N, H, W, 1, -1, 3)
        # print(near, far, sample_points.shape, points2sample.shape)

        kq_type = self.args.models.transformer.kq_type
        kq_L = self.args.models.transformer.kq_L
        if kq_type == 30:
            kq = torch.norm(points2sample, dim=-1)
            kernels = {
                0: (N_sample+N_sample*2*kq_L, 'dot'),
            }
        elif kq_type == 31:
            kq = points2sample.flatten(-2)
            kernels = {
                0: (N_sample+N_sample*2*kq_L, 'dot'),
            }
        else:
            raise ValueError('Invalid kq type')
        assert self.kq_dim == kq.shape[-1]
        # assert self.n_kernel == len(kernels)

        value_type = self.args.models.transformer.value_type
        if value_type == 23:
            value = torch.norm(points2sample, dim=-1)
        elif value_type == 24:
            value = points2sample.flatten(-2)
        else:
            raise ValueError('Invalid value type')
        assert self.value_dim == value.shape[-1]

        kq = posenc(kq.reshape(-1, self.kq_dim), self.args.models.transformer.kq_L).reshape(-1, num_pts, self.tx_kq_dim - self.kq_extra_dim)
        value = posenc(value.reshape(-1, self.value_dim), self.args.models.transformer.value_L).reshape(-1, num_pts, self.tx_v_dim - self.v_extra_dim)

        value *= self.args.models.transformer.vtemp

        if self.args.models.use_attn_mlp:
            attn_mlp_type = self.args.models.attn_mlp.type
            if attn_mlp_type in [3]:
                attn_input = None
            else:
                raise ValueError('Invalid attn mlp type')
        else:
            raise ValueError('Must use attn mlp')

        return kq, value, kernels, attn_input


    def _get_attn(self, attn_input):
        proj_dists, dists_to_rays, ray_norm_inner = attn_input
        
        sftmax = torch.nn.Softmax(dim=3)
        attn_opt = self.args.models.attn_fn
        if attn_opt.type == 1:
            attn = sftmax(-(proj_dists * attn_opt.cpd + dists_to_rays * attn_opt.cd2r))
        elif attn_opt.type == 2:
            attn = sftmax(-(proj_dists * attn_opt.cpd + dists_to_rays * attn_opt.cd2r) / (attn_opt.crni + torch.sigmoid(ray_norm_inner * attn_opt.csig)))
        else:
            raise ValueError('Invalid attn type')

        return attn


    def clear_grad(self):
        for name, optimizer in self.optimizers.items():
            optimizer.zero_grad()


    def step(self):
        for name, optimizer in self.optimizers.items():
            optimizer.step()

        for name, scheduler in self.schedulers.items():
            if scheduler is not None:
                scheduler.step()

        self.tx_lr = self.schedulers['transformer'].get_last_lr()[0]

    
    def _norm_layer(self, inp, act_type='tanh') :
        if self.args.models.norm_mlp.out_type == 'norm':
            return normalize_vector(inp)
        elif self.args.models.norm_mlp.out_type == 'rgb':
            return RerangeLayer(act_type)(inp)
        else:
            raise ValueError('Invalid norm mlp out type')


    def evaluate(self, rays_o, rays_d, c2w, step=-1):
        # print("start evaluate")
        points = self._get_points(c2w)
        pc_norms = self._get_pc_norms(c2w)
        kq, value, kernels, attn_input = self._get_kqv(rays_o, rays_d, points, pc_norms)
        N, H, W, _ = rays_d.shape
        num_pts, _ = points.shape

        # print("evaluating")

        encode = self.transformer(kq, kq, value, kernels)    # (N, H, W, num_pts, tx_dim)
        norm = self.norm_mlp(encode.reshape(-1, self.tx_out_dim)).reshape(N, H, W, num_pts, -1)
        norm = self._norm_layer(norm)

        # print("evaluated")

        if self.args.models.use_attn_mlp:
            if self.args.models.attn_mlp.type in [3]:
                attn_input = encode.reshape(-1, self.tx_out_dim)
            attn = self.attn_mlp(N, H, W, attn_input)   # (N, H, W, num_pts, 1)
        else:
            attn = self._get_attn(attn_input)

        if self.args.models.out_fuse_type == 1:
            out = torch.sum(norm * attn, dim=3)
        elif self.args.models.out_fuse_type == 2:
            out = torch.mean(norm, dim=3)
        else:
            raise ValueError('Invalid out fuse type')

        return encode.reshape(N, H, W, num_pts, -1), norm[..., :3], attn, out


    def forward(self, rays_o, rays_d, c2w, step=-1):
        points = self._get_points(c2w)
        pc_norms = self._get_pc_norms(c2w)
        kq, value, kernels, attn_input = self._get_kqv(rays_o, rays_d, points, pc_norms)
        N, H, W, _ = rays_d.shape
        num_pts, _ = points.shape

        # for step in range(10000):
        #     if step % 100 == 0:
        #         print(step, rate(step, self.args.models.transformer.dim, self.args.training.lr.transformer.factor, self.args.training.lr.transformer.warmup))

        encode = self.transformer(kq, kq, value, kernels)    # (N, H, W, num_pts, tx_dim)
        norm = self.norm_mlp(encode.reshape(-1, self.tx_out_dim)).reshape(N, H, W, num_pts, -1)
        norm = self._norm_layer(norm)

        if self.args.models.use_attn_mlp:
            if self.args.models.attn_mlp.type in [3]:
                attn_input = encode.reshape(-1, self.tx_out_dim)
            attn = self.attn_mlp(N, H, W, attn_input)   # (N, H, W, num_pts, 1)
        else:
            attn = self._get_attn(attn_input)

        if self.args.models.out_fuse_type == 1:
            out = torch.sum(norm * attn, dim=3)
        elif self.args.models.out_fuse_type == 2:
            out = torch.mean(norm, dim=3)
        else:
            raise ValueError('Invalid out fuse type')

        if step >= 0 and step % 1000 == 0:
            print('encode:', encode.shape, encode.min().item(), encode.max().item(), encode.mean().item(), encode.std().item())
            print('norm:', norm.shape, norm.min().item(), norm.max().item(), norm.mean().item(), norm.std().item())
            print('attn:', attn.shape, attn.min().item(), attn.max().item(), attn.mean().item(), attn.std().item())
            print('out:', out.shape, out.min().item(), out.max().item(), out.mean().item(), out.std().item())

        return out

    
    def get_ray_norm_inner(self, rays_d):
        assert self.args.geoms.point_norms.cam_world == 'world'
        assert self.args.dataset.rays.cam_world == 'world'
        return torch.sum(rays_d.unsqueeze(2) * self.pc_norms.reshape(1, 1, -1, 3), dim=-1, keepdim=True)


    def save(self, step, save_dir):
        torch.save({str(step): self.state_dict()}, os.path.join(save_dir, 'model.pth'))

        optimizers_state_dict = {}
        for name, optimizer in self.optimizers.items():
            optimizers_state_dict[name] = optimizer.state_dict()
        torch.save(optimizers_state_dict, os.path.join(save_dir, 'optimizers.pth'))

        schedulers_state_dict = {}
        for name, scheduler in self.schedulers.items():
            if scheduler is not None:
                schedulers_state_dict[name] = scheduler.state_dict()
            else:
                schedulers_state_dict[name] = None
        torch.save(schedulers_state_dict, os.path.join(save_dir, 'schedulers.pth'))

    
    def load(self, load_dir):
        optimizers_state_dict = torch.load(os.path.join(load_dir, 'optimizers.pth'))
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(optimizers_state_dict[name])
        
        schedulers_state_dict = torch.load(os.path.join(load_dir, 'schedulers.pth'))
        for name, scheduler in self.schedulers.items():
            if scheduler is not None:
                scheduler.load_state_dict(schedulers_state_dict[name])
            else:
                assert schedulers_state_dict[name] is None

        model_state_dict = torch.load(os.path.join(load_dir, 'model.pth'))
        for step, state_dict in model_state_dict.items():
            self.load_state_dict(state_dict)
            return int(step)

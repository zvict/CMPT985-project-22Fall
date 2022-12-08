import torch
from torch import nn
import torch.optim.lr_scheduler as lr_scheduler


def cam_to_world(coords, c2w, vector=True):
    """
        coords: [N, H, W, 3] or [H, W, 3] or [K, 3]
        c2w: [N, 4, 4] or [4, 4]
    """
    if vector:  # Convert to homogeneous coordinates
        coords = torch.cat([coords, torch.zeros_like(coords[..., :1])], -1)
    else:
        coords = torch.cat([coords, torch.ones_like(coords[..., :1])], -1)

    if coords.ndim == 4:
        assert c2w.ndim == 3
        N, H, W, _ = coords.shape
        transformed_coords = torch.sum(coords.unsqueeze(-2) * c2w.reshape(N, 1, 1, 4, 4), -1) # [N, H, W, 3]
    elif coords.ndim == 3:
        assert c2w.ndim == 2
        H, W, _ = coords.shape
        transformed_coords = torch.sum(coords.unsqueeze(-2) * c2w.reshape(1, 1, 4, 4), -1)    # [H, W, 3]
    elif coords.ndim == 2:
        assert c2w.ndim == 2
        K, _ = coords.shape
        transformed_coords = torch.sum(coords.unsqueeze(-2) * c2w.reshape(1, 4, 4), -1)   # [K, 3]
    else:
        raise ValueError('Wrong dimension of coords')
    return transformed_coords[:, :3]


def world_to_cam(coords, c2w, vector=True):
    """
        coords: [N, H, W, 3] or [H, W, 3] or [K, 3]
        c2w: [N, 4, 4] or [4, 4]
    """
    if vector:  # Convert to homogeneous coordinates
        coords = torch.cat([coords, torch.zeros_like(coords[..., :1])], -1)
    else:
        coords = torch.cat([coords, torch.ones_like(coords[..., :1])], -1)

    c2w = torch.inverse(c2w)
    if coords.ndim == 4:
        assert c2w.ndim == 3
        N, H, W, _ = coords.shape
        transformed_coords = torch.sum(coords.unsqueeze(-2) * c2w.reshape(N, 1, 1, 4, 4), -1) # [N, H, W, 3]
    elif coords.ndim == 3:
        assert c2w.ndim == 2
        H, W, _ = coords.shape
        transformed_coords = torch.sum(coords.unsqueeze(-2) * c2w.reshape(1, 1, 4, 4), -1)    # [H, W, 3]
    elif coords.ndim == 2:
        assert c2w.ndim == 2
        K, _ = coords.shape
        transformed_coords = torch.sum(coords.unsqueeze(-2) * c2w.reshape(1, 4, 4), -1)   # [K, 3]
    else:
        raise ValueError('Wrong dimension of coords')
    return transformed_coords[:, :3]


def activation_func(act_type='leakyrelu', neg_slope=0.2, inplace=True, num_channels=128, a=1., b=1., trainable=False):
    act_type = act_type.lower()
    if act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_channels)
    elif act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'tanh':
        layer = nn.Tanh()
    elif act_type == 'sigmoid':
        layer = nn.Sigmoid()
    elif act_type == 'softplus':
        layer = nn.Softplus()
    elif act_type == 'gelu':
        layer = nn.GELU()
    elif act_type == 'gaussian':
        layer = GaussianActivation(a, trainable)
    elif act_type == 'quadratic':
        layer = QuadraticActivation(a, trainable)
    elif act_type == 'multi-quadratic':
        layer = MultiQuadraticActivation(a, trainable)
    elif act_type == 'laplacian':
        layer = LaplacianActivation(a, trainable)
    elif act_type == 'super-gaussian':
        layer = SuperGaussianActivation(a, b, trainable)
    elif act_type == 'expsin':
        layer = ExpSinActivation(a, trainable)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


def posenc(x, L_embed):
    rets = [x]
    for i in range(L_embed):
        for fn in [torch.sin, torch.cos]:
            rets.append(fn(2.**i * x))
    # return torch.cat(rets, 1)
    return torch.flatten(torch.stack(rets, -1), start_dim=-2, end_dim=-1)   # To make sure the dimensions of the same meaning are together


def normalize_vector(x):
    # assert(x.shape[-1] == 3)
    return x / torch.norm(x, dim=-1, keepdim=True)


def create_learning_rate_fn(optimizer, max_steps, args, debug=False):
    """Create learning rate schedule."""
    # Linear warmup
    # print(args)
    if args.type == "none":
        return None

    warmup_fn = lr_scheduler.LinearLR(optimizer,
                                      start_factor=1e-16,
                                      end_factor=1.0,
                                      total_iters=args.warmup,
                                      verbose=debug)

    if args.type == "linear":
        decay_fn = lr_scheduler.LinearLR(optimizer,
                                        start_factor=1.0,
                                        end_factor=0.,
                                        total_iters=max_steps - args.warmup,
                                        verbose=debug)
    elif args.type == "cosine":
        cosine_steps = max(max_steps - args.warmup, 1)
        decay_fn = lr_scheduler.CosineAnnealingLR(optimizer,
                                                    T_max=cosine_steps,
                                                    verbose=debug)
    else:
        raise NotImplementedError

    schedule_fn = lr_scheduler.SequentialLR(optimizer, 
                                            schedulers=[warmup_fn, decay_fn],
                                            milestones=[args.warmup],
                                            verbose=debug)
    return schedule_fn


class GaussianActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-x**2/(2*self.a**2))


class QuadraticActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return 1/(1+(self.a*x)**2)


class MultiQuadraticActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return 1/(1+(self.a*x)**2)**0.5


class LaplacianActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-torch.abs(x)/self.a)


class SuperGaussianActivation(nn.Module):
    def __init__(self, a=1., b=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))
        self.register_parameter('b', nn.Parameter(b*torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-x**2/(2*self.a**2))**self.b


class ExpSinActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-torch.sin(self.a*x))
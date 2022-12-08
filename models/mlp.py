import torch
from torch import nn
from torch.nn.utils import weight_norm
from .utils import activation_func, posenc


def get_pt_embed_mlp(args):
    if args.pt_embed_mlp.type in [1, 2]:
        inp_dim = 3
        out_dim = args.pt_embed_mlp.out_dim
    else:
        raise NotImplementedError('pt embed mlp type [{:d}] is not supported'.format(args.type))
    return PTMLP(args.pt_embed_mlp, inp_dim=inp_dim, out_dim=out_dim)


def get_w_mlp(args):
    if args.type in [8, 9]:
        inp_dim = 3
        out_dim = 1
    else:
        raise NotImplementedError('w mlp type [{:d}] is not supported'.format(args.type))
    return WMLP(args, inp_dim=inp_dim, out_dim=out_dim)


def get_attn_mlp(args):
    if args.attn_mlp.type == 1:
        inp_dim = 2
        out_dim = 2
    elif args.attn_mlp.type == 2:
        inp_dim = 2
        out_dim = 1
    elif args.attn_mlp.type == 3:
        inp_dim = args.transformer.dim * (int(args.transformer.concat) * (args.transformer.num_layers - 1) + 1)
        out_dim = 1
    else:
        raise NotImplementedError('attention mlp type [{:d}] is not supported'.format(args.type))
    return AttnMLP(args.attn_mlp, inp_dim=inp_dim, out_dim=out_dim)


def get_norm_mlp(args):
    if args.norm_mlp.type == 1:
        inp_dim = args.transformer.dim * (int(args.transformer.concat) * (args.transformer.num_layers - 1) + 1)
    else:
        raise NotImplementedError('normal mlp type [{:d}] is not supported'.format(args.type))
    return NormMLP(args.norm_mlp, inp_dim=inp_dim, out_dim=args.norm_mlp.out_dim)


class MLP(nn.Module):
    def __init__(self, inp_dim=2, num_layers=3, num_channels=128, out_dim=2, act_type="leakyrelu", last_act_type="none", 
                    use_wn=True, a=1., b=1., trainable=False):
        super(MLP, self).__init__()
        if num_layers == 0:
            if last_act_type == "none":
                layers = [nn.Identity()]
            else:
                layers = [activation_func(act_type=last_act_type, num_channels=inp_dim, a=a, b=b, trainable=trainable)]
        elif num_layers > 1:
            c_act = activation_func(act_type=act_type, num_channels=num_channels, a=a, b=b, trainable=trainable)
            if use_wn:
                layers = [weight_norm(nn.Linear(inp_dim, num_channels), name='weight'), c_act]
            else:
                layers = [nn.Linear(inp_dim, num_channels), c_act]
        else:
            if use_wn:
                layers = [weight_norm(nn.Linear(inp_dim, out_dim), name='weight')]
            else:
                layers = [nn.Linear(inp_dim, out_dim)]
            if last_act_type != "none":
                c_act = activation_func(act_type=last_act_type, num_channels=out_dim, a=a, b=b, trainable=trainable)
                layers.append(c_act)
        for i in range(num_layers - 1):
            if i < num_layers - 2:
                c_act = activation_func(act_type=act_type, num_channels=num_channels, a=a, b=b, trainable=trainable)
                if use_wn:
                    layers.append(weight_norm(nn.Linear(num_channels, num_channels), name='weight'))
                else:
                    layers.append(nn.Linear(num_channels, num_channels))
                layers.append(c_act)
            else:
                if use_wn:
                    layers.append(weight_norm(nn.Linear(num_channels, out_dim), name='weight'))
                else:
                    layers.append(nn.Linear(num_channels, out_dim))
                if last_act_type != "none":
                    c_act = activation_func(act_type=last_act_type, num_channels=out_dim, a=a, b=b, trainable=trainable)
                    layers.append(c_act)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class PTMLP(nn.Module):
    def __init__(self, args, inp_dim=3, out_dim=64):
        super(PTMLP, self).__init__()
        self.args = args
        if args.type in [1, 2]:
            inp_dim = inp_dim + inp_dim * 2 * args.pt_L
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.model = MLP(inp_dim=inp_dim, num_layers=args.num_layers, num_channels=args.dim, out_dim=out_dim, 
                            act_type=args.act, last_act_type=args.last_act, use_wn=args.use_wn)

    def forward(self, N, x):
        out = self.model(x)
        if self.args.type == 1:
            return out.reshape(N, -1)
        elif self.args.type == 2:
            return out.reshape(N, N, -1)
        else:
            raise NotImplementedError("Attn mlp not implemented for out_dim > 2")


class WMLP(nn.Module):
    def __init__(self, args, inp_dim=2, out_dim=2):
        super(WMLP, self).__init__()
        self.args = args
        inp_dim = inp_dim + inp_dim * 2 * args.w_L
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.model = MLP(inp_dim=inp_dim, num_layers=args.num_layers, num_channels=args.dim, out_dim=out_dim, 
                            act_type=args.act, last_act_type=args.last_act, use_wn=args.use_wn)

    def forward(self, H, W, x):
        out = self.model(x).reshape(H, W, -1, self.out_dim)
        sftmax = nn.Softmax(dim=2)
        if self.out_dim == 2:
            return (sftmax(out[:, :, :, 0]) * out[:, :, :, 1])
        elif self.out_dim == 1:
            return sftmax(out).squeeze(-1)
        else:
            raise NotImplementedError("Attn mlp not implemented for out_dim > 2")


class AttnMLP(nn.Module):
    def __init__(self, args, inp_dim=2, out_dim=2):
        super(AttnMLP, self).__init__()
        self.args = args
        if args.type not in [3]:
            inp_dim = inp_dim + inp_dim * 2 * args.attn_L
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.model = MLP(inp_dim=inp_dim, num_layers=args.num_layers, num_channels=args.dim, out_dim=out_dim, 
                            act_type=args.act, last_act_type=args.last_act, use_wn=args.use_wn)

    def forward(self, N, H, W, x):
        out = self.model(x).reshape(N, H, W, -1, self.out_dim)
        sftmax = nn.Softmax(dim=3)
        if self.out_dim == 2:
            return (sftmax(out[:, :, :, :, 0]) * out[:, :, :, :, 1]).unsqueeze(-1)
        elif self.out_dim == 1:
            return sftmax(out)
        else:
            raise NotImplementedError("Attn mlp not implemented for out_dim > 2")


class NormMLP(nn.Module):
    def __init__(self, args, inp_dim, out_dim=3):
        super(NormMLP, self).__init__()
        self.args = args
        self.model = MLP(inp_dim=inp_dim, num_layers=args.num_layers, num_channels=args.dim, out_dim=out_dim, 
                            act_type=args.act, last_act_type=args.last_act, use_wn=args.use_wn)

    def forward(self, x):
        out = self.model(x)
        # if self.args.last_act == "tanh":
        #     return (out + 1) / 2
        # else:
        #     return out
        return out
        
    
import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
import warnings
import matplotlib.pyplot as plt
from .mlp import MLP


def get_transformer(args, v_extra_dim=0, kq_extra_dim=0, N_sample=64):
    if args.ray_embed_type == 1:
        ray_embed_dim = 3
    elif args.ray_embed_type == 2:
        ray_embed_dim = 6
    else:
        raise NotImplementedError('ray embed type [{:d}] is not supported'.format(args.ray_embed_type))

    kq_dim_map = {
        30: N_sample,
        31: N_sample * 3,
    }
    kq_dim = kq_dim_map[args.kq_type]

    n_kernel_map = {
        30: 1,
        31: 1,
    }
    n_kernel = n_kernel_map[args.kq_type]

    value_dim_map = {
        23: N_sample,
        24: N_sample * 3,
    }
    value_dim = value_dim_map[args.value_type]

    tx_kq_dim = kq_dim + kq_dim * 2 * args.kq_L + kq_extra_dim
    tx_v_dim = value_dim + value_dim * 2 * args.value_L + v_extra_dim

    if args.type == 'embed':
        return EmbedTransformer(d_kq=tx_kq_dim, d_v=tx_v_dim, n_kernel=n_kernel, n_ff_layer=args.n_ff_layer, 
                        ff_act=args.ff_act, N=args.num_layers, d_model=args.dim, d_ff=args.ff_dim, 
                        h=args.num_heads, dropout=args.dropout, temperature=args.temp, 
                        norm=args.norm, residual=args.residual, residual_embed=args.residual_embed,
                        concat=args.concat, share_embed=args.share_embed, d_ff_embed=args.d_ff_embed, 
                        n_ff_layer_embed=args.n_ff_layer_embed, ff_act_embed=args.ff_act_embed, 
                        dropout_embed=args.dropout_embed, norm_embed=args.norm_embed, act_a_embed=args.act_a_embed,
                        act_b_embed=args.act_b_embed, act_a_ff=args.act_a_ff, act_b_ff=args.act_b_ff), kq_dim, value_dim, \
                        tx_kq_dim, tx_v_dim, n_kernel
    else:
        raise NotImplementedError('transformer type [{:s}] is not supported'.format(args.type))
                        

def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    "Construct a layernorm module"

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class InstanceNorm(nn.Module):
    "Construct a InstanceNorm module"

    def __init__(self, eps=1e-6):
        super(InstanceNorm, self).__init__()
        self.eps = eps

    def forward(self, x):
        mean = x.mean(0, keepdim=True)
        std = x.std(0, keepdim=True)
        return (x - mean) / (std + self.eps)


def attention(query, key, kernel_type, temperature=1.0):
    """
        Compute 'Scaled Dot Product Attention'
        query: [batch_size, n_heads, seq_len, d_kq] or [batch_size, seq_len, d_kq]
        key:   [batch_size, n_heads, seq_len, d_kq] or [batch_size, seq_len, d_kq]
    """
    d_kq = query.size(-1)
    
    if kernel_type == "scaled-dot":
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_kq)
    elif kernel_type == "-scaled-dot":
        scores = -torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_kq)
    elif kernel_type == "dot":
        scores = torch.matmul(query, key.transpose(-2, -1))
    elif kernel_type == "-dot":
        scores = -torch.matmul(query, key.transpose(-2, -1))
    elif kernel_type == "l1-dist":
        scores = torch.norm(query.unsqueeze(-2) - key.unsqueeze(-3), p=1, dim=-1)
    elif kernel_type == "-l1-dist":
        scores = -torch.norm(query.unsqueeze(-2) - key.unsqueeze(-3), p=1, dim=-1)
    elif kernel_type == "l2-dist":
        scores = torch.norm(query.unsqueeze(-2) - key.unsqueeze(-3), p=2, dim=-1)
    elif kernel_type == "-l2-dist":
        scores = -torch.norm(query.unsqueeze(-2) - key.unsqueeze(-3), p=2, dim=-1)
    elif kernel_type == "scaled-l2-dist":
        scores = torch.norm(query.unsqueeze(-2) - key.unsqueeze(-3), p=2, dim=-1) / math.sqrt(d_kq)
    elif kernel_type == "-scaled-l2-dist":
        scores = -torch.norm(query.unsqueeze(-2) - key.unsqueeze(-3), p=2, dim=-1) / math.sqrt(d_kq)
    elif kernel_type == "cosine":
        scores = torch.matmul(query, key.transpose(-2, -1)) / (
            torch.norm(query, dim=-1, keepdim=True)
            * torch.norm(key, dim=-1, keepdim=True).transpose(-2, -1)
        )
    else:
        raise ValueError("Unknown kernel type: {}".format(kernel_type))

    return scores


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, norm="layernorm", residual=True, temperature=1.0):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.residual = residual
        self.temperature = temperature

        if norm == "layernorm":
            self.norm = LayerNorm(d_model)
        elif norm == "instancenorm":
            self.norm = InstanceNorm()
        elif norm == "none":
            self.norm = nn.Identity()
        else:
            raise ValueError("Invalid Transformer norm type")

    def forward(self, k, q, v, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = k.size(0)

        # inp = self.norm(inp)
        k = self.norm(k)
        q = self.norm(q)
        v = self.norm(v)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (k, q, v))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        scores = attention(query, key, 'scaled-dot')
        # print(self.attn.shape)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = (scores * self.temperature).softmax(dim=-1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        x = torch.matmul(p_attn, value) # [batch_size, n_heads, seq_len, d_model]
        self.attn = p_attn

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        if self.residual:
            return self.linears[-1](x) + v
        else:
            return self.linears[-1](x)


class FeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_input, d_model, d_ff, n_layer=2, act="relu", dropout=0.1, norm="layernorm", 
                    residual=True, act_a=1.0, act_b=1.0):
        super(FeedForward, self).__init__()
        if norm == "layernorm":
            self.norm = LayerNorm(d_input)
        elif norm == "instancenorm":
            self.norm = InstanceNorm()
        elif norm == "none":
            self.norm = nn.Identity()
        else:
            raise ValueError("Invalid Transformer norm type")
        self.dropout = nn.Dropout(dropout)

        print("Act:", act, act_a)
        # self.mlp = MLP(d_input, n_layer, d_ff, d_model, act_type=act, last_act_type="none", use_wn=False)
        self.mlp = MLP(d_input, n_layer, d_ff, d_model, act_type=act, last_act_type="none", use_wn=False, 
                        a=act_a, b=act_b, trainable=False)
        self.residual = residual

    def forward(self, x):
        if self.residual:
            return x + self.dropout(self.mlp(self.norm(x)))
        else:
            return self.dropout(self.mlp(self.norm(x)))


class EmbedTransformer(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many other models.
    """
    def __init__(self, d_kq, d_v, n_kernel, n_ff_layer, ff_act="relu", N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, 
                 temperature=1.0, norm="layernorm", residual=True, residual_embed=False, concat=False, share_embed=False,
                 d_ff_embed=2048, n_ff_layer_embed=2, ff_act_embed="relu", dropout_embed=0.1, norm_embed="layernorm",
                 act_a_embed=1.0, act_b_embed=1.0, act_a_ff=1.0, act_b_ff=1.0):
        super(EmbedTransformer, self).__init__()
        c = copy.deepcopy

        if norm == "layernorm":
            out_dim = d_ff * N if concat else d_ff
            self.norm = LayerNorm(out_dim)
        elif norm == "instancenorm":
            self.norm = InstanceNorm()
        elif norm == "none":
            self.norm = nn.Identity()
        else:
            raise ValueError("Invalid Transformer norm type")

        self.N = N
        self.concat = concat
        self.share_embed = share_embed

        if share_embed:
            assert d_kq == d_v
            self.embed_mlp = FeedForward(d_kq, d_model, d_ff_embed, n_ff_layer_embed, ff_act_embed, dropout_embed, norm_embed, residual_embed, act_a_embed, act_b_embed)
        else:
            self.k_embed_mlp = FeedForward(d_kq, d_model, d_ff_embed, n_ff_layer_embed, ff_act_embed, dropout_embed, norm_embed, residual_embed, act_a_embed, act_b_embed)
            self.q_embed_mlp = FeedForward(d_kq, d_model, d_ff_embed, n_ff_layer_embed, ff_act_embed, dropout_embed, norm_embed, residual_embed, act_a_embed, act_b_embed)
            self.v_embed_mlp = FeedForward(d_v, d_model, d_ff_embed, n_ff_layer_embed, ff_act_embed, dropout_embed, norm_embed, residual_embed, act_a_embed, act_b_embed)

        # self.selfattn_1 = SelfAttnFirstLayer(n_kernel, d_kq, d_v, d_model, dropout, temperature)
        self.add_module("selfattn_1", MultiHeadedAttention(h, d_model, dropout, norm, residual))
        self.add_module("ff_1", FeedForward(d_model, d_model, d_ff, n_ff_layer, ff_act, dropout, norm, residual, act_a_ff, act_b_ff))
        for i in range(2, N+1):
            self.add_module("selfattn_{}".format(i), MultiHeadedAttention(h, d_model, dropout, norm, residual))
            self.add_module("ff_{}".format(i), FeedForward(d_model, d_model, d_ff, n_ff_layer, ff_act, dropout, norm, residual, act_a_ff, act_b_ff))

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, k, q, v, kernels):
        self.middle_outputs = []
        if self.share_embed:
            k = self.embed_mlp(k)
            q = self.embed_mlp(q)
            v = self.embed_mlp(v)
        else:
            k = self.k_embed_mlp(k)
            q = self.q_embed_mlp(q)
            v = self.v_embed_mlp(v)
        x = getattr(self, "selfattn_1")(k, q, v)
        x = getattr(self, "ff_1")(x)
        self.middle_outputs.append(x)
        for i in range(2, self.N+1):
            x = getattr(self, "selfattn_{}".format(i))(x, x, x)
            x = getattr(self, "ff_{}".format(i))(x)
            self.middle_outputs.append(x)
        if self.concat:
            x = torch.cat(self.middle_outputs, dim=-1)
        # print(x.shape)
        return self.norm(x)


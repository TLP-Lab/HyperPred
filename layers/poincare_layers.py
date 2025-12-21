import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, softmax, add_self_loops
from torch_scatter import scatter, scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch.nn.parameter import Parameter
from torch_geometric.nn.inits import glorot, zeros
from typing import Tuple
from functools import partial
from inspect import signature
from torch_geometric.nn.inits import glorot

class HGATConv(nn.Module):
    """
    Poincare graph convolution layer.
    """
    def __init__(self, manifold, in_features, out_features, c_in, c_out, device, act=F.leaky_relu,
                 dropout=0.6, att_dropout=0.6, use_bias=True, heads=2, concat=False):
        super(HGATConv, self).__init__()
        out_features = out_features * heads
        self.linear = HypLinear(manifold, in_features, out_features, c_in, device, dropout=dropout, use_bias=use_bias)
        self.agg = HypAttAgg(manifold, c_in, out_features, device, att_dropout, heads=heads, concat=concat)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.device = device

    def forward(self, x, edge_index):
        h = self.linear.forward(x)
        h = self.agg.forward(h, edge_index)
        h = self.hyp_act.forward(h)
        return h


class HGCNConv(nn.Module):
    """
    Poincare graph convolution layer, from HGCN。
    """
    def __init__(self, manifold, in_features, out_features, device, c_in=1.0, c_out=1.0, dropout=0.6, act=F.leaky_relu,
                 use_bias=True):
        super(HGCNConv, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c_in, device, dropout=dropout, use_bias=use_bias)
        self.agg = HypAgg(manifold, c_in, out_features, device, bias=use_bias)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)
        self.manifold = manifold
        self.c_in = c_in
        self.device = device

    def forward(self, x, edge_index):
        h = self.linear.forward(x)
        h = self.agg.forward(h, edge_index)
        h = self.hyp_act.forward(h)
        return h


class HypLinear(nn.Module):
    """
    Poincare linear layer.
    """
    def __init__(self, manifold, in_features, out_features, c, device, dropout=0.6, use_bias=True):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.device = device
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = Parameter(torch.Tensor(out_features).to(device), requires_grad=True)
        self.weight = Parameter(torch.Tensor(out_features, in_features).to(device), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, p=self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )


class HypAttAgg(MessagePassing):
    def __init__(self, manifold, c, out_features, device, att_dropout=0.6, heads=1, concat=False):
        super(HypAttAgg, self).__init__()
        self.manifold = manifold
        self.dropout = att_dropout
        self.out_channels = out_features // heads
        self.negative_slope = 0.2
        self.heads = heads
        self.c = c
        self.device = device
        self.concat = concat
        self.att_i = Parameter(torch.Tensor(1, heads, self.out_channels).to(device), requires_grad=True)
        self.att_j = Parameter(torch.Tensor(1, heads, self.out_channels).to(device), requires_grad=True)
        glorot(self.att_i)
        glorot(self.att_j)

    def forward(self, x, edge_index):
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index,
                                       num_nodes=x.size(self.node_dim))

        edge_index_i = edge_index[0]
        edge_index_j = edge_index[1]

        x_tangent0 = self.manifold.logmap0(x, c=self.c)  # project to origin
        x_i = torch.nn.functional.embedding(edge_index_i, x_tangent0)
        x_j = torch.nn.functional.embedding(edge_index_j, x_tangent0)
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)

        alpha = (x_i * self.att_i).sum(-1) + (x_j * self.att_j).sum(-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes=x_i.size(0))
        alpha = F.dropout(alpha, self.dropout, training=self.training)
        support_t = scatter(x_j * alpha.view(-1, self.heads, 1), edge_index_i, dim=0)

        if self.concat:
            support_t = support_t.view(-1, self.heads * self.out_channels)
        else:
            support_t = support_t.mean(dim=1)
        support_t = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)

        return support_t


class HypAct(Module):
    """
    Poincare activation layer.
    """
    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )


class HypAgg(MessagePassing):
    """
    Poincare aggregation layer using degree.
    """
    def __init__(self, manifold, c, out_features, device, bias=True):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c
        self.device = device
        self.use_bias = bias
        if bias:
            self.bias = Parameter(torch.Tensor(out_features).to(device))
        else:
            self.register_parameter('bias', None)
        zeros(self.bias)
        self.mlp = nn.Sequential(nn.Linear(out_features * 2, 1).to(device))

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index=None):
        x_tangent = self.manifold.logmap0(x, c=self.c)
        edge_index, norm = self.norm(edge_index, x.size(0), dtype=x.dtype)
        node_i = edge_index[0]
        node_j = edge_index[1]
        x_j = torch.nn.functional.embedding(node_j, x_tangent)
        support = norm.view(-1, 1) * x_j
        support_t = scatter(support, node_i, dim=0, dim_size=x.size(0))  # aggregate the neighbors of node_i
        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)
    

def build_kwargs_from_config(config, target_func):
    valid_keys = list(signature(target_func).parameters)
    kwargs = {}
    for key in config:
        if key in valid_keys:
            kwargs[key] = config[key]
    return kwargs

def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2

def val2tuple(x, min_len=1, idx_repeat=0):
    x = val2list(x)
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]
    return tuple(x)

def val2list(x, repeat_time=1):
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]

REGISTERED_ACT_DICT = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "hswish": nn.Hardswish,
    "silu": nn.SiLU,
    "gelu": partial(nn.GELU, approximate="tanh"),
}

def build_act(name, **kwargs):
    if name in REGISTERED_ACT_DICT:
        act_cls = REGISTERED_ACT_DICT[name]
        args = build_kwargs_from_config(kwargs, act_cls)
        return act_cls(**args)
    return None

REGISTERED_NORM_DICT = {
    "bn1d": nn.BatchNorm1d,
    "ln": nn.LayerNorm,
}

def build_norm(name="bn1d", num_features=None, **kwargs):
    if name == "ln":
        kwargs["normalized_shape"] = num_features
    else:
        kwargs["num_features"] = num_features
    if name in REGISTERED_NORM_DICT:
        norm_cls = REGISTERED_NORM_DICT[name]
        args = build_kwargs_from_config(kwargs, norm_cls)
        return norm_cls(**args)
    return None

class HypConvLayer(nn.Module):
    def __init__(self, manifold,k_in,in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 groups=1, use_bias=False, dropout=0, norm="bn1d", act_func="relu"):
        super().__init__()
        self.manifold=manifold
        self.casual_kernel_size=2*kernel_size-1
        self.dilation=dilation
        self.pad=(self.casual_kernel_size - 1)//2 * dilation
        self.dropout = nn.Dropout(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=self.casual_kernel_size,
                              stride=stride, padding=self.pad,
                              dilation=self.dilation, groups=groups, bias=use_bias)
        self.norm = build_norm(norm, num_features=out_channels)
        self.act = build_act(act_func)
        self.c=k_in
        self.reset_parameters()
    
    def reset_parameters(self):
        glorot(self.conv.weight)

    def to_tangent(self, x, c=1.0):
        x_tan = self.manifold.logmap0(x, c)
        x_tan = self.manifold.proj_tan0(x_tan, c)
        return x_tan

    def to_hyper(self, x, c=1.0):
        x_tan = self.manifold.proj_tan0(x, c)
        x_hyp = self.manifold.expmap0(x_tan, c)
        x_hyp = self.manifold.proj(x_hyp, c)
        return x_hyp

    def forward(self, x):
        x = self.to_tangent(self.manifold.proj(x, self.c), self.c)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.permute(1, 2, 0)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        x = x.permute(2,0,1)
        if self.act:
            x = self.act(x)
        x=self.to_hyper(x)
        return x

class EfficientMLHA(nn.Module):
    def __init__(self, manifold,k_hidden,in_channels, out_channels, casual_kernel_size,dilation,heads_ratio=1.0, dim=8,
                 use_bias=False,kernel_func="relu", scales: Tuple[int, ...] = (5,), eps=1e-15):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.heads = int(in_channels // dim * heads_ratio)
        total_dim = self.heads * dim

        use_bias = val2tuple(use_bias, 2)
        
        self.manifold=manifold
        self.c=k_hidden
        self.qkv = HypConvLayer(self.manifold,self.c,in_channels, 3 * total_dim, kernel_size=casual_kernel_size,dilation=dilation,
                               use_bias=use_bias[0])
        self.aggreg = nn.ModuleList()
        for scale in scales:
            module = nn.Sequential(
                nn.Conv1d(3 * total_dim, 3 * total_dim, kernel_size=scale,
                          padding=get_same_padding(scale), groups=3 * total_dim, bias=use_bias[0]),
                nn.Conv1d(3 * total_dim, 3 * total_dim, kernel_size=1,
                          groups=3 * self.heads, bias=use_bias[0])
            )
            self.aggreg.append(module)

        self.kernel_func = build_act(kernel_func, inplace=False)
    
        self.conv_res = nn.Conv1d(total_dim * (1 + len(scales)), out_channels, kernel_size=1)

        self.conv_skip = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def to_tangent(self, x, c=1.0):
        x_tan = self.manifold.logmap0(x, c)
        return x_tan

    def to_hyper(self, x, c=1.0):
        x_hyp = self.manifold.expmap0(x, c)
        x_hyp = self.manifold.proj(x_hyp, c)
        return x_hyp

    def relu_linear_att(self, qkv):
        qkv = self.to_tangent(self.manifold.proj(qkv, self.c), self.c)
        L, N, C = qkv.shape
        qkv = qkv.permute(1,2,0).contiguous()
        group = C // (3 * self.dim)
        qkv = qkv.view(N, group, 3 * self.dim, L)
        q, k, v = torch.chunk(qkv, 3, dim=2)
        print("q.shape:",q.shape)

        q = self.kernel_func(q)
        k = self.kernel_func(k)

        vk = torch.einsum('ngdl,ngel->ngde', v, k)   # [N,group,dim,dim]
        out = torch.einsum('ngde,ngdl->ngdl', vk, q) # [N,group,dim,L]

        norm_factor = v.sum(dim=-1, keepdim=True) + self.eps
        out = out / norm_factor
        out = out.reshape(N, group * self.dim, L)
        out=out.permute(2,0,1)
        out=self.to_hyper(out,self.c)

        return out  

    def forward(self, x):
        # multi-tokens generation
        qkv = self.qkv(x)
        qkv = self.to_tangent(self.manifold.proj(qkv, self.c), self.c)
        qkv = qkv.permute(1, 2, 0)
        multi_scale = [qkv]
        for op in self.aggreg:
            multi_scale.append(op(qkv))
        qkv = torch.cat(multi_scale, dim=1)  # [N,C_total,L]
        attn_input = qkv.permute(2, 0, 1)    # [L,N,C_total]
        attn_input=self.to_hyper(attn_input)
        print("attn_input.shape:",attn_input.shape)
        fx = self.relu_linear_att(attn_input)    # [L,N,C]
        print("fx.shape:",fx.shape) 
        # projection
        fx = self.to_tangent(self.manifold.proj(fx, self.c), self.c)
        fx = fx.permute(1, 2, 0)
        fx = self.conv_res(fx)
        fx = fx.permute(2, 0, 1)
        fx = self.to_hyper(fx, self.c)
        # residual connection
        skip = self.to_tangent(self.manifold.proj(fx, self.c), self.c)
        skip = skip.permute(1, 2, 0)
        skip = self.conv_skip(skip)
        skip = skip.permute(2, 0, 1)
        skip = self.to_hyper(skip, self.c)
        print("temporal模块skip的形状：",skip.shape)
        residual = self.manifold.mobius_add(fx, x, self.c)
        return fx,residual

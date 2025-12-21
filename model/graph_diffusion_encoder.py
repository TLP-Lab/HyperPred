import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot

from manifolds import Euclidean, Lorentzian, PoincareDisk
from layers import HGCNConv,HypLinear

class GraphDiffusionEncoder(nn.Module):
    def __init__(self, args):
        super(GraphDiffusionEncoder, self).__init__()
        self.feat = Parameter((torch.ones(args.num_nodes, args.nfeat)).to(args.device), requires_grad=True)
        self.linear = nn.Linear(args.nfeat, args.nhid).to(args.device)
        if args.manifold == 'PoincareDisk':
            self.manifold = PoincareDisk()
        elif args.manifold == 'Lorentzian':
            self.manifold = Lorentzian()
        elif args.manifold == 'Euclidean':
            self.manifold = Euclidean()
        else:
            raise RuntimeError('invalid argument: manifold')
        self.c = Parameter(torch.ones(len(args.spatial_dilated_factors) * 3 + 1, 1).to(args.device) * args.curvature,
                           requires_grad=args.trainable_curvature)
        self.c_in=self.c[0]
        self.c_out = self.c[-1]
        self.spatial_layers = []
        for i in range(len(args.spatial_dilated_factors)):
                layer1 = HGCNConv(self.manifold, args.nhid, args.nhid, args.device, self.c[i * 3],
                                  self.c[i * 3 + 1], dropout=args.dropout, use_bias=args.bias).to(device=args.device)
                layer2 = HGCNConv(self.manifold, args.nhid, args.nout, args.device, self.c[i * 3 + 1],
                                  self.c[i * 3 + 2], dropout=args.dropout, use_bias=args.bias).to(device=args.device)
                self.spatial_layers.append([layer1, layer2])

        self.nhid = args.nhid
        self.nout = args.nout
        self.diffusion_steps = args.diffusion_steps
        self.Q = Parameter(torch.ones((args.nout, args.nhid)).to(args.device), requires_grad=True)
        self.r = Parameter(torch.ones((args.nhid, 1)).to(args.device), requires_grad=True)
        self.reset_parameter()

    def reset_parameter(self):
        glorot(self.feat)
        glorot(self.linear.weight)
        glorot(self.Q)
        glorot(self.r)

    def to_hyper(self, x, c=1.0):
        x_tan = self.manifold.proj_tan0(x, c)
        x_hyp = self.manifold.expmap0(x_tan, c)
        x_hyp = self.manifold.proj(x_hyp, c)
        return x_hyp

    def to_tangent(self, x, c=1.0):
        x_tan = self.manifold.logmap0(x, c)
        return x_tan

    def init_hyper(self, x, c=1.0):
        if isinstance(self.manifold, Lorentzian):
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)
        return self.to_hyper(x, c)

    def aggregate_diffusion(self, diffusion):
        att = torch.matmul(torch.tanh(torch.matmul(diffusion, self.Q)), self.r)
        att = torch.reshape(att, (len(self.diffusion_steps), -1))
        att = F.softmax(att, dim=0).unsqueeze(2)
        diffusion_reshape = torch.reshape(diffusion, [len(self.diffusion_steps), -1, self.nout])
        diffusion_agg = torch.mean(att * dilated_reshape, dim=0)
        return diffusion_agg

    def forward(self, diffusion_edge_index, x=None):
        x_list = []
        if x is None:
            x = self.linear(self.feat)
        else:
            x = self.linear(x)
        for i in range(len(self.diffusion_steps)):
            x_f = self.init_hyper(x, self.c[i * 3])
            x_f = self.spatial_layers[i][0](x_f, diffusion_edge_index[i])
            x_f = self.manifold.proj(x_f, self.c[i * 3 + 1])
            x_f = self.spatial_layers[i][1](x_f, diffusion_edge_index[i])
            x_list.append(x_f)
        x = torch.cat([self.to_tangent(x_list[i], self.c[i * 3 + 2])
                       for i in range(len(self.diffusion_steps))], dim=0)
        x = self.aggregate_diffusion(x)
        x = self.to_hyper(x, self.c_out)
        return x

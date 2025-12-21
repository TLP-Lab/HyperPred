import os
import numpy as np
import torch
from scipy.sparse import coo_matrix
from tqdm import tqdm
from torch_geometric.utils import remove_self_loops
import scipy.sparse as sp
from scipy.sparse import coo_matrix, eye
from utils.util import logger
from scipy.linalg import expm

def mkdirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def loader(args):
    filepath = os.path.join(args.data_pt_path, '{}.data'.format(args.dataset))
    if os.path.isfile(filepath):
        logger.info('loading {} ...'.format(args.dataset))
        return torch.load(filepath)
    else:
        logger.info('{} does not exist, exiting...')
        return None

def prepare_diffusion_edge_index(data, diffusion_steps, device, top_k=None, eps=0.005, t: float = 1.0):
    diffusion_edge_index_list = []
    for edge_index in tqdm(data['edge_index_list']):
        adj = coo_matrix(([1] * len(edge_index[0]), (list(edge_index[0]), list(edge_index[1]))), dtype=np.float32)
        adj = adj.tocsr() + adj.transpose().tocsr()
        num_nodes = adj.shape[0]

        diffusion_edge_index = []
        for k in diffusion_steps:
            A_tilde = adj + eye(num_nodes, dtype=np.float32, format='csr')
            deg = np.array(A_tilde.sum(1)).flatten()
            deg[deg == 0] = 1.0
            D_inv_sqrt = sp.diags(1.0 / np.sqrt(deg))
            H = D_inv_sqrt.dot(A_tilde).dot(D_inv_sqrt)
            L = eye(num_nodes, format='csr', dtype=np.float32) - H
            power = eye(num_nodes, format='csr', dtype=np.float32)
            M = coo_matrix(eye(num_nodes, dtype=np.float32))
            coef = 1.0
            for n in range(1, k + 1): 
                power = power.dot(L)
                coef = coef * (-t) / n
                M = M + coef * power
            M = M.tocoo()
            M.eliminate_zeros()
            if top_k is not None:
                new_rows, new_cols, new_data = [], [], []
                for i in range(num_nodes):
                    mask = M.row == i
                    row_vals = M.data[mask]
                    row_cols = M.col[mask]
                    if len(row_vals) > 0:
                        if len(row_vals) > top_k:
                            top_idx = np.argsort(row_vals)[-top_k:]
                            row_vals = row_vals[top_idx]
                            row_cols = row_cols[top_idx]
                        new_rows.extend([i]*len(row_vals))
                        new_cols.extend(row_cols)
                        new_data.extend(row_vals)
                M = coo_matrix((new_data, (new_rows, new_cols)), shape=(num_nodes, num_nodes))
            elif eps is not None:
                mask = M.data >= eps
                if np.any(mask):
                    M = coo_matrix((M.data[mask], (M.row[mask], M.col[mask])), shape=(num_nodes, num_nodes))
                else:
                    M = coo_matrix((num_nodes, num_nodes))
            coords = np.vstack((M.row, M.col))
            if coords.shape[1] > 0:
                np.random.shuffle(coords.T)
            coords_tensor, _ = remove_self_loops(torch.tensor(coords, dtype=torch.long))
            diffusion_edge_index.append(coords_tensor.to(device))

        diffusion_edge_index_list.append(diffusion_edge_index)
    return diffusion_edge_index_list


def prepare_train_test_data(data, t, device):
    edge_index = data['edge_index_list'][t].long().to(device)
    pos_index = data['pedges'][t].long().to(device)
    neg_index = data['nedges'][t].long().to(device)
    new_pos_index = data['new_pedges'][t].long().to(device)
    new_neg_index = data['new_nedges'][t].long().to(device)
    return edge_index, pos_index, neg_index, new_pos_index, new_neg_index

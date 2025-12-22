import torch
import numpy as np
import time
import geoopt
import networkx as nx
from math import isnan
from tqdm import tqdm
import matplotlib.pyplot as plt
from config import args
from utils.data_utils import loader, prepare_diffusion_edge_index, prepare_train_test_data
from utils.util import set_random, logger, hyperbolicity_sample
from model import HyperPred
from loss import ReconLoss
def tensor_to_list(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

class Trainer(object):
    def __init__(self):
        self.data = loader(args)
        if self.data is None:
            raise RuntimeError('dataset not exsits')
        args.num_nodes = self.data['num_nodes']
        self.train_shots = list(range(0, self.data['time_length'] - args.test_length))
        self.test_shots = list(range(self.data['time_length'] - args.test_length, self.data['time_length']))
        self.model = HyperPred(args).to(args.device)
        self.loss = ReconLoss(args, self.model.c_out)
        if args.use_riemannian_adam:
            self.optimizer = geoopt.optim.radam.RiemannianAdam(self.model.parameters(), lr=args.lr,
                                                               weight_decay=args.weight_decay)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        set_random(args.seed)

    def train(self):
        diffusion_edge_index_list = prepare_diffusion_edge_index(
            self.data,
            args.diffusion_steps,
            args.device
        )
        t_total = time.time()
        test_result = []
        min_loss = 1.0e8
        patience = 0
        auc_list=[]
        ap_list=[]
        er_list=[]
        new_auc_list=[]
        new_ap_list=[]
        new_er_list=[]
        best_auc=0
        best_ap=0
        best_er=1
        best_new_auc=0
        best_new_ap=0
        best_new_er=1
        for epoch in range(1, args.max_epoch + 1):
            t_epoch = time.time()
            epoch_losses = []
            z = None
            epoch_loss = None
            self.model.init_history()
            self.model.train()
            for t in self.train_shots:
                edge_index, _, _, _, _ = prepare_train_test_data(self.data,
                                                                 t if (t + 1) not in self.train_shots else (t + 1),
                                                                 args.device)
                diffusion_edge_index = diffusion_edge_index_list[t]
                self.optimizer.zero_grad()
                z = self.model(diffusion_edge_index)
                epoch_loss = self.loss(z, edge_index) + self.model.htc(z)
                epoch_loss.backward()
                if isnan(epoch_loss):
                    logger.info('==' * 25)
                    logger.info('nan loss')
                    break
                self.optimizer.step()
                epoch_losses.append(epoch_loss.item())
                self.model.update_history(z)
            if isnan(epoch_loss):
                break
            self.model.eval()
            average_epoch_loss = np.mean(epoch_losses)
            train_result = self.test(z, is_training=True)
            test_result = self.test(z)
            auc_list.append(test_result[0])
            ap_list.append(test_result[1])
            er_list.append(test_result[2])
            new_auc_list.append(test_result[3])
            new_ap_list.append(test_result[4])
            new_er_list.append(test_result[5])
            if best_auc<test_result[0]:
                best_auc=test_result[0]
            if best_ap<test_result[1]:
                best_ap=test_result[1]
            if best_er>test_result[2]:
                best_er=test_result[2]
            if best_new_auc<test_result[3]:
                best_new_auc=test_result[3]
            if best_new_ap<test_result[4]:
                best_new_ap=test_result[4]
            if best_new_er>test_result[5]:
                best_new_er=test_result[5]

            if average_epoch_loss < min_loss:
                min_loss = average_epoch_loss
                patience = 0

            else:
                patience += 1
                if epoch > args.min_epoch and patience > args.patience:
                    logger.info('==' * 25)
                    logger.info('early stopping!')
                    break
            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
            if epoch == 1 or epoch % args.log_interval == 0:
                logger.info('==' * 25)
                logger.info('Epoch:{}, Loss:{:.4f}, Time:{:.3f},GPU: {:.1f}MB'
                            .format(epoch,
                                    average_epoch_loss,
                                    time.time() - t_epoch,gpu_mem_alloc))
                logger.info('Epoch:{}, Train, AUC:{:.4f}, AP:{:.4f},ER:{:.4f}, new AUC:{:.4f}, new AP:{:.4f},new ER:{:.4f}'
                            .format(epoch,
                                    train_result[0],
                                    train_result[1],
                                    train_result[2],
                                    train_result[3],
                                    train_result[4],
                                    train_result[5]))
                logger.info('Epoch:{}, Test, AUC:{:.4f}, AP:{:.4f},ER:{:.4f}, new AUC:{:.4f}, new AP:{:.4f},new ER:{:.4f}'
                            .format(epoch,
                                    test_result[0],
                                    test_result[1],
                                    test_result[2],
                                    test_result[3],
                                    test_result[4],
                                    test_result[5]))
        logger.info('==' * 25)
        logger.info('Total time: {:.3f}'.format(time.time() - t_total))
        return test_result


    def test(self, embeddings, is_training=False):
        auc_list, ap_list,er_list, new_auc_list, new_ap_list,new_er_list = [], [], [], [],[],[]
        embeddings = embeddings.detach()
        shots = [self.train_shots[-1]] if is_training else self.test_shots
        for t in shots:
            _, pos_index, neg_index, new_pos_index, new_neg_index = \
                prepare_train_test_data(self.data, t, args.device)
            auc, ap,er = self.loss.predict(embeddings, pos_index, neg_index)
            new_auc, new_ap,new_er = self.loss.predict(embeddings, new_pos_index, new_neg_index)
            auc_list.append(auc)
            ap_list.append(ap)
            er_list.append(er)
            new_auc_list.append(new_auc)
            new_ap_list.append(new_ap)
            new_er_list.append(new_er)
        return np.mean(auc_list), np.mean(ap_list),np.mean(er_list), np.mean(new_auc_list), np.mean(new_ap_list),np.mean(new_er_list)


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
    

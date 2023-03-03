import torch, random
from domainbed_model import domainbedNet, domainbedDataset

import sys
import os
import numpy as np
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
 
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
# adding the parent directory to
# the sys.path.
sys.path.append(parent)
from domainbed import networks
import copy
from domainbed import datasets
from domainbed import hparams_registry
# from domainbed import algorithms 
# from models.domainbed_net import *
from domainbed.lib import misc
from tqdm import tqdm

class empirical_metrics_real:
    def __init__(self, model:domainbedNet, dataset:domainbedDataset, num_trials=20):
        self.model = model
        self.dataset = dataset
        self.num_trials = num_trials
        self.target_var = None
        self.source_target_var = None
        self.projected_target_var = None
        self.target_proj_source_var = None

    def compute_quantities(self, subsample_ratio=0.5):
        # compute source target difference
        grads_S = []
        for dls in self.dataset.source_dls:
            grads_S.append(self.model.get_gradients(dls, False)) # using all source data
        grads_T = self.model.get_gradients(self.dataset.target_dls, True) # using all target training data
        # grads_S_arr = torch.cat(grads_S, dim=0)
        # avg_grads_S = torch.mean(grads_S_arr, dim=0)
        # print(grads_S_arr.shape, avg_grads_S.shape)
        avg_grads_S = 0

        for idx, frac in enumerate(self.dataset.source_frac):
            avg_grads_S += grads_S[idx] / len(grads_S)

        self.source_target_var = torch.norm(grads_T - avg_grads_S)**2

        self.target_proj_source_var = 0

        for idx, grad in enumerate(grads_S):
            normalized_grad_S = grad / torch.norm(grad)
            self.target_proj_source_var += (1. / len(grads_S)) * torch.norm((grads_T * normalized_grad_S).sum() * normalized_grad_S - grads_T)**2

        # self.target_proj_source_var /= len(grads_S)

        # compute target variance
        self.target_var = 0
        self.projected_target_var = 0
        grad_proj_filter_avg = 0
        for _ in range(self.num_trials):
            grads_T_subsample = self._compute_subsample_target_grads(subsample_ratio)
            self.target_var += torch.norm(grads_T - grads_T_subsample)**2
            for idx, grad in enumerate(grads_S):
                normalized_grad_S = grad / torch.norm(grad)
                if (grads_T_subsample * grad).sum() > 0:
                    grad_proj_filter = 1
                else:
                    grad_proj_filter = 0
                
                self.projected_target_var += torch.sum((grads_T - grads_T_subsample) * normalized_grad_S)**2 * grad_proj_filter * self.dataset.source_frac[idx]
                grad_proj_filter_avg += grad_proj_filter * self.dataset.source_frac[idx]


        self.target_var = self.target_var * subsample_ratio / (1-subsample_ratio) / self.num_trials
        self.projected_target_var = self.projected_target_var * subsample_ratio / (1-subsample_ratio) / self.num_trials
        grad_proj_filter_avg = grad_proj_filter_avg / self.num_trials
        self.target_proj_source_var = self.target_proj_source_var * grad_proj_filter_avg

    def _compute_subsample_target_grads(self, subsample_ratio):
        # subsample data
        target_data = self.dataset.target_dls.dataset
        subset, _ = misc.split_dataset(target_data,
                    int(len(target_data)*subsample_ratio))
        subset_dls = torch.utils.data.DataLoader(
                subset,
                batch_size=16)
        grads_T_subsample = self.model.get_gradients(subset_dls, True)
        return grads_T_subsample

    def compute_error_convex_combine(self, beta):
        return  (1 - beta)**2 * self.target_var + beta**2 * (self.source_target_var - self.target_var)

    def compute_error_gradient_proj(self, beta):
        return (2*beta-beta**2) * self.projected_target_var + (1-beta)**2 * self.target_var + beta**2 * (self.target_proj_source_var - self.target_var - self.projected_target_var)

if __name__ == '__main__':
    data_dir = '/shared/rsaas/enyij2/domainbed/data'

    datasets = ['ColoredMNIST', 'VLCS', 'PACS', 'TerraIncognita'] #'ColoredMNIST' #'PACS' #'TerraIncognita' #'VLCS'
    data2tr = {'ColoredMNIST': 0.01, 'VLCS': 0.05, 'PACS': 0.15, 'TerraIncognita': 0.05}
    # fix the initialized model weights
    for dataset in datasets:
        print(dataset)
        hparams = hparams_registry.default_hparams('fedgp', dataset)
        for test_env in [0,1,2,3]:
            domainbed_dataset = domainbedDataset(hparams, data_dir, dataset, test_env, target_ratio=data2tr[dataset])

            model = domainbedNet(domainbed_dataset.dataset.input_shape, domainbed_dataset.dataset.num_classes,
                len(domainbed_dataset.dataset) - 1, hparams)
            metric = empirical_metrics_real(model, domainbed_dataset)

            metric.compute_quantities(subsample_ratio=0.5)

            source_only = metric.compute_error_convex_combine(1)
            fedda = metric.compute_error_convex_combine(0.5)
            fedgp = metric.compute_error_gradient_proj(0.5)
            target_only = metric.compute_error_convex_combine(0)
            lst_res = sorted([source_only, fedda, fedgp, target_only])
            print(lst_res)
            # source only
            print('source only:', source_only, lst_res.index(source_only))

            # fedda
            print('FedDA:', fedda, lst_res.index(fedda))

            # fedgp
            print('FedGP:', fedgp, lst_res.index(fedgp))
            
            # target only
            print('target only:', target_only, lst_res.index(target_only))
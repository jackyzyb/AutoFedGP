
import sys
import os
import numpy as np
import torch
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
 
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
# adding the parent directory to
# the sys.path.
sys.path.append(parent)
global device
# device = 'cpu'

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from domainbed import networks
import copy
from domainbed import datasets
from domainbed import hparams_registry
# from domainbed import algorithms 
# from models.domainbed_net import *
from domainbed.lib import misc
from tqdm import tqdm
# from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

def get_param_list(model):
	m_dict = model.state_dict()
	param = []
	for key in m_dict.keys():
		param.append(torch.linalg.norm(m_dict[key].float()))
	return torch.FloatTensor(param)

def get_model_updates(init_model, new_model):
    ret_updates = []
    init = get_param_list(init_model)
    new = get_param_list(new_model)
    return (new - init).reshape(1, -1)

class domainbedNet(nn.Module):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(domainbedNet, self).__init__()
        self.hparams = hparams
        featurizer = networks.Featurizer(input_shape, self.hparams)
        classifier = networks.Classifier(
            featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.criterion = torch.nn.CrossEntropyLoss().to(device)
        self.network = nn.Sequential(featurizer, classifier)
        self.network.to(device)

    def forward(self, x):
        x = self.network(x)
        return x

    def get_gradients(self, dls, is_target=False):
        # self.optim = torch.optim.Adam(self.network.parameters(), lr= self.hparams["lr"])
        # old_net = copy.deepcopy(self.network)
        # old_dict = old_net.state_dict()
        for _ in range(1):
            cont = 0
            with tqdm(dls, unit="batch") as tepoch:
                self.network.zero_grad()
                for (imgs, labels) in tepoch:
                    # print(imgs.shape, labels.shape)
                    cont += labels.shape[0]
                    imgs, labels = imgs.to(device), labels.to(device)
        #             self.optim.zero_grad()
                    # zero-out the gradients
                    
                    # train one step and send back the model updates
                    pred = self.forward(imgs)
                    loss = self.criterion(pred, labels)
                    loss.backward()
                    # if cont > 10:
                    #     break
                # for key in self.network.state_dict().keys():
                #     print(key)
                # more layers for calculating gradients
                grads = []
                for _, param in self.network.named_parameters():
                    # print(param.shape)
                    grads.append(param.grad.detach().clone().flatten())
                    # print(grads)
                # last_w = self.network[1].weight.grad.detach().clone().flatten()
                # last_b = self.network[1].bias.grad.detach().clone().flatten()
                grad = torch.cat(grads) / cont
        # print(grad)
        #             # self.optim.step()
        # grad = get_model_updates(old_net, self.network)
        # # print(grad.shape)
        # # reset the weights
        # self.network.load_state_dict(old_dict)
        # return grad
        # print(self.network[1].bias)
        return grad

class domainbedDataset:
    def __init__(self, hparams, data_dir, dataset, test_env, target_ratio=0.05, source_ratio=0.2):
        super(domainbedDataset, self).__init__()
        self.hparams = hparams
        if dataset in vars(datasets):
            dataset = vars(datasets)[dataset](data_dir,
                [test_env], hparams)
            self.dataset = dataset
        else:
            raise NotImplementedError
        
        self.source_dls = []
        self.source_frac = []

        for env_i, env in enumerate(dataset):
            uda = []

            # split training/testing data
            out, in_ = misc.split_dataset(env,
                int(len(env)*source_ratio),
                misc.seed_hash(1, env_i))
            # split finetuning set from testing data
            if env_i  == test_env:
                # print(env_i)
                uda, in_ = misc.split_dataset(in_,
                    int(len(in_)*target_ratio),
                    misc.seed_hash(1, env_i))
                print(f"number of target samples: {len(uda)}")
                self.target_dls = torch.utils.data.DataLoader(
                uda,
                num_workers=dataset.N_WORKERS,
                batch_size=16)
            else:
                print(f"number of source samples: {len(in_)}")
                self.source_dls.append(torch.utils.data.DataLoader(
                in_,
                num_workers=dataset.N_WORKERS,
                batch_size=16))
                self.source_frac.append(len(in_))
                # print(hparams['batch_size'])
        total = sum(self.source_frac)
        self.source_frac = [ele / total for ele in self.source_frac]
        print(self.source_frac)
    
if __name__ == '__main__':
    dataset = 'VLCS'
    hparams = hparams_registry.default_hparams('fedgp', dataset)
    data_dir = '/shared/rsaas/enyij2/domainbed/data'
    domainbed_dataset = domainbedDataset(hparams, data_dir, dataset, test_env=0, target_ratio=0.05)

    algorithm = domainbedNet(domainbed_dataset.dataset.input_shape, domainbed_dataset.dataset.num_classes,
        len(domainbed_dataset.dataset) - 1, hparams)
    
    algorithm.get_gradients(domainbed_dataset.target_dls)
'''
Code is adapted from the following link:
https://github.com/uiuc-federated-learning/ml-fault-injector/blob/master/federated.py &
https://github.com/Xtra-Computing/NIID-Bench/blob/5371adbff98156793a413c7658923673b4aef7d7/experiments.py

'''

import argparse
import copy
import json
import os
import time
import copy
import random
import gc

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torchvision import datasets, transforms
from torch import nn
from tqdm import tqdm
from imgaug import augmenters as iaa
import imgaug as ia
from PIL import Image
# from sklearn.metrics.pairwise import cosine_similarity
from noniid_utils import *
from models.noniid_models import *
from torch.utils.data import SubsetRandomSampler, WeightedRandomSampler
from models.resnet import ResNetClassifier, ResNetOrig
from models.amazon import AmazonMLP, AmazonClassifier, AmazonNN
from models.cnn import CNN
from utils import constants
from utils.data_sampler import get_subset_indices, get_train_valid_indices
from utils.utils import deterministic
from torch.multiprocessing import Pool
from empirical_metric import empirical_metrics_batch

def set_device():
    global device
    device = torch.device(0 if torch.cuda.is_available() else "cpu")

def init_nets(net_configs, dropout_p, n_parties, args):

    nets = {net_i: None for net_i in range(n_parties)}

    if args.dataset in {'mnist', 'cifar10', 'svhn', 'fmnist'}:
        n_classes = 10
    elif args.dataset == 'celeba':
        n_classes = 2
    elif args.dataset == 'cifar100':
        n_classes = 100
    elif args.dataset == 'tinyimagenet':
        n_classes = 200
    elif args.dataset == 'femnist':
        n_classes = 62
    elif args.dataset == 'emnist':
        n_classes = 47
    elif args.dataset in {'a9a', 'covtype', 'rcv1', 'SUSY'}:
        n_classes = 2
    if args.use_projection_head:
        add = ""
        if "mnist" in args.dataset and args.model == "simple-cnn":
            add = "-mnist"
        for net_i in range(n_parties):
            net = ModelFedCon(args.model+add, args.out_dim, n_classes, net_configs)
            nets[net_i] = net
    else:
        if args.alg == 'moon':
            add = ""
            if "mnist" in args.dataset and args.model == "simple-cnn":
                add = "-mnist"
            for net_i in range(n_parties):
                net = ModelFedCon_noheader(args.model+add, args.out_dim, n_classes, net_configs)
                nets[net_i] = net
        else:
            for net_i in range(n_parties):
                if args.dataset == "generated":
                    net = PerceptronModel()
                elif args.model == "mlp":
                    if args.dataset == 'covtype':
                        input_size = 54
                        output_size = 2
                        hidden_sizes = [32,16,8]
                    elif args.dataset == 'a9a':
                        input_size = 123
                        output_size = 2
                        hidden_sizes = [32,16,8]
                    elif args.dataset == 'rcv1':
                        input_size = 47236
                        output_size = 2
                        hidden_sizes = [32,16,8]
                    elif args.dataset == 'SUSY':
                        input_size = 18
                        output_size = 2
                        hidden_sizes = [16,8]
                    net = FcNet(input_size, hidden_sizes, output_size, dropout_p)
                elif args.model == "vgg":
                    net = vgg11()
                elif args.model == "simple-cnn":
                    if args.dataset in ("cifar10", "cinic10", "svhn"):
                        net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
                    elif args.dataset in ("mnist", 'femnist', 'fmnist'):
                        net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
                    elif args.dataset == 'celeba':
                        net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=2)
                elif args.model == "vgg-9":
                    if args.dataset in ("mnist", 'femnist'):
                        net = ModerateCNNMNIST()
                    elif args.dataset in ("cifar10", "cinic10", "svhn"):
                        # print("in moderate cnn")
                        net = ModerateCNN()
                    elif args.dataset == 'celeba':
                        net = ModerateCNN(output_dim=2)
                elif args.model == "resnet":
                    if args.pretrained:
                        net = ResNetClassifier(num_classes=n_classes)
                    else:
                        net = ResNetClassifier(num_classes=n_classes, pretrained=False)
                elif args.model == "vgg16":
                    net = vgg16()
                else:
                    print("not supported yet")
                    exit(1)
                nets[net_i] = net

    return nets #, model_meta_data, layer_type



def train(args, da_phase, model, criterion: torch.nn.Module, train_dl):
    global device

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.source_lr if da_phase=='source' else args.target_lr)
    model.train()
    grads_all_epochs = []
    num_epochs = args.num_source_epochs if da_phase == 'source' else args.num_target_epochs

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0
        y_true_list = list()
        y_pred_list = list()
        grads = [] # length N - number of batches


        with tqdm(train_dl, unit="batch") as tepoch:
            for (imgs, labels) in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                inputs = imgs.to(device)
                labels = labels.long().to(device)
                model_init = copy.deepcopy(model)
                
                
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                tepoch.set_postfix(loss=loss.item())
                for i in range(len(outputs)):
                    y_true_list.append(labels[i].cpu().data.tolist())

                # Backward pass
                loss.backward()
                optimizer.step()
                
                cur_grad = []
                
                # we use the model update as the grad
                cur_grad = get_model_updates(model_init, model)
                grads.append(cur_grad)

                # Keep track of performance metrics (loss is mean-reduced)
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs.data, 1)
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / len(y_true_list)
            epoch_acc = float(running_corrects) / len(y_true_list)
        
        grads = torch.stack(grads) # [Number of batches, m]
        grads_all_epochs.append(grads)
    
    grads_all_epochs = torch.mean(torch.stack(grads_all_epochs),dim=0)
    grads_all_epochs = grads_all_epochs * (args.target_lr / args.source_lr)
    if da_phase == 'source':
        grads_all_epochs = torch.mean(grads_all_epochs, dim=0) # get average grad across batches

    # Keep track of current training loss and accuracy
    final_train_loss = epoch_loss
    final_train_acc = epoch_acc


    return model, grads_all_epochs, (final_train_loss, final_train_acc, None)

def test(args: argparse.Namespace, model: torch.nn.Module,
         criterion: torch.nn.Module, test_loader: torch.utils.data.DataLoader):
    global device

    model.to(device)
    model.eval()
    trial_results = dict()

    running_loss = 0.0
    running_corrects = 0
    y_true_list = list()
    y_pred_list = list()

    # Iterate over dataloader
    for (imgs, labels) in test_loader:
        inputs = imgs.to(device)
        labels = labels.long().to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            for i in range(len(outputs)):
                y_true_list.append(labels[i].cpu().data.tolist())

            # Keep track of performance metrics
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs.data, 1)
            running_corrects += torch.sum(preds == labels.data).item()

    test_loss = running_loss / len(y_true_list)
    test_acc = float(running_corrects) / len(y_true_list)

    print('Test Loss: {:.4f} Acc: {:.4f}'.format(
          test_loss, test_acc), flush=True)
    print(flush=True)

    return (test_loss, test_acc, None) 

def average_weights(w, alpha):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = torch.zeros_like(w_avg[key]).float()
        for i in range(len(w)):
            w_avg[key] += w[i][key] * alpha[i]

    return w_avg

def update_dict(old_model_dict, new_model_dict, alpha):
    new_w = copy.deepcopy(old_model_dict)
    for key in new_w.keys():
        new_w[key] = torch.zeros_like(new_w[key]).float()
        new_w[key] = old_model_dict[key] * alpha + new_model_dict[key] * (1-alpha)
    return new_w

def update_global(args, local_models_dict, old_global_model_dict, finetune_global_model_dict, clients_size, clients_size_frac, cur_epoch, beta_GP):
    ret_dict = copy.deepcopy(old_global_model_dict)
    b = args.proj_w
    part_r = args.part_rate
    indices = np.random.choice(len(clients_size), int(np.ceil(len(clients_size) * part_r)), replace=False)
    cur_sum = sum([clients_size[i] for i in range(len(clients_size)) if i in indices])
    cos = torch.nn.CosineSimilarity()
    for key in ret_dict.keys():
        if ret_dict[key].shape != torch.Size([]):
            global_grad = finetune_global_model_dict[key] - old_global_model_dict[key]
            for idx, local_dict in enumerate(local_models_dict):
                local_grad = local_dict[key] - old_global_model_dict[key]
                cur_sim = cos(global_grad.reshape(1,-1), local_grad.reshape(1,-1))
                if cur_sim > 0:
                    ret_dict[key] = ret_dict[key] + beta_GP[idx] * (args.target_lr / args.source_lr) * ((args.n_target_samples/args.target_batch_size)/(clients_size[idx]/args.source_batch_size)) * clients_size_frac[idx] * cur_sim * local_grad
                ret_dict[key] = ret_dict[key] + (1-beta_GP[idx]) * global_grad * clients_size_frac[idx]
        else:
            ret_dict[key] = old_global_model_dict[key]
    return ret_dict

def update_global_reverse(args, local_models_dict, old_global_model_dict, finetune_global_model_dict, clients_size, clients_size_frac, cur_epoch):
    ret_dict = copy.deepcopy(old_global_model_dict)
    b = args.proj_w
    count = 0
    cos = torch.nn.CosineSimilarity()
    for key in ret_dict.keys():
        if ret_dict[key].shape != torch.Size([]):
            global_grad = finetune_global_model_dict[key] - old_global_model_dict[key]
            for idx, local_dict in enumerate(local_models_dict):
                if idx in indices:
                    local_grad = local_dict[key] - old_global_model_dict[key]
                    cur_sim = cos(global_grad.reshape(1,-1), local_grad.reshape(1,-1))
                    if cur_sim > 0:
                        ret_dict[key] = ret_dict[key] + b * clients_size_frac[idx] * cur_sim * global_grad
                    else:
                        count += 1
            ret_dict[key] = ret_dict[key] + (1-b) * global_grad
        else:
            ret_dict[key] = old_global_model_dict[key]
    print(f'negative times {count}')
    return ret_dict

def update_global_convex(args, local_models_dict, old_global_model_dict, finetune_global_model_dict, clients_size, clients_size_frac, cur_epoch, beta_DA):
    ret_dict = copy.deepcopy(old_global_model_dict)
    b = args.proj_w
    part_r = args.part_rate
    indices = np.random.choice(len(clients_size), int(np.ceil(len(clients_size) * part_r)), replace=False)
    cur_sum = sum([clients_size[i] for i in range(len(clients_size)) if i in indices])
    cos = torch.nn.CosineSimilarity()
    for key in ret_dict.keys():
        if ret_dict[key].shape != torch.Size([]):
            global_grad = finetune_global_model_dict[key] - old_global_model_dict[key]
            for idx, local_dict in enumerate(local_models_dict):
                if idx in indices:
                    local_grad = local_dict[key] - old_global_model_dict[key]
                    ret_dict[key] = ret_dict[key] + beta_DA[idx] * (clients_size[idx] / cur_sum) * local_grad
                    ret_dict[key] = ret_dict[key] + (1-beta_DA[idx]) * global_grad * (clients_size[idx] / cur_sum)
        else:
            ret_dict[key] = old_global_model_dict[key]
    return ret_dict

# get the grad updates
def get_model_updates(init_model, new_model):
    init = get_param_list(init_model)
    new = get_param_list(new_model)
    
    return (new - init)

def get_param_list(model):
    m_dict = model.state_dict()
    param = []
    for key in m_dict.keys():
        if m_dict[key].shape != torch.Size([]):
            param.append(m_dict[key].detach().clone().flatten())
    return torch.cat(param)

if __name__ == '__main__':
    set_device()
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_dir', type=str, default='fl_noniid')
    parser.add_argument('--iter_idx', type=str, default='0')
    parser.add_argument('--resnet', type=str, default='resnet18')
    parser.add_argument('--load_trained_model', action='store_true')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--num_source_epochs', type=int, default=1)
    parser.add_argument('--num_target_epochs', type=int, default=1)
    parser.add_argument('--num_global_epochs', type=int, default=50)
    parser.add_argument('--source_lr', type=float, default=0.01)
    parser.add_argument('--target_lr', type=float, default=0.005)
    parser.add_argument('--source_batch_size', type=int, default=64)
    parser.add_argument('--target_batch_size', type=int, default=16)
    parser.add_argument('--no_drop_last', action='store_false')
    parser.add_argument('--train_seed', type=int, default=8)
    parser.add_argument('--data_sampler_seed', type=int, default=8)
    parser.add_argument('--n_target_samples', type=int, default=100)
    parser.add_argument('--target_frac', type=float, default=0.1)
    parser.add_argument('--valid_fraction', type=float, default=None)
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--data_aug_times', type=int, default=1)
    parser.add_argument('--use_sim', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--datadir', type=str, default='/data/enyij2/msda/noniid')
    parser.add_argument('--n_parties', type=int, default=5,  help='number of workers in a distributed cluster')
    parser.add_argument('--partition', type=str, default='homo', help='the data partitioning strategy')
    parser.add_argument('--model', type=str, default='simple-cnn', help='neural network used in training')
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--noise', type=float, default=0, help='how much noise we add to some party')
    parser.add_argument('--use_projection_head', type=bool, default=False, help='whether add an additional header to model or not (see MOON)')
    parser.add_argument('--alg', type=str, default='fedavg',
                            help='fl algorithms: fedavg/fedprox/scaffold/fednova/moon')
    parser.add_argument('--proj_w', type=float, required=False, default=0.5, help='how much weight for leveraging info from the source domains')
    parser.add_argument('--flip', type=float, required=False, default=0, help='whether to flip the gradient with some probability')
    parser.add_argument('--agg_before_gp', action='store_true', help='whether to avg the weights before gp')
    parser.add_argument('--convex_agg', action='store_true', help='whether to do convex combination with fedavg')
    parser.add_argument('--reverse_gp', action='store_true', help='whether to do reverse gradient projection')
    parser.add_argument('--pretrained', action='store_true', help='whether to pretrain the original model')
    parser.add_argument('--part_rate', type=float, required=False, default=1, help='subsample how many clients for updating model')

    args = parser.parse_args()
    timestamp = time.strftime("%Y-%m-%d-%H%M")

    exp_dir = os.path.join('experiments', args.exp_dir)
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, f'args_{args.iter_idx}.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)
    deterministic(args.train_seed)

    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)

    n_classes = len(np.unique(y_train))


 
    
    # Initialize the server & clients' models
    local_models = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
    global_model = local_models[args.n_parties-1]
    del local_models[args.n_parties-1]
    global_model.to(device)

    # Initialize the datasets & data loader
    clients_dls = {'train':[], 'test':[]}
    server_dls = {'train':[], 'test':[]}
    # construct clients' dataloaders
    for net_id in range(args.n_parties): # 0-9
        
        if net_id < args.n_parties - 1:
            dataidxs = net_dataidx_map[net_id]
            train_dl, test_dl, _, _ = get_dataloader(args.dataset, args.datadir, args.source_batch_size, 32, dataidxs, 0)
            clients_dls['train'].append(train_dl)
            clients_dls['test'].append(test_dl)
        else:
            dataidxs = net_dataidx_map[net_id]
            if args.dataset == 'cifar10':
                if args.partition == 'homo':
                    train_dl, test_dl, train_ds, test_ds = get_dataloader(args.dataset, args.datadir, args.target_batch_size, 32, None, args.noise)
                    randperm = torch.randperm(len(train_ds))
                    indices = randperm[:int(len(train_ds)*0.075)]
                    args.n_target_samples = int(len(train_ds)*0.075)
                    rest_indices = randperm[int(len(train_ds)*0.075):int(len(train_ds)*0.5)]#(len(clients_dls['train'][0])*args.source_batch_size)]
                else:
                    train_dl, test_dl, train_ds, test_ds = get_dataloader(args.dataset, args.datadir, args.target_batch_size, 32, dataidxs, args.noise)
                    randperm = torch.randperm(len(train_ds))
                    indices = randperm[:int(len(train_ds)*args.target_frac)]
                    args.n_target_samples = int(len(train_ds)*args.target_frac)
                    rest_indices = randperm[int(len(train_ds)*args.target_frac):]
            else:
                train_dl, test_dl, train_ds, test_ds = get_dataloader(args.dataset, args.datadir, args.target_batch_size, 32, dataidxs, args.noise)
                randperm = torch.randperm(len(train_ds))
                if args.partition == 'homo':
                    indices = randperm[:args.n_target_samples]
                    rest_indices = randperm[args.n_target_samples:]
                else:
                    indices = randperm[:int(len(train_ds)*0.15)]
                    args.n_target_samples = int(len(train_ds)*0.15)
                    rest_indices = randperm[int(len(train_ds)*0.15):]
            server_dls['train'].append(train_dl)
            

            cur_sampler = SubsetRandomSampler(indices)
            cur_sampler_rest = SubsetRandomSampler(rest_indices)
            perturb_dl = torch.utils.data.DataLoader(train_ds, shuffle=False, batch_size=args.target_batch_size, sampler=cur_sampler)
            unlabeled_dl = torch.utils.data.DataLoader(train_ds, shuffle=False, batch_size=args.source_batch_size, sampler=cur_sampler_rest)
            if args.partition == 'homo':
                server_dls['test'].append(test_dl)
            else:
                server_dls['test'].append(unlabeled_dl)   

    # initialize datalaoders, models, optimizer, criterions
    
    num_clients = args.n_parties - 1
    dict_client = dict()
    for i in range(num_clients):
        dict_client.update({i: []})

    clients_size = [len(clients_dls['train'][i])*args.source_batch_size for i in range(num_clients)]
    clients_size_frac = np.array(clients_size) / sum(clients_size)
    print(clients_size, clients_size_frac)

    clients_grads = [None] * num_clients
    cos_sim = [None] * num_clients
    global_model_dict = global_model.state_dict()

    criterion = torch.nn.CrossEntropyLoss().to(device)
    clients_results = dict()
    clients_results['train'] = dict()
    clients_results['test_s'] = dict()
    clients_results['test_t'] = dict()
    clients_results['train']['loss'] = copy.deepcopy(dict_client)
    clients_results['train']['acc'] = copy.deepcopy(dict_client)
    clients_results['train']['auc'] = copy.deepcopy(dict_client)
    clients_results['test_s']['loss'] = copy.deepcopy(dict_client)
    clients_results['test_s']['acc'] = copy.deepcopy(dict_client)
    clients_results['test_s']['auc'] = copy.deepcopy(dict_client)
    clients_results['test_t']['loss'] = copy.deepcopy(dict_client)
    clients_results['test_t']['acc'] = copy.deepcopy(dict_client)
    clients_results['test_t']['auc'] = copy.deepcopy(dict_client)

    server_results = dict()
    server_results['train'] = dict()
    server_results['test'] = dict()
    server_results['train']['loss'] = []
    server_results['train']['acc'] = []
    server_results['train']['auc'] = []
    server_results['test']['loss'] = []
    server_results['test']['acc'] = []
    server_results['test']['auc'] = []

    # do fedavg for 2 epochs, to have a good initialization
    if args.load_trained_model:
        global_model.load_state_dict(torch.load(args.model_path))
    elif args.proj_w > 0:
        for _ in range(2):
            for idx in range(num_clients):
                local_models[idx].load_state_dict(global_model_dict)
                local_models[idx], _, (loss, acc, auc) = train(args, 'source', copy.deepcopy(local_models[idx]), criterion, clients_dls['train'][idx])
            global_model_dict = average_weights([model.state_dict() for model in local_models.values()], clients_size_frac)
            global_model.load_state_dict(global_model_dict)

    if args.dataset != 'cifar10':
        args.source_lr *= 0.1
        args.target_lr *= 0.1
    else:
        args.source_lr *= 0.35
        args.target_lr *= 0.35
    
    print(args.source_lr, args.target_lr)
    for i in range(args.num_global_epochs):
        # training local models
        source_grads = []
        target_grads = None
        if args.proj_w > 0:
            for idx in range(num_clients):
                local_models[idx].load_state_dict(global_model_dict)
                local_models[idx], source_grad, (loss, acc, auc) = train(args, 'source', copy.deepcopy(local_models[idx]), criterion, clients_dls['train'][idx])
                clients_results['train']['loss'][idx].append(loss)
                clients_results['train']['acc'][idx].append(acc)
                clients_results['train']['auc'][idx].append(auc)
                source_grads.append(source_grad)
                gc.collect()
                torch.cuda.empty_cache()
        
        # averaging the weights
        if args.use_sim:
            if args.agg_before_gp:
                global_model_dict = average_weights([model.state_dict() for model in local_models.values()], clients_size_frac)
                global_model.load_state_dict(global_model_dict)
            new_model, target_grads, (loss, acc, auc) = train(args, 'target', copy.deepcopy(global_model), criterion, perturb_dl)
            server_results['train']['loss'].append(loss)
            server_results['train']['acc'].append(acc)
            server_results['train']['auc'].append(auc)
            if args.proj_w > 0:
                if args.reverse_gp:
                    global_model_dict = update_global_reverse(args, [model.state_dict() for model in local_models.values()], global_model.state_dict(), new_model.state_dict(), clients_size, clients_size_frac, i)
                else:
                    metrics = empirical_metrics_batch(args.target_batch_size, source_grads, target_grads)
                    beta_GP = metrics.return_fedgp_with_thresh_beta()
                    global_model_dict = update_global(args, [model.state_dict() for model in local_models.values()], global_model.state_dict(), new_model.state_dict(), clients_size, clients_size_frac, i, beta_GP)
                global_model.load_state_dict(global_model_dict)
            else:
                global_model = copy.deepcopy(new_model)
        elif args.convex_agg:
            new_model, target_grads, (loss, acc, auc) = train(args, 'target', copy.deepcopy(global_model), criterion, perturb_dl)
            server_results['train']['loss'].append(loss)
            server_results['train']['acc'].append(acc)
            server_results['train']['auc'].append(auc)
            if args.proj_w > 0:
                metrics = empirical_metrics_batch(args.target_batch_size, source_grads, target_grads)
                beta_DA = metrics.return_fedda_beta()
                global_model_dict = update_global_convex(args, [model.state_dict() for model in local_models.values()], global_model.state_dict(), new_model.state_dict(), clients_size, clients_size_frac, i, beta_DA)
                global_model.load_state_dict(global_model_dict)
            else:
                global_model = copy.deepcopy(new_model)
        else:
            global_model_dict = average_weights([model.state_dict() for model in local_models.values()], clients_size_frac)
            global_model.load_state_dict(global_model_dict)
            if args.finetune:
                global_model, (loss, acc, auc) = train(args, 'target', global_model, criterion, perturb_dl)
                server_results['train']['loss'].append(loss)
                server_results['train']['acc'].append(acc)
                server_results['train']['auc'].append(auc)
                global_model_dict = global_model.state_dict()

        print('testing global model on its target domain')
        (loss, acc, auc) = test(args, global_model, criterion, server_dls['test'][0])
        server_results['test']['loss'].append(loss)
        server_results['test']['acc'].append(acc)
        server_results['test']['auc'].append(auc)

    with open(os.path.join(exp_dir,(f'clients_results_{args.iter_idx}.json')), 'w') as fp:
            json.dump(clients_results, fp, indent=4)
    fp.close()
    
    with open(os.path.join(exp_dir,(f'server_results_{args.iter_idx}.json')), 'w') as fp:
            json.dump(server_results, fp, indent=4)
    fp.close()
    
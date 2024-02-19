# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid
import copy

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
from tqdm import tqdm

from domainbed.lib import misc

from flamby.datasets.fed_heart_disease import (
    BATCH_SIZE,
    LR,
    NUM_EPOCHS_POOLED,
    Baseline,
    BaselineLoss,
    metric,
    NUM_CLIENTS,
    get_nb_max_rounds
)
from flamby.datasets.fed_heart_disease import FedHeartDisease as FedDataset
from domainbed.lib import misc
from flamby.utils import evaluate_model_on_tests

def train(args, da_phase, model, criterion, train_dl):
    global device

    model.to(device)
    lr = LR if da_phase=='source' else LR * args.lr_ratio
    optimizer = torch.optim.Adam(model.parameters(), lr= lr)

    model.train()
    num_epochs = args.num_source_epochs if da_phase == 'source' else args.num_target_epochs

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0
        y_true_list = list()
        y_pred_list = list()


        with tqdm(train_dl, unit="batch") as tepoch:
            for (imgs, labels) in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                inputs = imgs.to(device)
                labels = labels.to(device)


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

                # Keep track of performance metrics (loss is mean-reduced)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += metric(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())


            epoch_loss = running_loss / len(y_true_list)
            epoch_acc = float(running_corrects) / len(train_dl)

    # Keep track of current training loss and accuracy
    final_train_loss = epoch_loss
    final_train_acc = epoch_acc

    return model, (final_train_loss, final_train_acc, None)

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
            running_corrects += metric(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())

    test_loss = running_loss / len(y_true_list)
    test_acc = float(running_corrects) / len(test_loader)

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

def update_global(args, local_models_dict, old_global_model_dict, finetune_global_model_dict, clients_size, clients_size_frac, cur_epoch):
    ret_dict = copy.deepcopy(old_global_model_dict)
    b = args.proj_w
    cos = torch.nn.CosineSimilarity()
    for key in ret_dict.keys():
        if ret_dict[key].shape != torch.Size([]):
            global_grad = finetune_global_model_dict[key] - old_global_model_dict[key]
            for idx, local_dict in enumerate(local_models_dict):
                local_grad = local_dict[key] - old_global_model_dict[key]
                cur_sim = cos(global_grad.reshape(1,-1), local_grad.reshape(1,-1))
                if cur_sim > 0:
                    ret_dict[key] = ret_dict[key] + b * args.lr_ratio * (args.num_target_epochs / args.num_source_epochs) * ((args.n_target_samples/args.target_batch_size)/(clients_size[idx]/BATCH_SIZE)) * clients_size_frac[idx] * cur_sim * local_grad
            ret_dict[key] = ret_dict[key] + (1-b) * global_grad
        else:
            ret_dict[key] = torch.zeros_like(old_global_model_dict[key]).float()
            for idx, local_dict in enumerate(local_models_dict):
                ret_dict[key] += clients_size_frac[idx] * local_dict[key]
    return ret_dict

def update_global_convex(args, local_models_dict, old_global_model_dict, finetune_global_model_dict, clients_size, clients_size_frac, cur_epoch):
    ret_dict = copy.deepcopy(old_global_model_dict)
    b = args.proj_w
    for key in ret_dict.keys():
        if ret_dict[key].shape != torch.Size([]):
            global_grad = finetune_global_model_dict[key] - old_global_model_dict[key]
            for idx, local_dict in enumerate(local_models_dict):
                local_grad = local_dict[key] - old_global_model_dict[key]
                ret_dict[key] = ret_dict[key] + b * clients_size_frac[idx] * local_grad
            ret_dict[key] = ret_dict[key] + (1-b) * global_grad
        else:
            ret_dict[key] = torch.zeros_like(old_global_model_dict[key]).float()
            for idx, local_dict in enumerate(local_models_dict):
                ret_dict[key] += clients_size_frac[idx] * local_dict[key]
    return ret_dict

# get the grad updates
def get_model_updates(init_model, new_model):
    ret_updates = []
    init = get_param_list(init_model)
    new = get_param_list(new_model)
    return (new - init).reshape(1, -1)

def get_param_list(model):
	m_dict = model.state_dict()
	param = []
	for key in m_dict.keys():
		param.append(np.linalg.norm(m_dict[key]))
	return np.array(param)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MSDA')
    # arguments from fedgp
    parser.add_argument('--exp_dir', type=str, default='fl_flamby')
    parser.add_argument('--iter_idx', type=str, default='0')
    parser.add_argument('--load_trained_model', action='store_true')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--lr_ratio', type=float, default=1.0)
    parser.add_argument('--num_source_epochs', type=int, default=1)
    parser.add_argument('--num_target_epochs', type=int, default=1)
    parser.add_argument('--num_global_epochs', type=int, default=50)
    parser.add_argument('--auto_lr_ratio', type=float, default=1)
    parser.add_argument('--use_sim', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--proj_w', type=float, required=False, default=0.5, help='how much weight for leveraging info from the source domains')
    parser.add_argument('--convex_agg', action='store_true', help='whether to do convex combination with fedavg')
    parser.add_argument('--early_stop', action='store_true', help='whether to use validation set for early stopping')
    parser.add_argument('--target_batch_size', type=int, default=BATCH_SIZE, help='target domain train batch size')
    # arguments from domainbed
    parser.add_argument('--dataset', type=str, default="heart")
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0]) # which domain to be target domain.
    parser.add_argument('--uda_holdout_fraction', type=float, default=0.2,
        help="For domain adaptation, % of test to use unlabeled for training.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    global device

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.

    # in-split: training data for each domain
    # out-split: testing data for each domain
    # uda-split: finetuning data for the target domain
    clients_dls = {'train':[], 'test':[]}
    server_dls = {'train':[], 'test':[], 'val':[]}
    clients = []
    server = []

    for idx in range(NUM_CLIENTS):
        if idx in args.test_envs:
            target_train = FedDataset(center = idx, train = True, pooled = False)
            uda, in_ = misc.split_dataset(target_train,
                int(len(target_train)*args.uda_holdout_fraction),
                args.seed)
            print(len(uda), len(in_), len(target_train))
            args.n_target_samples = len(uda)
            server_dls['train'].append(torch.utils.data.DataLoader(uda,
                batch_size = args.target_batch_size,
                shuffle = False,
                num_workers = 0,
            ))
            server_dls['test'].append(
                 torch.utils.data.DataLoader(
                FedDataset(center = idx, train = False, pooled = False),
                batch_size = BATCH_SIZE,
                shuffle = False,
                num_workers = 0,
                 ))
            server.append(idx)
        else:
            clients.append(idx)
            clients_dls['train'].append(
                torch.utils.data.DataLoader(
                FedDataset(center = idx, train = True, pooled = False),
                batch_size = BATCH_SIZE,
                shuffle = True,
                num_workers = 0
            )
            )

            clients_dls['test'].append(
                torch.utils.data.DataLoader(
                FedDataset(center = idx, train = False, pooled = False),
                batch_size = BATCH_SIZE,
                shuffle = False,
                num_workers = 0,
            )
            )

        
    
    exp_dir = os.path.join('experiments', args.exp_dir, args.dataset, str(server[0]))
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, f'args_{args.iter_idx}.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)


    print(f'target:{server}, sources:{clients}')
    num_clients = len(clients)
    dict_client = dict()
    for i in range(num_clients):
        dict_client.update({clients[i]: []})
    clients_size = [len(clients_dls['train'][i])*BATCH_SIZE for i in range(num_clients)]
    clients_size_frac = np.array(clients_size) / sum(clients_size)
    print(clients_size, clients_size_frac)


    # intialize models
    
    global_model = Baseline()
    
    global_model.to(device)
    global_model_dict = global_model.state_dict()

    local_models = [Baseline() for _ in range(num_clients)]

    criterion = BaselineLoss()

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
    server_results['test']['metric'] = []
    server_results['best_val_test'] = dict()
    server_results['best_val_test']['loss'] = None
    server_results['best_val_test']['acc'] = None
    
    patience = 5
    best_val_loss = np.inf
    best_val_acc = 0
    best_model_weights = None
    epochs_no_improve = 0


    # do fedavg for 2 epochs, to have a good initialization
    if args.load_trained_model:
        global_model.load_state_dict(torch.load(args.model_path))
    elif args.proj_w > 0:
        num_init = 5
        for _ in range(num_init):
            for idx in range(num_clients):
                local_models[idx].load_state_dict(global_model_dict)
                local_models[idx], (loss, acc, auc) = train(args, 'source', copy.deepcopy(local_models[idx]), criterion, clients_dls['train'][idx])
            global_model_dict = average_weights([model.state_dict() for model in local_models], clients_size_frac)
            global_model.load_state_dict(global_model_dict)

    LR = LR * args.auto_lr_ratio

    print(LR, BATCH_SIZE, global_model)
    
    for i in range(args.num_global_epochs):
        # training local models
        if args.proj_w > 0:
            for idx in range(num_clients):
                local_models[idx].load_state_dict(global_model_dict)
                local_models[idx], (loss, acc, auc) = train(args, 'source', copy.deepcopy(local_models[idx]), criterion, clients_dls['train'][idx])
                clients_results['train']['loss'][clients[idx]].append(loss)
                clients_results['train']['acc'][clients[idx]].append(acc)
                clients_results['train']['auc'][clients[idx]].append(auc)
        
        # averaging the weights
        if args.use_sim:
            new_model, (loss, acc, auc) = train(args, 'target', copy.deepcopy(global_model), criterion, server_dls['train'][0])
            server_results['train']['loss'].append(loss)
            server_results['train']['acc'].append(acc)
            server_results['train']['auc'].append(auc)
            if args.proj_w > 0:
                global_model_dict = update_global(args, [model.state_dict() for model in local_models], global_model.state_dict(), new_model.state_dict(), clients_size, clients_size_frac, i)
                global_model.load_state_dict(global_model_dict)
            else:
                global_model = copy.deepcopy(new_model)
        elif args.convex_agg:
            new_model, (loss, acc, auc) = train(args, 'target', copy.deepcopy(global_model), criterion, server_dls['train'][0])
            server_results['train']['loss'].append(loss)
            server_results['train']['acc'].append(acc)
            server_results['train']['auc'].append(auc)
            if args.proj_w > 0:
                global_model_dict = update_global_convex(args, [model.state_dict() for model in local_models], global_model.state_dict(), new_model.state_dict(), clients_size, clients_size_frac, i)
                global_model.load_state_dict(global_model_dict)
            else:
                global_model = copy.deepcopy(new_model)
        else:
            global_model_dict = average_weights([model.state_dict() for model in local_models], clients_size_frac)
            global_model.load_state_dict(global_model_dict)
            if args.finetune:
                global_model, (loss, acc, auc) = train(args, 'target', global_model, criterion, server_dls['train'][0])
                server_results['train']['loss'].append(loss)
                server_results['train']['acc'].append(acc)
                server_results['train']['auc'].append(auc)
                global_model_dict = global_model.state_dict()

        print('testing global model on its target domain')

        dict_cindex = evaluate_model_on_tests(global_model, server_dls['test'], metric)
        print(dict_cindex)

        server_results['test']['metric'].append(list(dict_cindex.values())[0])
        
        # test on the validation set
        if args.early_stop:
            (val_loss, val_acc, _) = test(args, global_model, criterion, server_dls['val'][0])
            # print(val_loss, val_acc)
            if val_loss < best_val_loss and val_acc > best_val_acc:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_model_weights = global_model.state_dict()
                epochs_no_improve = 0
                server_results['best_val_test']['loss'] = loss
                server_results['best_val_test']['acc'] = acc
            else:
                epochs_no_improve += 1
        
            # Check if early stopping criteria are met
            if epochs_no_improve == patience:
                print("Early stopping! No improvement in validation loss for {} epochs.".format(patience))
                break
    
    with open(os.path.join(exp_dir,(f'server_results_{args.iter_idx}.json')), 'w') as fp:
            json.dump(server_results, fp, indent=4)
    fp.close()
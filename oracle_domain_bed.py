
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

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from models.domainbed_net import *
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

def train(args, hparams, da_phase, model, criterion: torch.nn.Module, train_dl):
    global device

    model.to(device)
    lr = hparams["lr"] if da_phase=='source' else hparams["lr"]/4
    optimizer = torch.optim.Adam(model.parameters(), lr= lr, weight_decay=hparams['weight_decay'])
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
                labels = labels.long().to(device)

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
                _, preds = torch.max(outputs.data, 1)
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / len(y_true_list)
            epoch_acc = float(running_corrects) / len(y_true_list)

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

def update_global(args, hparams, local_models_dict, old_global_model_dict, finetune_global_model_dict, clients_size, clients_size_frac, cur_epoch):
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
                    ret_dict[key] = ret_dict[key] + b * (0.25) * ((args.n_target_samples/16)/(clients_size[idx]/hparams['batch_size'])) * clients_size_frac[idx] * cur_sim * local_grad
            ret_dict[key] = ret_dict[key] + (1-b) * global_grad
        else:
            ret_dict[key] = old_global_model_dict[key]
    return ret_dict

def update_global_convex(args, local_models_dict, old_global_model_dict, finetune_global_model_dict, clients_size, clients_size_frac, cur_epoch):
    ret_dict = copy.deepcopy(old_global_model_dict)
    b = args.proj_w
    cos = torch.nn.CosineSimilarity()
    for key in ret_dict.keys():
        if ret_dict[key].shape != torch.Size([]):
            global_grad = finetune_global_model_dict[key] - old_global_model_dict[key]
            for idx, local_dict in enumerate(local_models_dict):
                local_grad = local_dict[key] - old_global_model_dict[key]
                ret_dict[key] = ret_dict[key] + b * clients_size_frac[idx] * local_grad
            ret_dict[key] = ret_dict[key] + (1-b) * global_grad
        else:
            ret_dict[key] = old_global_model_dict[key]
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
    parser.add_argument('--exp_dir', type=str, default='fl_domainbed')
    parser.add_argument('--iter_idx', type=str, default='0')
    parser.add_argument('--load_trained_model', action='store_true')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--num_source_epochs', type=int, default=1)
    parser.add_argument('--num_target_epochs', type=int, default=1)
    parser.add_argument('--num_global_epochs', type=int, default=50)
    parser.add_argument('--use_sim', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--proj_w', type=float, required=False, default=0.5, help='how much weight for leveraging info from the source domains')
    parser.add_argument('--convex_agg', action='store_true', help='whether to do convex combination with fedavg')
    # arguments from domainbed
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="fedgp")
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0]) # which domain to be target domain.
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0.15,
        help="For domain adaptation, % of test to use unlabeled for training.")
    args = parser.parse_args()

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

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

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs, hparams)
    else:
        raise NotImplementedError

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
    # clients_dls = {'train':[], 'test':[]}
    server_dls = {'train':[], 'test':[]}
    server = []
    for env_i, env in enumerate(dataset):
        uda = []

        # split training/testing data
        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))
        # split finetuning set from testing data
        if env_i in args.test_envs:
            server = dataset.ENVIRONMENTS[env_i]
            server_dls['train'].append(torch.utils.data.DataLoader(
            in_,
            num_workers=dataset.N_WORKERS,
            batch_size=hparams['batch_size']))
            server_dls['test'].append(torch.utils.data.DataLoader(
            out,
            num_workers=dataset.N_WORKERS,
            batch_size=64))
    
    exp_dir = os.path.join('experiments', args.exp_dir, args.dataset, server)
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, f'args_{args.iter_idx}.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    print(f'target:{server}')


    # intialize models
    global_model = domainbedNet(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), hparams)
    global_model.to(device)
    global_model_dict = global_model.state_dict()

    criterion = torch.nn.CrossEntropyLoss().to(device)

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

    for i in range(args.num_global_epochs):
        global_model, (loss, acc, auc) = train(args, hparams, 'target', global_model, criterion, server_dls['train'][0])
        server_results['train']['loss'].append(loss)
        server_results['train']['acc'].append(acc)
        server_results['train']['auc'].append(auc)
        global_model_dict = global_model.state_dict()

        print('testing global model on its target domain')
        (loss, acc, auc) = test(args, global_model, criterion, server_dls['test'][0])
        server_results['test']['loss'].append(loss)
        server_results['test']['acc'].append(acc)
        server_results['test']['auc'].append(auc)
    
    with open(os.path.join(exp_dir,(f'server_results_{args.iter_idx}.json')), 'w') as fp:
            json.dump(server_results, fp, indent=4)
    fp.close()

    torch.save(global_model.state_dict(),os.path.join(exp_dir,f'server_checkpoint_{args.iter_idx}.pt'))
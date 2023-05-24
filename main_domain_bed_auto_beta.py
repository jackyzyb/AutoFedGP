
import argparse
import collections
import json
import os
import random
import sys
import time
import uuid
import copy
import gc

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
from empirical_metric import empirical_metrics_batch

def train(args, hparams, da_phase, model, criterion: torch.nn.Module, train_dl):
    global device

    model.to(device)
    
    lr = hparams["lr"] if da_phase=='source' else hparams["lr"] * args.lr_ratio
    
    optimizer = torch.optim.Adam(model.parameters(), lr= lr) #, weight_decay=lr*0.1)
    
    grads_all_epochs = []

    model.train()
    num_epochs = args.num_source_epochs if da_phase == 'source' else args.num_target_epochs

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0
        y_true_list = list()
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
                if args.use_original_grad:
                    for _, param in model.named_parameters():
                        cur_grad.append(param.grad.detach().clone().flatten())
                    cur_grad = torch.cat(cur_grad)
                    grads.append(cur_grad)
                else:
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
    
    if not args.use_original_grad:
        grads_all_epochs = grads_all_epochs * args.lr_ratio
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
        # w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def update_dict(old_model_dict, new_model_dict, alpha):
    new_w = copy.deepcopy(old_model_dict)
    for key in new_w.keys():
        new_w[key] = torch.zeros_like(new_w[key]).float()
        new_w[key] = old_model_dict[key] * alpha + new_model_dict[key] * (1-alpha)
    return new_w

def update_global(args, hparams, local_models_dict, old_global_model_dict, finetune_global_model_dict, clients_size, clients_size_frac, cur_epoch, beta_GP):
    ret_dict = copy.deepcopy(old_global_model_dict)
    b = beta_GP
    # b = 0.5 * (1 - cur_epoch / args.num_global_epochs) + 0.5
    cos = torch.nn.CosineSimilarity()
    for key in ret_dict.keys():
        if ret_dict[key].shape != torch.Size([]):
            global_grad = finetune_global_model_dict[key] - old_global_model_dict[key]
            for idx, local_dict in enumerate(local_models_dict):
                local_grad = local_dict[key] - old_global_model_dict[key]
                # if key.split('.')[1] == '1':
                cur_sim = cos(global_grad.reshape(1,-1), local_grad.reshape(1,-1))
                # print(global_grad.shape, local_grad.shape)
                # print(cos(global_grad.reshape(1,-1), local_grad.reshape(1,-1)))
                # ret_dict[key] = ret_dict[key] + b * clients_size_frac[idx] * cos_sim[idx] * local_grad
                if cur_sim > 0:
                    ret_dict[key] = ret_dict[key] + beta_GP[idx] * args.lr_ratio * (args.num_target_epochs / args.num_source_epochs) * ((args.n_target_samples/args.target_batch_size)/(clients_size[idx]/hparams['batch_size'])) * clients_size_frac[idx] * cur_sim * local_grad
                # ret_dict[key] = ret_dict[key] + b * (clients_size[idx] / args.n_target_samples) * clients_size_frac[idx] * cur_sim * local_grad
                # else:
                    # ret_dict[key] = ret_dict[key] + b * clients_size_frac[idx] * local_grad
            ret_dict[key] = ret_dict[key] + (1-beta_GP[idx]) * global_grad
        else:
            ret_dict[key] = torch.zeros_like(old_global_model_dict[key]).float()
            for idx, local_dict in enumerate(local_models_dict):
                ret_dict[key] += clients_size_frac[idx] * local_dict[key]
            # ret_dict[key] = old_global_model_dict[key]
    return ret_dict

def update_global_convex(args, local_models_dict, old_global_model_dict, finetune_global_model_dict, clients_size, clients_size_frac, cur_epoch, beta_DA):
    ret_dict = copy.deepcopy(old_global_model_dict)
    # b = beta_DA
    # b = 0.5 * (1 - cur_epoch / args.num_global_epochs) + 0.5
    # cos = torch.nn.CosineSimilarity()
    for key in ret_dict.keys():
        if ret_dict[key].shape != torch.Size([]):
            global_grad = finetune_global_model_dict[key] - old_global_model_dict[key]
            for idx, local_dict in enumerate(local_models_dict):
                local_grad = local_dict[key] - old_global_model_dict[key]
                # cur_sim = cos(global_grad.reshape(1,-1), local_grad.reshape(1,-1))
                # print(global_grad.shape, local_grad.shape)
                # print(cos(global_grad.reshape(1,-1), local_grad.reshape(1,-1)))
                # ret_dict[key] = ret_dict[key] + b * clients_size_frac[idx] * cos_sim[idx] * local_grad
                ret_dict[key] = ret_dict[key] + beta_DA[idx] * clients_size_frac[idx] * local_grad
                    # ret_dict[key] = ret_dict[key] + b * (clients_size[idx] / args.n_target_samples) * clients_size_frac[idx] * cur_sim * local_grad
            ret_dict[key] = ret_dict[key] + (1-beta_DA[idx]) * global_grad
        else:
            ret_dict[key] = torch.zeros_like(old_global_model_dict[key]).float()
            for idx, local_dict in enumerate(local_models_dict):
                ret_dict[key] += clients_size_frac[idx] * local_dict[key]
            # ret_dict[key] = old_global_model_dict[key]
    return ret_dict

# get the grad updates
def get_model_updates(init_model, new_model):
    init = get_param_list(init_model)
    new = get_param_list(new_model)
    
    # print(new.shape, init.shape)
    return (new - init)

def get_param_list(model):
    m_dict = model.state_dict()
    param = []
    for key in m_dict.keys():
        if m_dict[key].shape != torch.Size([]):
            param.append(m_dict[key].detach().clone().flatten())
    return torch.cat(param)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MSDA')
    # arguments from fedgp
    parser.add_argument('--exp_dir', type=str, default='fl_domainbed_auto')
    parser.add_argument('--iter_idx', type=str, default='0')
    parser.add_argument('--load_trained_model', action='store_true')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--lr_ratio', type=float, default=0.2)
    parser.add_argument('--num_source_epochs', type=int, default=1)
    parser.add_argument('--num_target_epochs', type=int, default=1)
    parser.add_argument('--num_global_epochs', type=int, default=50)
    parser.add_argument('--target_batch_size', type=int, required=False, default=16)
    parser.add_argument('--use_sim', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--proj_w', type=float, required=False, default=0.5, help='how much weight for leveraging info from the source domains')
    parser.add_argument('--convex_agg', action='store_true', help='whether to do convex combination with fedavg')
    parser.add_argument('--use_original_grad', action='store_true', help='if true we use the original grad instead of the model updates for computing metrics')
    parser.add_argument('--log_metric', action='store_true', help='whether to log the metric contents')
    parser.add_argument('--early_stop', action='store_true', help='whether to use validation set for early stopping')
    # arguments from domainbed
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="fedgp")
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=1,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0]) # which domain to be target domain.
    # parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0.15,
        help="For domain adaptation, % of test to use unlabeled for training.")


    args = parser.parse_args()
    # deterministic(args.train_seed)

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
    clients_dls = {'train':[], 'test':[]}
    server_dls = {'train':[], 'test':[], 'val': []}
    clients = []
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
            uda, in_ = misc.split_dataset(env,
                int(len(in_)*args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))
            out, in_ = misc.split_dataset(in_,
            int(len(in_)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))
            # valid, in_ = misc.split_dataset(env,
            # int(len(in_)*args.uda_holdout_fraction),
            # misc.seed_hash(args.trial_seed, env_i))
            args.n_target_samples = len(uda)
            print(f"number of target samples: {len(uda)}")
            server_dls['train'].append(torch.utils.data.DataLoader(
            uda,
            num_workers=dataset.N_WORKERS,
            batch_size=args.target_batch_size))
            # use the validation set the same size as the training set
            # server_dls['val'].append(torch.utils.data.DataLoader(
            # valid,
            # num_workers=dataset.N_WORKERS,
            # batch_size=args.target_batch_size))
            server_dls['test'].append(torch.utils.data.DataLoader(
            out,
            num_workers=dataset.N_WORKERS,
            batch_size=64))
        else:
            clients.append(dataset.ENVIRONMENTS[env_i])
            clients_dls['train'].append(torch.utils.data.DataLoader(
            in_,
            num_workers=dataset.N_WORKERS,
            batch_size=hparams['batch_size']))

            clients_dls['test'].append(torch.utils.data.DataLoader(
            out,
            num_workers=dataset.N_WORKERS,
            batch_size=64))
    
    exp_dir = os.path.join('experiments', args.exp_dir, args.dataset, server)
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, f'args_{args.iter_idx}.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    print(f'target:{server}, sources:{clients}')
    num_clients = len(clients)
    dict_client = dict()
    for i in range(num_clients):
        dict_client.update({clients[i]: []})
    clients_size = [len(clients_dls['train'][i])*hparams['batch_size'] for i in range(num_clients)]
    clients_size_frac = np.array(clients_size) / sum(clients_size)
    print(clients_size, clients_size_frac)


    # intialize models
    global_model = domainbedNet(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), hparams)
    
    # print(dataset.input_shape)
    # print(global_model)
    # weiotu
    global_model.to(device)
    global_model_dict = global_model.state_dict()

    local_models = [domainbedNet(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), hparams) for _ in range(num_clients)]

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
    server_results['beta'] = copy.deepcopy(dict_client)
    server_results['train']['loss'] = []
    server_results['train']['acc'] = []
    server_results['train']['auc'] = []
    server_results['test']['loss'] = []
    server_results['test']['acc'] = []
    server_results['test']['auc'] = []
    server_results['best_val_test'] = dict()
    server_results['best_val_test']['loss'] = None
    server_results['best_val_test']['acc'] = None
    
    metric_results = dict()
    metric_results['target_var'] = []
    metric_results['source_target_var'] = copy.deepcopy(dict_client)
    metric_results['projected_norm'] = copy.deepcopy(dict_client)
    metric_results['delta'] = copy.deepcopy(dict_client)
    
    patience = 5
    best_val_loss = np.inf
    best_model_weights = None
    epochs_no_improve = 0

    # do fedavg for 2 epochs, to have a good initialization
    if args.load_trained_model:
        global_model.load_state_dict(torch.load(args.model_path))
    elif args.proj_w > 0:
        if args.dataset == 'TerraIncognita':
            num_init = 10
        else:
            num_init = 2
        for _ in range(num_init):
            for idx in range(num_clients):
                local_models[idx].load_state_dict(global_model_dict)
                local_models[idx], _, (loss, acc, auc) = train(args, hparams, 'source', copy.deepcopy(local_models[idx]), criterion, clients_dls['train'][idx])
                gc.collect()
                torch.cuda.empty_cache()
            # print(local_models)
            global_model_dict = average_weights([model.state_dict() for model in local_models], clients_size_frac)
            global_model.load_state_dict(global_model_dict)

    # beta_start = torch.tensor(0.5)
    # beta_start.to(device)
    # gamma = 0.5
    # beta_GP_0 = [beta_start for i in range(num_clients)]
    hparams["lr"] = hparams["lr"] * 0.25
    for i in range(args.num_global_epochs):
        # training local models
        source_grads = []
        target_grads = None
        if args.proj_w > 0:
            for idx in range(num_clients):
                local_models[idx].load_state_dict(global_model_dict)
                local_models[idx], source_grad, (loss, acc, auc) = train(args, hparams, 'source', copy.deepcopy(local_models[idx]), criterion, clients_dls['train'][idx])
                clients_results['train']['loss'][clients[idx]].append(loss)
                clients_results['train']['acc'][clients[idx]].append(acc)
                clients_results['train']['auc'][clients[idx]].append(auc)
                source_grads.append(source_grad)
                gc.collect()
                torch.cuda.empty_cache()
        
        # averaging the weights
        if args.use_sim:
            new_model, target_grads, (loss, acc, auc) = train(args, hparams, 'target', copy.deepcopy(global_model), criterion, server_dls['train'][0])
            server_results['train']['loss'].append(loss)
            server_results['train']['acc'].append(acc)
            server_results['train']['auc'].append(auc)
            if args.proj_w > 0:
                metrics = empirical_metrics_batch(args.target_batch_size, source_grads, target_grads)
                if args.log_metric:
                    metric_results['target_var'].append(metrics.target_var.item())
                    for i in range(num_clients):
                        metric_results['source_target_var'][clients[i]].append(metrics.source_target_var[i].item())
                        metric_results['projected_norm'][clients[i]].append(metrics.projected_grads_norm_square[i].item())
                        metric_results['delta'][clients[i]].append(metrics.deltas[i])
                beta_GP_1 = metrics.return_fedgp_beta()
                # if i == 0:
                #     beta_GP_0 = copy.deepcopy(beta_GP_1)
                # beta_GP_1 = [beta_GP_0[i] * gamma + beta_GP_1[i] * (1-gamma)  for i in range(num_clients)]
                # print(beta_GP_1)
                for idx, beta in enumerate(beta_GP_1):
                    server_results['beta'][clients[idx]].append(beta.item())
                global_model_dict = update_global(args, hparams, [model.state_dict() for model in local_models], global_model.state_dict(), new_model.state_dict(), clients_size, clients_size_frac, i, beta_GP_1)
                global_model.load_state_dict(global_model_dict)
                # beta_GP_0 = copy.deepcopy(beta_GP_1)
            else:
                global_model = copy.deepcopy(new_model)
        elif args.convex_agg:
            new_model, target_grads, (loss, acc, auc) = train(args, hparams, 'target', copy.deepcopy(global_model), criterion, server_dls['train'][0])
            server_results['train']['loss'].append(loss)
            server_results['train']['acc'].append(acc)
            server_results['train']['auc'].append(auc)
            if args.proj_w > 0:
                metrics = empirical_metrics_batch(args.target_batch_size, source_grads, target_grads)
                if args.log_metric:
                    metric_results['target_var'].append(metrics.target_var.item())
                    for i in range(num_clients):
                        metric_results['source_target_var'][clients[i]].append(metrics.source_target_var[i].item())
                beta_DA = metrics.return_fedda_beta()
                # print(beta_DA)
                for idx, beta in enumerate(beta_DA):
                    server_results['beta'][clients[idx]].append(beta.item())
                global_model_dict = update_global_convex(args, [model.state_dict() for model in local_models], global_model.state_dict(), new_model.state_dict(), clients_size, clients_size_frac, i, beta_DA)
                global_model.load_state_dict(global_model_dict)
            else:
                global_model = copy.deepcopy(new_model)
        else:
            global_model_dict = average_weights([model.state_dict() for model in local_models], clients_size_frac)
            global_model.load_state_dict(global_model_dict)
            if args.finetune:
                # Freeze all but last layer
                # for name, param in global_model.named_parameters():
                    # if not 'linear' in name:
                    #     param.requires_grad = False
                global_model, (loss, acc, auc) = train(args, hparams, 'target', global_model, criterion, server_dls['train'][0])
                server_results['train']['loss'].append(loss)
                server_results['train']['acc'].append(acc)
                server_results['train']['auc'].append(auc)
                # unfreeze all
                # for name, param in global_model.named_parameters():
                #     param.requires_grad = True
                global_model_dict = global_model.state_dict()

        print('testing global model on its target domain')
        (loss, acc, auc) = test(args, global_model, criterion, server_dls['test'][0])
        server_results['test']['loss'].append(loss)
        server_results['test']['acc'].append(acc)
        server_results['test']['auc'].append(auc)
        
        # test on the validation set
        if args.early_stop:
            (val_loss, val_acc, _) = test(args, global_model, criterion, server_dls['val'][0])
            # print(val_loss, val_acc)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
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

    # with open(os.path.join(exp_dir,(f'clients_results_{args.iter_idx}.json')), 'w') as fp:
    #         json.dump(clients_results, fp, indent=4)
    # fp.close()
    
    with open(os.path.join(exp_dir,(f'server_results_{args.iter_idx}.json')), 'w') as fp:
            json.dump(server_results, fp, indent=4)
    fp.close()
    
    if args.log_metric:
        with open(os.path.join(exp_dir,(f'metric_results_{args.iter_idx}.json')), 'w') as fp:
                json.dump(metric_results, fp, indent=4)
        fp.close()

    # torch.save(best_model_weights, os.path.join(exp_dir,f'server_checkpoint_{args.iter_idx}.pt'))

    # for idx in local_models:
    #     torch.save(local_models[idx].state_dict(),os.path.join(exp_dir,f'client_{idx}_checkpoint_{args.iter_idx}.pt'))
from models import NN, Metrics_n_Datasets
import torch
import numpy as np
from empirical_metrics import empirical_metrics

# settings
n = 50  # dim of x
d = 10  # dim of y
m = 100     # num of feature vectors
N = 5000   # num of samples
num_centers = 10
num_epoch = 300
beta = 0.5
seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('running on ' + str(device))


# load source data
data_file_handle = 'rbf-n-' + str(n) + '-d-' + str(d) + '-m-' + str(m) + '-N-' + str(N) + '-num_centers-' + str(
        num_centers) + 'source_target_diff-' + str(0.0)
data_file = 'datasets/' + data_file_handle + '.npz'
data = np.load(data_file)
source_X = torch.Tensor(data['X']).to(device)
source_Y = torch.Tensor(data['Y']).to(device)

target_sample_ratio_list = np.exp(np.arange(np.log(0.003), np.log(0.5),  (np.log(0.4) - np.log(0.001)) / 10))
# train models
for source_target_dist in list(np.arange(0., 1., 0.07)):
    for target_sample_ratio in list(target_sample_ratio_list):

        # train models
        data_file_handle = 'rbf-n-' + str(n) + '-d-' + str(d) + '-m-' + str(m) + '-N-' + str(N) + '-num_centers-' + str(
            num_centers) + 'source_target_diff-' + str(source_target_dist)
        data_file = 'datasets/' + data_file_handle + '.npz'
        data = np.load(data_file)
        target_X_all = torch.Tensor(data['X']).to(device)
        target_Y_all = torch.Tensor(data['Y']).to(device)
        dataset = Metrics_n_Datasets((source_X, source_Y), (target_X_all, target_Y_all), target_sample_ratio, None)

        initialization_model_file = './models/temp_model'
        model_gp = NN(n, d, m, device)
        model_gp.save_model(initialization_model_file)
        # ensure the same initialization
        model_convex = NN(n, d, m, device)
        model_convex.load_model(initialization_model_file, device)
        model_target_only = NN(n, d, m, device)
        model_target_only.load_model(initialization_model_file, device)
        model_source_only = NN(n, d, m, device)
        model_source_only.load_model(initialization_model_file, device)

        dataset.list_of_model = [model_convex]  # compute at initialization, all models are the same
        target_var = dataset.compute_target_var() # accurate estimation using all target data
        source_target_var = dataset.compute_source_target_diff() # accurate
        print('target var is {}; source-target var is {}'.format(target_var, source_target_var))

        # now use only training data to do estimation at initialization (all models are the same)
        emp_metrics = empirical_metrics(model_convex, dataset, num_trials=20)
        emp_metrics.compute_quantities()
        # compute estimated error terms
        estimated_error_gp = emp_metrics.compute_error_gradient_proj(beta=beta)
        print('estimated error for gradient proj is {}'.format(estimated_error_gp))
        estimated_error_convex = emp_metrics.compute_error_convex_combine(beta=beta)
        print('estimated error for convex combine is {}'.format(estimated_error_convex))
        estimated_error_target_only = emp_metrics.compute_error_convex_combine(beta=0)
        print('estimated error for target only is {}'.format(estimated_error_target_only))
        estimated_error_source_only = emp_metrics.compute_error_convex_combine(beta=1)
        print('estimated error for source only is {}'.format(estimated_error_source_only))


        print('training GP with source-target diff {} and target sample ratio {}'.format(source_target_dist, target_sample_ratio))
        test_loss_gp = model_gp.train(dataset, num_epoch, 'gradient_proj', beta)
        print('training Convex Combine with source-target diff {} and target sample ratio {}'.format(source_target_dist,
                                                                                         target_sample_ratio))
        test_loss_convex = model_convex.train(dataset, num_epoch, 'convex_combine', beta)
        print('training Target Only with source-target diff {} and target sample ratio {}'.format(source_target_dist,
                                                                                                     target_sample_ratio))
        test_loss_target_only = model_target_only.train(dataset, num_epoch, 'convex_combine', beta=0)
        print('training Source Only with source-target diff {} and target sample ratio {}'.format(source_target_dist,
                                                                                                     target_sample_ratio))
        test_loss_source_only = model_source_only.train(dataset, num_epoch, 'convex_combine', beta=1)

        result_dict = {'test_loss_gp': test_loss_gp, 'test_loss_convex': test_loss_convex, 'test_loss_target_only':
            test_loss_target_only, 'test_loss_source_only': test_loss_source_only, 'source_target_var': source_target_var,
                       "target_var": target_var, 'estimated_error_gp': estimated_error_gp,
                       'estimated_error_convex': estimated_error_convex, 'estimated_error_target_only': estimated_error_target_only,
                       'estimated_error_source_only': estimated_error_source_only}

        result_file_handle = 'test_loss-rbf-n-' + str(n) + '-d-' + str(d) + '-m-' + str(m) + '-N-' + str(N) + '-num_centers-' + str(
            num_centers) + '-source_target_diff-' + str(source_target_dist) + '-target-sample-ratio-' + str(target_sample_ratio)
        data_file = './results/' + result_file_handle
        torch.save(result_dict, data_file)
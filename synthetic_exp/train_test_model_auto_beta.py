from models import NN, Metrics_n_Datasets
import torch
import numpy as np
from empirical_metrics_batch import empirical_metrics_batch

# settings
n = 50  # dim of x
d = 10  # dim of y
m = 100     # num of feature vectors
N = 5000   # num of samples
dim = m * (n+d)
num_centers = 10
num_epoch = 300
initialization_num_epoch = 50
beta = 0.5
auto_beta = True
is_estimation = True # choose to use estimated quantities to compute beta
seed = 1
batch_size = 2
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

target_sample_ratio_list = np.array(list(np.exp(np.arange(np.log(0.003), np.log(0.5),  (np.log(0.4) - np.log(0.001)) / 10))))
source_target_dist_list = np.array(list(np.arange(0., 1., 0.07)))
# train models
for source_target_dist in source_target_dist_list:
    for target_sample_ratio in target_sample_ratio_list:

        # train models
        data_file_handle = 'rbf-n-' + str(n) + '-d-' + str(d) + '-m-' + str(m) + '-N-' + str(N) + '-num_centers-' + str(
            num_centers) + 'source_target_diff-' + str(source_target_dist)
        data_file = 'datasets/' + data_file_handle + '.npz'
        data = np.load(data_file)
        target_X_all = torch.Tensor(data['X']).to(device)
        target_Y_all = torch.Tensor(data['Y']).to(device)
        dataset = Metrics_n_Datasets((source_X, source_Y), (target_X_all, target_Y_all), target_sample_ratio, None)

        initialization_model_file = './models/temp_model'
        model_initial = NN(n, d, m, device)
        # model_initial.train(dataset, initialization_num_epoch, 'initialization_train')
        model_initial.save_model(initialization_model_file)
        # ensure the same initialization
        model_gp = NN(n, d, m, device)
        # model_gp.save_model(initialization_model_file)
        model_gp.load_model(initialization_model_file, device)
        model_convex = NN(n, d, m, device)
        model_convex.load_model(initialization_model_file, device)
        model_target_only = NN(n, d, m, device)
        model_target_only.load_model(initialization_model_file, device)
        model_source_only = NN(n, d, m, device)
        model_source_only.load_model(initialization_model_file, device)

        dataset.list_of_model = [model_convex]  # compute at initialization, all models are the same
        true_target_var = dataset.compute_target_var(normalized=True) # accurate estimation using all target data
        true_source_target_var = dataset.compute_source_target_var(normalized=True) # accurate and
        ##### Note: the square is already applied
        true_tau = dataset.compute_tau() # accurate
        true_projected_grad_norm_square = dataset.compute_projected_grads_norm_square(normalized=True)

        # now estimate the metrics using only training data at initialization (all models are the same)
        emp_metrics = empirical_metrics_batch(model_convex, target_batch_size=batch_size)
        emp_metrics.get_grads_subsample(dataset)
        emp_metrics.compute_quantities()
        estimated_target_var = emp_metrics.target_var # estimation using only training data
        estimated_source_target_var = emp_metrics.source_target_var # estimation using only training data
        estimated_tau = emp_metrics.tau
        estimated_delta = emp_metrics.delta
        estimated_target_norm_square = emp_metrics.target_norm_square
        estimated_projected_grad_norm_square = emp_metrics.projected_grads_norm_square
        print(' ')
        print('true target var: {}; its estimation: {}'.format(true_target_var, estimated_target_var))
        print('true source-target var: {}; its estimation: {}'.format(true_source_target_var, estimated_source_target_var))
        print('true tau is {}; its estimation: {}'.format(true_tau, estimated_tau))
        print('estimated delta: {}'.format(estimated_delta))
        print('true projected_grad_norm_square: {}; its estimation: {}'.format(true_projected_grad_norm_square, estimated_projected_grad_norm_square))
        # compute estimated error terms


        # best_beta = target_var / (target_var + source_target_var)

        if is_estimation:
            target_var = estimated_target_var
            source_target_var = estimated_source_target_var
            tau = estimated_tau
            projected_grad_norm_square = estimated_projected_grad_norm_square
        else:
            target_var = true_target_var
            source_target_var = true_source_target_var
            tau = true_tau
            projected_grad_norm_square = true_projected_grad_norm_square 
        # choice of beta for GP and DA
        if auto_beta:
            # beta_GP = target_var / (target_var + tau ** 2 * source_target_var) # not using delta
            #beta_GP = target_var / (target_var + estimated_delta * tau ** 2 * source_target_var + (1-estimated_delta) * estimated_target_norm_square) # using estimated delta
            beta_GP = target_var / (target_var + projected_grad_norm_square)
            # beta_GP = target_var / (target_var + true_projected_grad_norm_square)
            beta_DA = target_var / (target_var + source_target_var)
        else:
            beta_GP = beta
            beta_DA = beta

        print('beta_GP = {}; beta_DA={}'.format(beta_GP, beta_DA))
        true_delta_error_gp = dataset.compute_delta_error_gp(beta_GP, normalized=True)
        approximated_delta_error_gp = ((1-beta_GP)**2 + (2*beta_GP-beta_GP ** 2)/dim) * target_var + beta_GP**2 * true_projected_grad_norm_square
        print('true delta error gp is {}, its approximation is {}'.format(true_delta_error_gp, approximated_delta_error_gp))




        print('training GP with source-target diff {} and target sample ratio {}'.format(source_target_dist, target_sample_ratio))
        test_loss_gp = model_gp.train(dataset, num_epoch, 'gradient_proj', beta_GP)
        print('training Convex Combine with source-target diff {} and target sample ratio {}'.format(source_target_dist,
                                                                                         target_sample_ratio))
        test_loss_convex = model_convex.train(dataset, num_epoch, 'convex_combine', beta_DA)
        print('training Target Only with source-target diff {} and target sample ratio {}'.format(source_target_dist,
                                                                                                     target_sample_ratio))
        test_loss_target_only = model_target_only.train(dataset, num_epoch, 'convex_combine', beta=0)
        print('training Source Only with source-target diff {} and target sample ratio {}'.format(source_target_dist,
                                                                                                     target_sample_ratio))
        test_loss_source_only = model_source_only.train(dataset, num_epoch, 'convex_combine', beta=1)

        # result_dict = {'test_loss_gp': test_loss_gp, 'test_loss_convex': test_loss_convex, 'test_loss_target_only':
        #     test_loss_target_only, 'test_loss_source_only': test_loss_source_only, 'source_target_var': source_target_var,
        #                "target_var": target_var, 'estimated_error_gp': estimated_error_gp,
        #                'estimated_error_convex': estimated_error_convex, 'estimated_error_target_only': estimated_error_target_only,
        #                'estimated_error_source_only': estimated_error_source_only}
        result_dict = {'test_loss_gp': test_loss_gp, 'test_loss_convex': test_loss_convex, 'test_loss_target_only':
            test_loss_target_only, 'test_loss_source_only': test_loss_source_only, 'source_target_var': true_source_target_var,
                       "target_var": true_target_var, 'estimated_source_target_var': estimated_source_target_var,
                       "estimated_target_var": estimated_target_var, "true_projected_grad_norm_square": true_projected_grad_norm_square,
                       "estimated_projected_grad_norm_square": estimated_projected_grad_norm_square, "true_delta_error_gp": true_delta_error_gp,
                       "beta_GP": beta_GP, "beta_DA": beta_DA}

        result_file_handle = 'test_loss-rbf-n-' + str(n) + '-d-' + str(d) + '-m-' + str(m) + '-N-' + str(N) + '-num_centers-' + str(
            num_centers) + '-seed-' + str(seed) + '-source_target_diff-' + str(source_target_dist) + '-target-sample-ratio-' + str(target_sample_ratio) + '-auto_beta=' + str(auto_beta) + '-is_estimateion=' + str(is_estimation)
        data_file = './results/' + result_file_handle
        torch.save(result_dict, data_file)
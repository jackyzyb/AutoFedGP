import matplotlib.pyplot as plt
import matplotlib.collections as mcol
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
import numpy as np
import torch
from matplotlib.pyplot import figure


# settings
n = 50  # dim of x
d = 10  # dim of y
m = 100     # num of feature vectors
N = 5000   # num of samples
dim = m * (d + n)   # mode size
num_centers = 10
num_epoch = 300
beta = 0.5
# seed = 1
seeds = [1]

auto_beta = False
is_estimation = False
is_prediction = False

gp_list = [[], []]
convex_list = [[], []]
target_only_list =[[], []]
source_only_list = [[], []]
lists = [gp_list, convex_list, target_only_list, source_only_list]
max_target_var = 0
max_source_target_var = 0

target_sample_ratio_list = np.array(list(np.exp(np.arange(np.log(0.003), np.log(0.5),  (np.log(0.4) - np.log(0.001)) / 10))))
source_target_dist_list = np.array(list(np.arange(0., 1., 0.07)))
# train models
for source_target_dist in source_target_dist_list:
    for target_sample_ratio in target_sample_ratio_list:
        # result_dict = {'test_loss_gp': test_loss_gp, 'test_loss_convex': test_loss_convex, 'test_loss_target_only':
        #     test_loss_target_only, 'test_loss_source_only': test_loss_source_only,
        #                'source_target_var': source_target_var,
        #                "target_var": target_var}
        test_loss_gp_list = []
        test_loss_convex_list = []
        test_loss_target_only_list = []
        test_loss_source_only_list = []
        source_target_var_list = []
        target_var_list = []
        true_projected_grad_norm_square_list =[]
        true_delta_error_gp_list = []
        beta_GP_list = []
        beta_DA_list = []
        delta_error_gp_list = []
        delta_error_convex_list = []
        delta_error_target_only_list = []
        delta_error_source_only_list = []

        for seed in seeds:
            result_file_handle = 'test_loss-rbf-n-' + str(n) + '-d-' + str(d) + '-m-' + str(m) + '-N-' + str(
                N) + '-num_centers-' + str(
                num_centers) + '-seed-' + str(seed) + '-source_target_diff-' + str(source_target_dist) + '-target-sample-ratio-' + str(
                target_sample_ratio) + '-auto_beta=' + str(auto_beta) + '-is_estimateion=' + str(is_estimation)

            data_file = './results/' + result_file_handle
            result_dict = torch.load(data_file)
            test_loss_gp = result_dict['test_loss_gp']
            test_loss_convex = result_dict['test_loss_convex']
            test_loss_target_only = result_dict['test_loss_target_only']
            test_loss_source_only = result_dict['test_loss_source_only']
            source_target_var = result_dict['source_target_var']
            target_var = result_dict['target_var']
            true_projected_grad_norm_square = result_dict["true_projected_grad_norm_square"]
            true_delta_error_gp = result_dict["true_delta_error_gp"]
            beta_GP = result_dict["beta_GP"]
            beta_DA = result_dict["beta_DA"]

            test_loss_gp_list.append(test_loss_gp)
            test_loss_convex_list.append(test_loss_convex)
            test_loss_target_only_list.append(test_loss_target_only)
            test_loss_source_only_list.append(test_loss_source_only)
            source_target_var_list.append(source_target_var)
            target_var_list.append(target_var)
            true_projected_grad_norm_square_list.append(true_projected_grad_norm_square)
            true_delta_error_gp_list.append(true_delta_error_gp)
            beta_GP_list.append(beta_GP)
            beta_DA_list.append(beta_DA)

            delta_error_gp = true_delta_error_gp
            # delta_error_gp = ((1-beta_GP)**2 + (2*beta_GP-beta_GP ** 2)/m) * target_var + beta_GP**2 * true_projected_grad_norm_square
            delta_error_convex = (1 - beta_GP) ** 2 * target_var + beta_GP ** 2 * source_target_var
            delta_error_target_only = target_var
            delta_error_source_only = source_target_var

            delta_error_gp_list.append(delta_error_gp)
            delta_error_convex_list.append(delta_error_convex)
            delta_error_target_only_list.append(delta_error_target_only)
            delta_error_source_only_list.append(delta_error_source_only)

        if target_var > max_target_var:
            max_target_var = target_var
        if source_target_var > max_source_target_var:
            max_source_target_var = source_target_var

        test_loss_gp = torch.mean(torch.tensor(test_loss_gp_list))
        test_loss_convex = torch.mean(torch.tensor(test_loss_convex_list))
        test_loss_target_only = torch.mean(torch.tensor(test_loss_target_only_list))
        test_loss_source_only = torch.mean(torch.tensor(test_loss_source_only_list))
        losses = torch.Tensor([test_loss_gp, test_loss_convex, test_loss_target_only, test_loss_source_only])

        delta_error_gp = torch.mean(torch.tensor(delta_error_gp_list))
        delta_error_convex = torch.mean(torch.tensor(delta_error_convex_list))
        delta_error_target_only = torch.mean(torch.tensor(delta_error_target_only_list))
        delta_error_source_only = torch.mean(torch.tensor(delta_error_source_only_list))
        delta_errors = torch.Tensor([delta_error_gp, delta_error_convex, delta_error_target_only, delta_error_source_only])

        if is_prediction:
            best_method = delta_errors.argmin()
        else:
            best_method = losses.argmin()

        lists[best_method][0].append(source_target_var)
        lists[best_method][1].append(target_var)

figure(figsize=(8, 6), dpi=80)

# lists = [gp_list, convex_list, target_only_list, source_only_list]
if auto_beta:
    plt.plot(gp_list[0], gp_list[1], 'bo', label='FedGP (auto)', markersize=11)
    plt.plot(convex_list[0], convex_list[1], 'rX', label='FedDA (auto)', markersize=11)
else:
    plt.plot(gp_list[0], gp_list[1], 'bo', label='FedGP', markersize=11)
    plt.plot(convex_list[0], convex_list[1], 'rX', label='FedDA', markersize=11)
plt.plot(target_only_list[0], target_only_list[1], 'g^', label='Target Only', markersize=10)
plt.plot(source_only_list[0], source_only_list[1], 'yv', label='Source Only', markersize=10)
plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.15)
#
axis_lim = max(max_source_target_var, max_target_var)
plt.xlim([-axis_lim/20, axis_lim * 1.1])
plt.ylim([-axis_lim/20, axis_lim * 1.1])

# plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=15)
plt.legend(loc='upper left', fontsize=20)


if is_prediction:
    plt.title('Method with the Smallest Delta Error', fontsize=27)
else:
    plt.title('Method with the Best Test Result', fontsize=27)
plt.ylabel('Target Domain Variance', fontsize=27)
plt.xlabel('Source-Target Domain Distance', fontsize=27)
plt.yticks(fontsize=25)
plt.xticks(fontsize=25)
plt.savefig('test_error.pdf')
plt.show()
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
num_centers = 10
num_epoch = 300
beta = 0.5

auto_beta = True
is_estimation = True

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

        result_file_handle = 'test_loss-rbf-n-' + str(n) + '-d-' + str(d) + '-m-' + str(m) + '-N-' + str(
            N) + '-num_centers-' + str(
            num_centers) + '-source_target_diff-' + str(source_target_dist) + '-target-sample-ratio-' + str(
            target_sample_ratio) + '-auto_beta=' + str(auto_beta) + '-is_estimateion=' + str(is_estimation)

        data_file = './results/' + result_file_handle
        result_dict = torch.load(data_file)
        test_loss_gp = result_dict['test_loss_gp']
        test_loss_convex = result_dict['test_loss_convex']
        test_loss_target_only = result_dict['test_loss_target_only']
        test_loss_source_only = result_dict['test_loss_source_only']
        source_target_var = result_dict['source_target_var']
        target_var = result_dict['target_var']


        if target_var > max_target_var:
            max_target_var = target_var
        if source_target_var > max_source_target_var:
            max_source_target_var = source_target_var

        losses = torch.Tensor([test_loss_gp, test_loss_convex, test_loss_target_only, test_loss_source_only])
        best_method = losses.argmin()
        lists[best_method][0].append(source_target_var)
        lists[best_method][1].append(target_var)

figure(figsize=(8, 6), dpi=80)

# lists = [gp_list, convex_list, target_only_list, source_only_list]
plt.plot(gp_list[0], gp_list[1], 'bo', label='FedGP')
plt.plot(convex_list[0], convex_list[1], 'rX', label='FedDA')
plt.plot(target_only_list[0], target_only_list[1], 'g^', label='Target Only')
plt.plot(source_only_list[0], source_only_list[1], 'yv', label='Source Only')
#
axis_lim = max(max_source_target_var, max_target_var)
plt.xlim([-axis_lim/20, axis_lim * 1.1])
plt.ylim([-axis_lim/20, axis_lim * 1.1])

plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=15)

plt.title('Actual Best Test Results', fontsize=20)
plt.ylabel('Target Domain Variance', fontsize=20)
plt.xlabel('Source-Target Domain Distance', fontsize=20)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.savefig('test_error.pdf')
plt.show()
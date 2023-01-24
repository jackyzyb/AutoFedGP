import matplotlib.pyplot as plt
import matplotlib.collections as mcol
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
import numpy as np
import torch


# settings
n = 50  # dim of x
d = 10  # dim of y
m = 100     # num of feature vectors
N = 5000   # num of samples
num_centers = 10
num_epoch = 300
beta = 0.5
seed = 233

is_prediction = False

gp_list = [[], []]
convex_list = [[], []]
target_only_list =[[], []]
source_only_list = [[], []]
lists = [gp_list, convex_list, target_only_list, source_only_list]
max_target_var = 0
max_source_target_var = 0

target_sample_ratio_list = np.exp(np.arange(np.log(0.003), np.log(0.5),  (np.log(0.4) - np.log(0.001)) / 10))
for source_target_diff in list(np.arange(0., 1., 0.07)):
    for target_sample_ratio in list(target_sample_ratio_list):
        # result_dict = {'test_loss_gp': test_loss_gp, 'test_loss_convex': test_loss_convex, 'test_loss_target_only':
        #     test_loss_target_only, 'test_loss_source_only': test_loss_source_only,
        #                'source_target_var': source_target_var,
        #                "target_var": target_var}

        result_file_handle = 'test_loss-rbf-n-' + str(n) + '-d-' + str(d) + '-m-' + str(m) + '-N-' + str(N)\
                             + '-num_centers-' + str(num_centers) + '-source_target_diff-' + str(source_target_diff) + '-target-sample-ratio-' + str(
            target_sample_ratio)
        data_file = './results/' + result_file_handle
        result_dict = torch.load(data_file)
        test_loss_gp = result_dict['test_loss_gp']
        test_loss_convex = result_dict['test_loss_convex']
        test_loss_target_only = result_dict['test_loss_target_only']
        test_loss_source_only = result_dict['test_loss_source_only']
        source_target_var = result_dict['source_target_var']
        target_var = result_dict['target_var']

        estimated_error_gp = result_dict['estimated_error_gp']
        estimated_error_convex = result_dict['estimated_error_convex']
        estimated_error_target_only = result_dict['estimated_error_target_only']
        estimated_error_source_only = result_dict['estimated_error_source_only']

        if target_var > max_target_var:
            max_target_var = target_var
        if source_target_var > max_source_target_var:
            max_source_target_var = source_target_var
        if is_prediction:
            losses = torch.Tensor([estimated_error_gp, estimated_error_convex, estimated_error_target_only, estimated_error_source_only])
        else:
            losses = torch.Tensor([test_loss_gp, test_loss_convex, test_loss_target_only, test_loss_source_only])
        best_method = losses.argmin()
        lists[best_method][0].append(source_target_var)
        lists[best_method][1].append(target_var)

# lists = [gp_list, convex_list, target_only_list, source_only_list]
plt.plot(gp_list[0], gp_list[1], 'bo', label='Gradient Projection')
plt.plot(convex_list[0], convex_list[1], 'ro', label='Convex Combination')
plt.plot(target_only_list[0], target_only_list[1], 'go', label='Target Only')
plt.plot(source_only_list[0], source_only_list[1], 'yo', label='Source Only')

axis_lim = max(max_source_target_var, max_target_var)
plt.xlim([-axis_lim/20, axis_lim * 1.1])
plt.ylim([-axis_lim/20, axis_lim * 1.1])

plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
if is_prediction:
    plt.title('Predicted Best Methods')
else:
    plt.title('Actual Best Methods')
plt.ylabel('Target Domain Variance')
plt.xlabel('Source-Target Domain Difference')
plt.show()
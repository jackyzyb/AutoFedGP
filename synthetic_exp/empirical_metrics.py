import torch, random
from models import NN, Metrics_n_Datasets

class empirical_metrics:
    def __init__(self, model:NN, dataset:Metrics_n_Datasets, num_trials=20):
        self.model = model
        self.dataset = dataset
        self.num_trials = num_trials
        self.target_var = None
        self.source_target_var = None
        self.projected_target_var = None
        self.target_proj_source_var = None

    def compute_quantities(self, subsample_ratio=0.5):
        # compute source target difference
        grads_S = self.model.get_gradients(self.dataset.source_X, self.dataset.source_Y) # using all source data
        grads_T = self.model.get_gradients(self.dataset.target_X, self.dataset.target_Y) # using all target training data
        self.source_target_var = torch.norm(grads_T - grads_S)**2
        normalized_grad_S = grads_S / torch.norm(grads_S)
        self.target_proj_source_var = torch.norm((grads_T * normalized_grad_S).sum() * normalized_grad_S - grads_T)**2

        # compute target variance
        self.target_var = 0
        self.projected_target_var = 0
        grad_proj_filter_avg = 0
        for trial in range(self.num_trials):
            grads_T_subsample = self._compute_subsample_target_grads(subsample_ratio)
            if (grads_T_subsample * grads_S).sum() > 0:
                grad_proj_filter = 1
            else:
                grad_proj_filter = 0
            self.target_var += torch.norm(grads_T - grads_T_subsample)**2
            self.projected_target_var += torch.sum((grads_T - grads_T_subsample) * normalized_grad_S)**2 * grad_proj_filter
            grad_proj_filter_avg += grad_proj_filter
        self.target_var = self.target_var * subsample_ratio / (1-subsample_ratio) / self.num_trials
        self.projected_target_var = self.projected_target_var * subsample_ratio / (1-subsample_ratio) / self.num_trials
        grad_proj_filter_avg = grad_proj_filter_avg / self.num_trials
        self.target_proj_source_var = self.target_proj_source_var * grad_proj_filter_avg

    def _compute_subsample_target_grads(self, subsample_ratio):
        # subsample data
        N = self.dataset.target_X.shape[1] # number of target training data
        N_target_sub_samples = int(N * subsample_ratio) # use half of the target training data
        data_split_list = list(range(N))
        random.shuffle(data_split_list)
        data_split_list = data_split_list[:N_target_sub_samples]
        X_target = self.dataset.target_X[:, data_split_list]
        Y_target = self.dataset.target_Y[:, data_split_list]
        grads_T_subsample = self.model.get_gradients(X_target, Y_target)
        return grads_T_subsample

    def compute_error_convex_combine(self, beta):
        return  (1 - beta)**2 * self.target_var + beta**2 * (self.source_target_var - self.target_var)

    def compute_error_gradient_proj(self, beta):
        return (2*beta-beta**2) * self.projected_target_var + (1-beta)**2 * self.target_var + beta**2 * (self.target_proj_source_var - self.target_var - self.projected_target_var)

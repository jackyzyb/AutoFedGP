import torch, random
from models import NN, Metrics_n_Datasets

class empirical_metrics_batch:
    def __init__(self, model:NN, target_batch_size):
        self.model = model
        self.target_batch_size = target_batch_size # we don't need this for the source client

        self.source_grad = None # the source grad is simply the average of all batches size (M, )
        self.target_grads = None # A tensor of size (N, M) where N is the number of batches and M is the dim
        self.target_grad = None # the average of self.target_grads, size of (M, )

        # call self.compute_quantities() to compute the following quantities after getting the above three quantities
        self.target_var = None
        self.source_target_var = None
        self.tau = None
        self.delta = None
        self.target_norm_square = None

    def get_grads_subsample(self, dataset:Metrics_n_Datasets):
        # directly get those gradients
        num_target_data = dataset.num_target_sample
        num_batches = int(num_target_data / self.target_batch_size)
        self.source_grad = self.model.get_gradients(dataset.source_X, dataset.source_Y)  # using all source data
        self.target_grad = self.model.get_gradients(dataset.target_X, dataset.target_Y)  # using all target training data
        self.target_grads = torch.empty([num_batches, len(self.target_grad)], device=self.target_grad.device)
        for batch in range(num_batches):
            grads_T_subsample = self._compute_subsample_target_grads(dataset)
            self.target_grads[batch, :] = grads_T_subsample

    def compute_quantities(self):
        num_batches, dim = self.target_grads.shape
        # compute target variance
        sample_target_var = torch.sum((self.target_grads - self.target_grad) ** 2) / (num_batches - 1) / dim
        self.target_var = sample_target_var / num_batches
        # compute source target difference
        sample_source_target_var = torch.sum((self.target_grads - self.source_grad) ** 2) / num_batches / dim
        self.source_target_var = max(sample_source_target_var - sample_target_var, 0.)
        
        # compute tau
        eps = 0.0001  # room to numerical error
        diff = self.source_target_var * dim
        if diff < eps:
            tau = 0.
        else:
            tau_square = (torch.mean(torch.norm(self.target_grads, dim=1)) - sample_target_var * dim) / diff
        self.tau = max(tau_square, 0.) ** 0.5
        # diff = torch.norm(self.target_grad - self.source_grad)
        # cos_rho = (self.source_grad * self.target_grad).sum() / torch.norm(self.target_grad) / torch.norm(self.source_grad)
        # sin_rho = (1 - cos_rho ** 2) ** 0.5
        # if diff < eps:
        #     tau = 0
        # else:
        #     tau = (torch.norm(self.target_grad) * sin_rho / diff).item()
        # self.tau = tau

        # compute delta
        inner_products = torch.sum(self.target_grads * self.source_grad, dim=1)
        delta = torch.sum(inner_products > 0) / num_batches
        self.delta = 1 - (1 - delta.item()) / num_batches
        # compute norm of the target gradients
        self.target_norm_square = torch.norm(self.target_grad).item() ** 2 / dim

    def _compute_subsample_target_grads(self, dataset):
        # subsample data
        N = dataset.target_X.shape[1] # number of target training data
        N_target_sub_samples = self.target_batch_size # use half of the target training data
        data_split_list = list(range(N))
        random.shuffle(data_split_list)
        data_split_list = data_split_list[:N_target_sub_samples]
        X_target = dataset.target_X[:, data_split_list]
        Y_target = dataset.target_Y[:, data_split_list]
        grads_T_subsample = self.model.get_gradients(X_target, Y_target)
        return grads_T_subsample

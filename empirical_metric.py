import torch

class empirical_metrics_batch:
    def __init__(self, target_batch_size, source_grads, target_grads):
        self.target_batch_size = target_batch_size # we don't need this for the source client

        self.source_grads = source_grads # the source grad is simply the average of all batches size list of [(M, )]
        self.target_grads = target_grads # A tensor of size (N, M) where N is the number of batches and M is the dim
        self.target_grad = torch.mean(target_grads, dim=0) # the average of self.target_grads, size of (M, )

        # call self.compute_quantities() to compute the following quantities after getting the above three quantities
        self.target_var = None
        self.source_target_var = [] # lenght of number of source clients
        self.taus = [] # length of number of source clients
        self.deltas = []
        
        self.compute_quantities()

    def compute_quantities(self):
        num_batches, dim = self.target_grads.shape
        # compute target variance
        sample_target_var = torch.sum((self.target_grads - self.target_grad) ** 2) / (num_batches - 1) / dim
        self.target_var = sample_target_var / num_batches
        # compute norm of the target gradients
        self.target_norm_square = torch.norm(self.target_grad).item() ** 2 / dim
        # compute source target difference
        for source_grad in self.source_grads:
            sample_source_target_var = torch.sum((self.target_grads - source_grad) ** 2) / num_batches / dim
            self.source_target_var.append(max(sample_source_target_var - sample_target_var, 0.))
            # compute tau
            eps = 0.0001  # room to numerical error
            diff = torch.norm(self.target_grad - source_grad)
            cos_rho = (source_grad * self.target_grad).sum() / torch.norm(self.target_grad) / torch.norm(source_grad)
            sin_rho = (1 - cos_rho ** 2) ** 0.5
            if diff < eps:
                tau = 0
            else:
                tau = (torch.norm(self.target_grad) * sin_rho / diff).item()
                
            # compute delta
            inner_products = torch.sum(self.target_grads * source_grad, dim=1)
            delta = torch.sum(inner_products > 0) / num_batches
            self.deltas.append(1 - (1 - delta.item()) / num_batches)
            self.taus.append(tau)
    
    def return_fedda_beta(self):
        return [self.target_var / (self.target_var + s_t_var) for s_t_var in self.source_target_var]
    
    def return_fedgp_with_thresh_beta(self):
        return [self.target_var / (self.target_var + self.deltas[idx] * self.taus[idx] ** 2 * self.source_target_var[idx] + (1-self.deltas[idx]) * self.target_norm_square) for idx in range(len(self.source_grads))]
    
    def return_fedgp_beta(self):
        return [self.target_var / (self.target_var + self.taus[idx] ** 2 * self.source_target_var[idx]) for idx in range(len(self.source_grads))]
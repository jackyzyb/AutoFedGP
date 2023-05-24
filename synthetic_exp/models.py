import torch
from utils import convex_combine, gradient_proj, MSE
import random

def sigmoid_derivative(X: torch.Tensor):
    E = torch.exp(-X)
    return E / (1 + E).pow(2)


class Metrics_n_Datasets:
    def __init__(self, source_data, target_data_all, target_sample_ratio, list_of_model, num_of_trails=40):
        # data are of shape [data_dim, num_data]
        self.source_X, self.source_Y = source_data
        self.target_X_all, self.target_Y_all = target_data_all
        self.target_sample_ratio = target_sample_ratio
        self.list_of_model = list_of_model
        self.num_of_trails = num_of_trails
        N = self.target_X_all.shape[1]
        N_target_samples = int(N * self.target_sample_ratio)
        self.num_target_sample = N_target_samples
        data_split_list = list(range(N))
        random.shuffle(data_split_list)
        self.target_X = self.target_X_all[:, data_split_list[:N_target_samples]]
        self.target_Y = self.target_Y_all[:, data_split_list[:N_target_samples]]
        self.target_X_test = self.target_X_all[:, data_split_list[N_target_samples:]]
        self.target_Y_test = self.target_Y_all[:, data_split_list[N_target_samples:]]

    def compute_source_target_diff(self):
        if len(self.list_of_model) > 1:
            raise Exception("not implemented")
        model = self.list_of_model[0]
        grads_S = model.get_gradients(self.source_X, self.source_Y)
        grads_T = model.get_gradients(self.target_X_all, self.target_Y_all)
        diff = torch.norm(grads_T - grads_S)
        return diff

    def compute_projected_grads_norm_square(self, normalized=False):
        if len(self.list_of_model) > 1:
            raise Exception("not implemented")
        model = self.list_of_model[0]
        grads_S = model.get_gradients(self.source_X, self.source_Y)
        grads_T = model.get_gradients(self.target_X_all, self.target_Y_all)
        projected_grad = grads_T - torch.sum(grads_T * grads_S) * grads_S / torch.norm(grads_S) ** 2
        projected_grad_norm_square = projected_grad.norm() ** 2
        if normalized:
            return projected_grad_norm_square / len(grads_T)
        else:
            return projected_grad_norm_square

    def compute_source_target_var(self, normalized=False):
        if len(self.list_of_model) > 1:
            raise Exception("not implemented")
        model = self.list_of_model[0]
        grads_S = model.get_gradients(self.source_X, self.source_Y)
        grads_T = model.get_gradients(self.target_X_all, self.target_Y_all)
        if normalized:
            return torch.norm(grads_T - grads_S).item() ** 2 / len(grads_T)
        else:
            return torch.norm(grads_T - grads_S).item() ** 2

    def compute_target_var(self, normalized=False):
        diff = 0.
        if len(self.list_of_model) > 1:
            raise Exception("not implemented")
        for model in self.list_of_model:
            for i in range(self.num_of_trails):
                diff += self._compute_target_var_single_model(model, normalized)
        return diff / self.num_of_trails


    def _compute_target_var_single_model(self, model, normalized):
        # subsample data
        N = self.target_X_all.shape[1]
        N_target_samples = int(N * self.target_sample_ratio)
        data_split_list = list(range(N))
        random.shuffle(data_split_list)
        data_split_list = data_split_list[:N_target_samples]
        X_target = self.target_X_all[:, data_split_list]
        Y_target = self.target_Y_all[:, data_split_list]
        grads_T = model.get_gradients(X_target, Y_target)
        grads_T_all = model.get_gradients(self.target_X_all, self.target_Y_all)
        if normalized:
            return torch.norm(grads_T - grads_T_all).item() ** 2 / len(grads_T)
        else:
            return torch.norm(grads_T - grads_T_all).item() ** 2

    def compute_delta_error_gp(self, beta, normalized=False):
        delta_error = 0.
        if len(self.list_of_model) > 1:
            raise Exception("not implemented")
        for model in self.list_of_model:
            for i in range(self.num_of_trails):
                delta_error += self._compute_delta_error_gp_single_model(model, beta, normalized)
        return delta_error / self.num_of_trails

    def _compute_delta_error_gp_single_model(self, model, beta, normalized):
        # subsample data
        N = self.target_X_all.shape[1]
        N_target_samples = int(N * self.target_sample_ratio)
        data_split_list = list(range(N))
        random.shuffle(data_split_list)
        data_split_list = data_split_list[:N_target_samples]
        X_target = self.target_X_all[:, data_split_list]
        Y_target = self.target_Y_all[:, data_split_list]
        grads_T = model.get_gradients(X_target, Y_target)
        grads_T_all = model.get_gradients(self.target_X_all, self.target_Y_all)
        grads_S = model.get_gradients(self.source_X, self.source_Y)
        normalized_grad_S = grads_S / torch.norm(grads_S)
        grad_gp = (1-beta) * grads_T + beta * torch.sum(grads_T * normalized_grad_S) * normalized_grad_S
        if normalized:
            return torch.norm(grad_gp - grads_T_all).item() ** 2 / len(grads_T)
        else:
            return torch.norm(grad_gp - grads_T_all).item() ** 2

    def compute_tau(self):
        # see section 4.2 for definition
        eps = 0.001 # room to numerical error
        model = self.list_of_model[0]
        grads_S = model.get_gradients(self.source_X, self.source_Y)
        grads_T = model.get_gradients(self.target_X_all, self.target_Y_all)
        diff = torch.norm(grads_T - grads_S) ** 2
        cos_rho = (grads_S * grads_T).sum() / torch.norm(grads_T) / torch.norm(grads_S)
        sin_rho = (1-cos_rho**2)**0.5
        if diff < eps:
            tau = 0.
        else:
            tau = (torch.norm(grads_T) * sin_rho / diff ** 0.5).item()
        return tau

class NN:
    def __init__(self, n, d, m, device):
        # n: input dim
        # d: output dim
        # m: hidden layer dim
        self.n = n
        self.d = d
        self.m = m
        self.num_parameters = m * n + m + d * m + d
        # self.num_parameters = m * d
        self.W1 = torch.rand([m, n], device=device) - 0.5
        self.b1 = torch.rand([m, 1], device=device) - 0.5
        self.W2 = torch.rand([d, m], device=device) - 0.5
        self.b2 = torch.rand([d, 1], device=device) - 0.5
        self.W1.requires_grad = True
        self.W2.requires_grad = True
        self.b1.requires_grad = True
        self.b2.requires_grad = True
        self.sigmoid = torch.nn.Sigmoid()
        self.g_W1 = 0.
        self.g_b1 = 0.
        self.g_W2 = 0.
        self.g_b2 = 0.

    def forward(self, X, requires_grad=False):
        # X should be (n, N), where N is num of samples
        X.requires_grad = requires_grad
        Y_pred = self.W2.mm(self.sigmoid(self.W1.mm(X) + self.b1)) + self.b2
        return Y_pred

    def get_gradients(self, X: torch.Tensor, Y: torch.Tensor, is_flatten=True):
        Y_pred = self.forward(X)
        loss = MSE(Y, Y_pred)
        loss.backward()
        if is_flatten:
            grads = self._get_flattened_grad()
        else:
            grads = (self.W1.grad.detach().clone(), self.b1.grad.detach().clone(), self.W2.grad.detach().clone(),
                     self.b2.grad.detach().clone())
            self.W1.grad.zero_()
            self.W2.grad.zero_()
            self.b1.grad.zero_()
            self.b2.grad.zero_()
        return grads

    def test(self, dataset:Metrics_n_Datasets):
        Y_pred = self.forward(dataset.target_X_test)
        loss = MSE(dataset.target_Y_test, Y_pred)
        return loss

    def train(self, dataset:Metrics_n_Datasets , num_epoch, method, beta=0.5, lr=0.2, momentum=0.9):
        self.g_W1 = 0.
        self.g_b1 = 0.
        self.g_W2 = 0.
        self.g_b2 = 0.
        losses = []
        for epoch in range(num_epoch):
            grad_S = self.get_gradients(dataset.source_X, dataset.source_Y, is_flatten=False)
            grad_T = self.get_gradients(dataset.target_X, dataset.target_Y, is_flatten=False)
            grads_T_all = self.get_gradients(dataset.target_X_all, dataset.target_Y_all, is_flatten=False)
            if method == 'gradient_proj':
                W1_update = gradient_proj(grad_T[0], grad_S[0], beta)
                b1_update = gradient_proj(grad_T[1], grad_S[1], beta)
                W2_update = gradient_proj(grad_T[2], grad_S[2], beta)
                b2_update = gradient_proj(grad_T[3], grad_S[3], beta)
            elif method == 'convex_combine':
                W1_update = convex_combine(grad_T[0], grad_S[0], beta)
                b1_update = convex_combine(grad_T[1], grad_S[1], beta)
                W2_update = convex_combine(grad_T[2], grad_S[2], beta)
                b2_update = convex_combine(grad_T[3], grad_S[3], beta)
            elif method == 'initialization_train':
                W1_update = grads_T_all[0]
                b1_update = grads_T_all[1]
                W2_update = grads_T_all[2]
                b2_update = grads_T_all[3]
            else:
                raise NameError
            with torch.no_grad():
                self.g_W1 = momentum * self.g_W1 + lr * W1_update
                self.g_b1 = momentum * self.g_b1 + lr * b1_update
                self.g_W2 = momentum * self.g_W2 + lr * W2_update
                self.g_b2 = momentum * self.g_b2 + lr * b2_update
                self.W1 -= lr * self.g_W1
                self.W2 -= lr * self.g_W2
                self.b1 -= lr * self.g_b1
                self.b2 -= lr * self.g_b2
                self.W1.grad.zero_()
                self.W2.grad.zero_()
                self.b1.grad.zero_()
                self.b2.grad.zero_()

            # if epoch % 40 == 0:
            #     loss = self.test(dataset)
            #     print('loss at epoch {} is {}'.format(epoch, loss))

        loss = self.test(dataset)
            # losses.append(loss.item())
        print('final loss at epoch {} is {}'.format(epoch, loss))
        return loss

    # def optimize(self, momentum=0.9):
    #     with torch.no_grad():
    #         self.g_W1 = momentum * self.g_W1 + self.lr * self.W1.grad
    #         self.g_b1 = momentum * self.g_b1 + self.lr * self.b1.grad
    #         self.g_W2 = momentum * self.g_W2 + self.lr * self.W2.grad
    #         self.g_b2 = momentum * self.g_b2 + self.lr * self.b2.grad
    #         self.W1 -= self.g_W1
    #         self.W2 -= self.lr * self.g_W2
    #         self.b1 -= self.lr * self.g_b1
    #         self.b2 -= self.lr * self.g_b2
    #         self.W1.grad.zero_()
    #         self.W2.grad.zero_()
    #         self.b1.grad.zero_()
    #         self.b2.grad.zero_()

    def _get_flattened_grad(self):
        grad = torch.empty(self.num_parameters)
        with torch.no_grad():
            start = 0
            grad[start: self.m * self.n] = self.W1.grad.flatten()
            start += self.m * self.n
            grad[start: start + self.m] = self.b1.grad.flatten()
            start += self.m
            grad[start: start + self.m * self.d] = self.W2.grad.flatten()
            start += self.m * self.d
            grad[start:] = self.b2.grad.flatten()
            # grad[:] = self.W2.grad.flatten()
            self.W1.grad.zero_()
            self.W2.grad.zero_()
            self.b1.grad.zero_()
            self.b2.grad.zero_()
        return grad


    def save_model(self, save_handle):
        model_weights = {"W1": self.W1.detach(), "W2": self.W2.detach(), "b1": self.b1.detach(), "b2": self.b2.detach()}
        torch.save(model_weights, save_handle)

    def load_model(self, save_handle, device):
        model_weights = torch.load(save_handle)
        self.W1 = model_weights['W1'].to(device)
        self.W2 = model_weights['W2'].to(device)
        self.b1 = model_weights['b1'].to(device)
        self.b2 = model_weights['b2'].to(device)
        self.W1.requires_grad = True
        self.W2.requires_grad = True
        self.b1.requires_grad = True
        self.b2.requires_grad = True


    def jacobian(self, x: torch.Tensor):
        with torch.no_grad():
            return self.W2.mm(diagonalize(sigmoid_derivative(self.W1.mm(x.reshape([-1, 1])) + self.b1)).mm(self.W1))





class NNPretrained(NN):
    def __init__(self, n, d, m, lr, W1: torch.Tensor, b1: torch.Tensor, W2: torch.Tensor, b2: torch.Tensor):
        # n: input dim
        # d: output dim
        # m: hidden layer dim
        super(NNPretrained, self).__init__(n, d, m, lr)
        self.W1 = W1.clone().detach()
        self.b1 = b1.clone().detach()
        self.W1.requires_grad = False
        self.b1.requires_grad = False
        self.W2 = W2.clone().detach()
        self.b2 = b2.clone().detach()
        self.W2.requires_grad = True
        self.b2.requires_grad = True

    def prepare_train(self):
        self.W2.requires_grad = True
        self.b2.requires_grad = True

    def optimize(self):
        with torch.no_grad():
            self.W2 -= self.lr * self.W2.grad
            self.b2 -= self.lr * self.b2.grad
            self.W2.grad.zero_()
            self.b2.grad.zero_()



def diagonalize(x):
    n = len(x)
    D = torch.zeros([n, n])
    for i in range(n):
        D[i, i] = x[i]
    return D

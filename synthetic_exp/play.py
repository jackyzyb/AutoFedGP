import torch

def diagonalize(x):
    n = len(x)
    D = torch.zeros([n, n])
    for i in range(n):
        D[i, i] = x[i]
    return D

def jacobian(x: torch.Tensor, W1, b1):
    with torch.no_grad():
        return diagonalize(sigmoid_derivative(W1.mm(x.reshape([-1, 1])) + b1)).mm(W1)

def sigmoid_derivative(X: torch.Tensor):
    E = torch.exp(-X)
    return E / (1 + E).pow(2)

sigmoid = torch.nn.Sigmoid()

W1 = torch.tensor([[1., 2., 3., 4.],[2.,6.,-5., -10.]]) * 0.1
x = torch.tensor([5.,2.,1.,-1.], requires_grad=True)
b1 = torch.tensor([[2.], [-5.]]) * 0.1

z = sigmoid(W1.mm(x.reshape([-1, 1])) + b1)
i = 1
z[i].backward()
print('gradient at {} is {}'.format(i, x.grad))

jacbo = jacobian(x, W1, b1)
print('jacobian is {}'.format(jacbo))



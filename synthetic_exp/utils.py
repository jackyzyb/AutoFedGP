import torch
import random


def convex_combine(grad_T, grad_S, beta):
    return beta * grad_S + (1 - beta) * grad_T

def gradient_proj(grad_T, grad_S, beta):
    inner_prod = torch.sum(grad_T * grad_S)
    if inner_prod < 0:
        inner_prod = 0
    return beta * inner_prod / torch.norm(grad_S)**2 * grad_S + (1 - beta) * grad_T


def MSE(Y, Y_pred):
    N = Y.shape[1]  # num of data
    return ((Y - Y_pred) ** 2).sum() / N
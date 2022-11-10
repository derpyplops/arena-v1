import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
from typing import Optional, Callable
import ipywidgets as wg
from fancy_einsum import einsum
import math

import utils
import torch as t

PI = math.pi

def DFT_1d(arr : np.ndarray, inverse=False) -> np.ndarray:
    """
    Returns the DFT of the array `arr`, using the equation above.
    """
    n = len(arr)
    w = math.e ** (-2j * math.pi / n)
    if inverse:
        w = 1/w
    e = np.outer(np.arange(0,n), np.arange(0,n))
    L = np.power(w, e)
    res = einsum('i k,k->i', L, arr)
    if inverse:
        res = res / n
    return res

def dft_test(my_dft):
    np.testing.assert_allclose(my_dft(np.array([1, 2-1j, -1j, -1+2j])), np.array([2,-2-2j,-2j,4+4j]))

def integrate_function(func: Callable, x0: float, x1: float, n_samples: int = 1000):
    """
    Calculates the approximation of the Riemann integral of the function `func`, 
    between the limits x0 and x1.

    You should use the Left Rectangular Approximation Method (LRAM).
    height is what's on the left so each point informs rectagnle on the right
    """
    step = (x1-x0)/n_samples
    pts = np.arange(x0, x1, step)
    area = 0
    for pt in pts:
        area += func(pt) * step
    return area

utils.test_integrate_function(integrate_function)

def integrate_product(func1: Callable, func2: Callable, x0: float, x1: float):
    """
    Computes the integral of the function x -> func1(x) * func2(x).
    """
    func = lambda x: func1(x) * func2(x)
    return integrate_function(func, x0, x1)

utils.test_integrate_product(integrate_product)


# utils.test_DFT_func(DFT_1d)
# dft_test(DFT_1d)

def calculate_fourier_series(func: Callable, max_freq: int = 50):
    """
    Calculates the fourier coefficients of a function, 
    assumed periodic between [-pi, pi].

    Your function should return ((a_0, A_n, B_n), func_approx), where:
        a_0 is a float
        A_n, B_n are lists of floats, with n going up to `max_freq`
        func_approx is the fourier approximation, as described above
    """
    a0 = 1 / PI * integrate_function(func, -PI, PI)
    an = lambda n: 1 / PI * integrate_product(func, lambda x: np.cos(n*x), -PI, PI)
    bn = lambda n: 1 / PI * integrate_product(func, lambda x: np.sin(n*x), -PI, PI)
    An = map(an, range(1,max_freq+1))
    Bn = map(bn, range(1,max_freq+1))
    # func_approx = lambda x: 1/2 * a0 + sum(map(lambda a,n: a * math.cos(n * x), enumerate(An)))
    def func_approx(x: float):
        sec = 0
        for n, a in enumerate(An):
            sec += a * np.cos(n*x)
        third = 0
        for n, b in enumerate(Bn):
            third += b * np.sin(n*x)

        return 1/2 * a0 + sec + third
    return ((a0, An, Bn), func_approx)

step_func = lambda x: 1 * (x > 0)
# utils.create_interactive_fourier_graph(calculate_fourier_series, func = step_func)

# === Part 2 ===


NUM_FREQUENCIES = 2
TARGET_FUNC = lambda x: 1. * (x > 1.)
TOTAL_STEPS = 4000
LEARNING_RATE = 1e-6

x = t.linspace(-t.pi, t.pi, 2000)
y = TARGET_FUNC(x)

x_cos = t.stack([t.cos(n*x) for n in range(1, NUM_FREQUENCIES+1)])
x_sin = t.stack([t.sin(n*x) for n in range(1, NUM_FREQUENCIES+1)])

# input_batch = t.concat([x_cos, x_sin], dim=0).T # t.vstack((x_cos, x_sin)).T # einsum('n a, n b', x_cos, x_sin)
input_batch = t.concat([x_cos, x_sin], dim=0).T # we use .T so that it the 0th axis is batch dim

print(f'dim of in batch: {input_batch.shape}')

a_0 = t.randn(1, requires_grad=True)
A_n = t.randn(NUM_FREQUENCIES, requires_grad=True)
B_n = t.randn(NUM_FREQUENCIES, requires_grad=True)

# class OneLayer(t.nn.Module):
#     def __init__(self, D_in, D_out):
#         super(OneLayer, self).__init__()
#         self.linear = t.nn.Sequential(
#             t.nn.Linear(D_in, D_out),
#             t.nn.Flatten()
#         )
    
#     def forward(self,x):
#         mid = len(x) // 2
#         print(x.shape, x_cos.shape, x_sin.shape)
#         einsum('', x, x_cos, x_sin)
#         y_pred = 1/2 * a_0 + (x_cos.T @ x[:mid]) + (x_sin.T @ x[mid:])
#         return y_pred

# optimizer.zero_grad()  # clear previous gradients
# loss.backward()        # compute gradients of all variables wrt loss

# optimizer.step()       # perform updates using calculated gradients


y_pred_list = []
coeffs_list = []

model = t.nn.Sequential(
            t.nn.Linear(2*NUM_FREQUENCIES, 1),
            t.nn.Flatten(0,1)
        )

optimizer = t.optim.Adagrad(model.parameters(), lr=LEARNING_RATE*50000)

for step in range(TOTAL_STEPS):

    y_pred = model(input_batch)
    # y_pred = 1/2 * a_0 + (x_cos.T @ A_n) + (x_sin.T @ B_n)

    # loss = ((y-y_pred) ** 2).sum()
    loss_fn = t.nn.MSELoss(reduction='sum')
    loss = loss_fn(y_pred, y)
    t._cast_Float(loss)


    if step % 100 == 0:
        print(f"{loss = :.2f}")
        # coeffs_list.append([a_0, A_n.clone().detach().numpy(), B_n.clone().detach().numpy()])
        # y_pred_list.append(y_pred)
        A_n = list(model.parameters())[0].detach().numpy()[:3].squeeze()
        B_n = list(model.parameters())[0].detach().numpy()[:6].squeeze()
        a_0 = list(model.parameters())[1].item()
        y_pred_list.append(y_pred.cpu().detach().numpy())
        coeffs_list.append([a_0, A_n.copy(), B_n.copy()])

    # TODO: compute gradients of coeffs with respect to `loss`
    # dy = 2 * (y_pred - y)
    # da0 = (dy * 1/2).sum()
    
    # da = (dy @ x_cos.T)
    # db = (dy @ x_sin.T)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    

    # TODO update weights using gradient descent (using the parameter `LEARNING_RATE`)

    
    # with t.no_grad():
    #     for param in model.parameters():
    #         param -= LEARNING_RATE * param.grad
        # a_0 -= LEARNING_RATE * a_0.grad
        # A_n -= LEARNING_RATE * A_n.grad
        # B_n -= LEARNING_RATE * B_n.grad
    # a_0.grad.zero_()
    # A_n.grad.zero_()
    # B_n.grad.zero_()
    model.zero_grad()
    

# utils.visualise_fourier_coeff_convergence(x, y, y_pred_list, coeffs_list)
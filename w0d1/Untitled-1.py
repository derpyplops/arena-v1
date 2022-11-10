
NUM_FREQUENCIES = 2
TARGET_FUNC = lambda x: 1 * (x > 1)
TOTAL_STEPS = 4000
LEARNING_RATE = 1e-6

import torch
import math

dtype = torch.float
device = torch.device("cpu")

x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = TARGET_FUNC(x)

x_cos = torch.stack([torch.cos(n*x) for n in range(1, NUM_FREQUENCIES+1)])
x_sin = torch.stack([torch.sin(n*x) for n in range(1, NUM_FREQUENCIES+1)])

x_all = torch.concat([x_cos, x_sin], dim=0).T # we use .T so that it the 0th axis is batch dim

LEARNING_RATE = 1e-6
TOTAL_STEPS = 4000

y_pred_list = []
coeffs_list = []

model = torch.nn.Sequential(torch.nn.Linear(2 * NUM_FREQUENCIES, 1), torch.nn.Flatten(0, 1))

for step in range(TOTAL_STEPS):
    
    # Forward pass: compute predicted y
    y_pred = model(x_all)
    
    # Compute and print loss
    loss = torch.square(y - y_pred).sum()
    if step % 100 == 0:
        print(f"{loss = :.2f}")
        A_n = list(model.parameters())[0].detach().numpy()[:3].squeeze()
        B_n = list(model.parameters())[0].detach().numpy()[:6].squeeze()
        a_0 = list(model.parameters())[1].item()
        y_pred_list.append(y_pred.cpu().detach().numpy())
        coeffs_list.append([a_0, A_n.copy(), B_n.copy()])
    
    # Backprop to compute gradients of coeffs with respect to loss
    loss.backward()
    
    # Update weights using gradient descent
    with torch.no_grad():
        for param in model.parameters():
            param -= LEARNING_RATE * param.grad
    model.zero_grad()
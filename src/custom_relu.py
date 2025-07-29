import torch
import torch.nn as nn


class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        grad_in = grad_out.clone()
        grad_in[x <= 0] = 0
        print("MyReLU backward called")  # proof
        return grad_in


class MyReLUModule(nn.Module):  # <‑‑ thin wrapper
    def forward(self, x):
        return MyReLU.apply(x)

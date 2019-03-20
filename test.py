import torch
from torch.autograd import Variable
import pdb

def basic_fun(x):
    return 3*(x*x)

def get_grad(inp, grad_var):
    A = basic_fun(inp)
    A.backward()
    return grad_var.grad

x = Variable(torch.randn(2,2), requires_grad=True)
y = Variable(torch.zeros(2,2), requires_grad=True)

pdb.set_trace()

A = basic_fun(y)
A.backward()


# -*- coding: utf-8 -*-
"""
    pytorch learning

"""

import torch
# from torch.autograd import Variable
from torch.autograd.variable import Variable

a = torch.ones(3, 5, requires_grad=True)
# a=torch.ones(3,5)
print('a:', a)

b = a + 2
print('b:', b)

for i in range(3):
    # b.backwward()
    a.zero_grad()

    z = b.sum()
    z.backward()  # d(z)/da
    print('i:', i, z)
    print(a.grad)
    print(b.grad)

c = Variable(a)
c.requires_grad = True

c.backward()

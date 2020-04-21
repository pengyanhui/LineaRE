import torch
import numpy as np

lst = {(1, 2): 11}

x = np.array([1, 2, 3])
print(x.size)
print(x[:2])
print(lst[(x[0], x[1])])
print(tuple(x))

a = torch.rand(3, 3)
m = torch.max(a, dim=-1)
print(a)
print(m)
tk_val, tk_idx = torch.topk(a, dim=-1, k=2)
print(tk_idx)
print(torch.index_select(a[0], dim=-1, index=tk_idx[0]))

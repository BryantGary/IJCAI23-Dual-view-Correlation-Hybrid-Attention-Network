import torch

def corr(x, y, epsilon=0.0001):
    n, c, _, _ = x.size()
    x_ = x.view(n, c, -1)
    #print("x_:",x_)
    y_ = y.view(n, c, -1)
    #print("y_:",y_)
    x1 = x_ - x_.mean(-1, keepdim=True)
    #print("x1:",x1)
    y1 = y_ - y_.mean(-1, keepdim=True)
    #print("y1:",y1)
    x2 = x1/(torch.norm(x1, dim=-1, keepdim=True) + epsilon)
    #print("x2:",x2)
    y2 = y1/(torch.norm(y1, dim=-1, keepdim=True) + epsilon)
    #print("y2:",y2)
    #print("x2*y2:",torch.mean(x2 * y2))
    return 10* torch.mean(x2 * y2)


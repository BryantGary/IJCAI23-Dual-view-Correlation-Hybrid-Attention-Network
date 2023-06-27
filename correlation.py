import torch

def corr(x, y, epsilon=0.0001):
    n, c, _ = x.size()

    x_ = x.view(n, c, -1)    
    y_ = y.view(n, c, -1)

    x1 = x_ - x_.mean(-1, keepdim=True)
    y1 = y_ - y_.mean(-1, keepdim=True)

    x2 = x1/(torch.norm(x1, dim=-1, keepdim=True) + epsilon)
    y2 = y1/(torch.norm(y1, dim=-1, keepdim=True) + epsilon)

    return 10* torch.mean(x2 * y2)


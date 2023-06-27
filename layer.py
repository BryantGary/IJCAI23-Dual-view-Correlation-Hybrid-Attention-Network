import torch
import torch.backends.cudnn as cudnn

class GeometryPrior(torch.nn.Module):
    def __init__(self, k, channels, multiplier=0.5):
        super(GeometryPrior, self).__init__()
        self.channels = channels
        self.k = k
        self.position = 2 * torch.rand(1, 2, k, k, requires_grad=True)- 1
        self.position = self.position.cuda()
        self.l1 = torch.nn.Conv2d(2, int(multiplier * channels), 1)
        self.l2 = torch.nn.Conv2d(int(multiplier * channels), channels, 1)
        
    def forward(self, x):
        x = self.l2(torch.nn.functional.relu(self.l1(self.position)))
        return x.view(1, self.channels, 1, self.k ** 2)



class KeyQueryMap(torch.nn.Module):
    def __init__(self, channels, m):
        super(KeyQueryMap, self).__init__()
        self.l = torch.nn.Conv2d(channels, channels // m, 1)#3,stride=1,padding=1)
    
    def forward(self, x):
        return self.l(x)


class AppearanceComposability(torch.nn.Module):
    def __init__(self, k, padding, stride):
        super(AppearanceComposability, self).__init__()
        self.k = k
        self.unfold = torch.nn.Unfold(k, 1, padding, stride)
    
    def forward(self, x):
        key_map, query_map = x
        k = self.k
        key_map_unfold = self.unfold(key_map)
        
        query_map_unfold = self.unfold(query_map)
        key_map_unfold = key_map_unfold.view(
                    key_map.shape[0], key_map.shape[1],
                    -1,
                    key_map_unfold.shape[-2] // key_map.shape[1])
        query_map_unfold = query_map_unfold.view(
                    query_map.shape[0], query_map.shape[1],
                    -1,
                    query_map_unfold.shape[-2] // query_map.shape[1])
        
        return key_map_unfold * query_map_unfold[:, :, :, k**2//2:k**2//2+1] #-1*((query_map_unfold[:, :, :, k**2//2:k**2//2+1]-key_map_unfold)**2) #


def combine_prior(appearance_kernel, geometry_kernel):
    return  torch.nn.functional.softmax(appearance_kernel + geometry_kernel,
                                    dim=-1)


class LocalRelationalLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels,k, stride=2, m=None, padding=0):
        super(LocalRelationalLayer, self).__init__()
        self.channels = in_channels
        self.k = k
        self.stride = stride
        self.m = m or 8
        self.padding = padding
        self.kmap = KeyQueryMap(in_channels, self.m)
        self.qmap = KeyQueryMap(in_channels, self.m)
        self.ac = AppearanceComposability(k, padding, stride)
        self.gp = GeometryPrior(k, in_channels//m)
        self.unfold = torch.nn.Unfold(k, 1, padding, stride)
        self.final1x1 = torch.nn.Conv2d(in_channels, out_channels, 1)
        
    def forward(self, x):
        #print("x:",x.shape)
        gpk = self.gp(0)
        #print("gpk:", gpk.shape)
        # m = 8
        # 1, 64/8, 1, 49
        x=x.cuda()
        km = self.kmap(x)
       
        # 128, 9, 17, 17
        qm = self.qmap(x)
       
        ak = self.ac((km, qm))#[:, None, :, :, :]
        # 128 * 8 * 36 * 49
        ck = combine_prior(ak, gpk)[:, None, :, :, :]
        #print(ck.type())
        # ck 128 * 1 * 8 * 36 * 49
        x_unfold = self.unfold(x)
        # 128, 3136, 36
        x_unfold = x_unfold.view(x.shape[0], self.m, x.shape[1] // self.m,
                                 -1, x_unfold.shape[-2] // x.shape[1])
       
        # x_unfold:', (128, 8, 8, 36, 49
        pre_output = (ck * x_unfold).view(x.shape[0], x.shape[1],
                                          -1, ck.shape[-1])
       
        h_out = (x.shape[2] + 2 * self.padding - 1 * (self.k - 1) - 1) // \
                                                            self.stride + 1
        w_out = (x.shape[3] + 2 * self.padding - 1 * (self.k - 1) - 1) // \
                                                            self.stride + 1                                                                            
        pre_output = torch.sum(pre_output,dim=-1).view(x.shape[0], x.shape[1],
                                                         h_out, w_out)
        return self.final1x1(pre_output)

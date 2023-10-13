import numpy as np
import torch
from torch import nn
import odl
import os
from odl.contrib import torch as odl_torch
import warnings

warnings.filterwarnings('ignore')


class Forward_projection(nn.Module):  #torch.autograd.Function
    def __init__(self,device = 'skimage'):
        super(Forward_projection, self).__init__()
        reco_space = odl.uniform_discr(min_pt = [-20,-20],max_pt=[20,20],shape = [128,128])
        geometry = odl.tomo.parallel_beam_geometry(reco_space)
        projector = odl.tomo.RayTransform(reco_space, geometry, impl = device)
        
        self.FP_layer = odl_torch.OperatorAsModule(projector)
        
    def forward(self, x):
        x = x/12
        out = self.FP_layer.forward(x)
        return out
    
    
class Backward_projection(nn.Module):
    def __init__(self,device = 'skimage'):
        super(Backward_projection, self).__init__()
        reco_space = odl.uniform_discr(min_pt = [-20,-20],max_pt=[20,20],shape = [128,128])
        geometry = odl.tomo.parallel_beam_geometry(reco_space)
        projector = odl.tomo.RayTransform(reco_space, geometry, impl = device)
        FBP = odl.tomo.fbp_op(projector)
        
        self.BP_layer = odl_torch.OperatorAsModule(FBP)
        
    def forward(self, x):
        x = x*12
        out = self.BP_layer(x)

        return out
    
if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    FP = Forward_projection('astra_cuda')
    BP = Backward_projection('astra_cuda')
    a = torch.zeros([5,1,128,128]).to(device)
    b = FP(a)
    print(b.shape)
    c = BP(b)
    print(c.shape)
import torch
from torch.nn import Module
from torch.nn import functional as F
#%%
class SmoothL1Loss(Module):
    def __init__(self, use_gpu):
        super (SmoothL1Loss, self).__init__()
        self.use_gpu = use_gpu
        return
    
    def forward(self, clabel, target, routput, rlabel):
        
#        rloss = F.smooth_l1_loss(routput, rlabel)
        rloss = F.smooth_l1_loss(routput, rlabel, size_average=False, reduce=False)
        
            
        e = torch.eq(clabel.float(), target) 
        e = e.squeeze()
        e0,e1,e2,e3,e4 = e[0].unsqueeze(0),e[1].unsqueeze(0),e[2].unsqueeze(0),e[3].unsqueeze(0),e[4].unsqueeze(0)
        eq = torch.cat([e0,e0,e0,e0,e1,e1,e1,e1,e2,e2,e2,e2,e3,e3,e3,e3,e4,e4,e4,e4], dim=0).float()
        
        rloss = rloss.squeeze()
        rloss = torch.mul(eq, rloss)
        rloss = torch.sum(rloss)
        rloss = torch.div(rloss, eq.nonzero().shape[0]+1e-4)
        return rloss
#%%
class Myloss(Module):
    def __init__(self):
        super (Myloss, self).__init__()
        return 
    
    def forward(self, coutput, clabel, target, routput, rlabel, lmbda):
        closs = F.cross_entropy(coutput, clabel)

#        rloss = F.smooth_l1_loss(routput, rlabel)
        rloss = F.smooth_l1_loss(routput, rlabel, size_average=False, reduce=False)
        
            
        e = torch.eq(clabel.float(), target) 
        e = e.squeeze()
        e0,e1,e2,e3,e4 = e[0].unsqueeze(0),e[1].unsqueeze(0),e[2].unsqueeze(0),e[3].unsqueeze(0),e[4].unsqueeze(0)
        eq = torch.cat([e0,e0,e0,e0,e1,e1,e1,e1,e2,e2,e2,e2,e3,e3,e3,e3,e4,e4,e4,e4], dim=0).float()
        
        rloss = rloss.squeeze()
        rloss = torch.mul(eq, rloss)
        rloss = torch.sum(rloss)
        rloss = torch.div(rloss, eq.nonzero().shape[0]+1e-4)
        
        loss = torch.add(closs, lmbda, rloss)
        return loss
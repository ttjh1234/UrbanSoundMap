import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math



class weighted_loss(nn.Module):
    def __init__(self,baseline = 50, weight = 2):
        super(weighted_loss, self).__init__()
        self.baseline = baseline
        self.weight = weight

    def forward(self,pred,y):
        data_weight = torch.where(y>=self.baseline, self.weight, 1.0)
        weight_sum = torch.sum(data_weight,dim=0)
        weight = data_weight / weight_sum
        
        loss = torch.sum(weight * (y - pred).pow(2))
                
        return loss

class weighted_loss2(nn.Module):
    def __init__(self, weight = 2):
        super(weighted_loss2, self).__init__()
        self.weight = weight

    def forward(self,pred,y, masking):
        data_weight = torch.where(masking == 1, self.weight, 1.0)
        weight_sum = torch.sum(data_weight,dim=0)
        weight = data_weight / weight_sum
        
        loss = torch.sum(weight * (y - pred).pow(2))
                
        return loss
    
    
class point_and_quantile_loss(nn.Module):
    def __init__(self, weight_0=1/3, weight_1 = 1/3, weight_2 = 1/3,upper_q=0.975):
        super(point_and_quantile_loss, self).__init__()
        assert (weight_0<1) & (weight_0>0)
        assert (weight_1<1) & (weight_1>0)
        assert (weight_2<1) & (weight_2>0)
        self.weight0 = weight_0
        self.weight1 = weight_1
        self.weight2 = weight_2
        self.upper_q = upper_q
        
    def pinball_loss(self, pred, y, q):
        errors = y-pred
        
        return torch.mean(torch.max(q*errors, (q-1)*errors))
    
    def forward(self, pred, y):
        point_pred = pred[:,0]
        upper_pred = pred[:,1]
        delta_pred = pred[:,2]
        
        point_loss = torch.mean((point_pred - y)**2)
        upper_loss = self.pinball_loss(upper_pred, y, self.upper_q)
        lower_loss = self.pinball_loss(F.relu(upper_pred - delta_pred), y, 1-self.upper_q)
        
        loss = self.weight0 * point_loss + self.weight1 * upper_loss + self.weight2 * lower_loss
        
        return loss
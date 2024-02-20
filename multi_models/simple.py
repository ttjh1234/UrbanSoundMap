from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

class image_classification(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.class_num=num_classes
        self.conv1=nn.Conv2d(1,16,3,stride=1,padding=1)
        self.conv2=nn.Conv2d(16,64,3,stride=1,padding=1)
        self.conv3=nn.Conv2d(64,128,3,stride=1,padding=1)
        self.fc1=nn.Linear(2048,512)
        self.fc2=nn.Linear(512,self.class_num)

    def forward(self, src, is_feat=False, preact=False):
        # src : B, 3, 32, 32
        out1=F.tanh(self.conv1(src))
        out2=F.max_pool2d(out1,2) # B, 16, 16, 16
        out2=F.tanh(self.conv2(out2))
        out3=F.max_pool2d(out2,2) # B, 64, 8, 8
        out3=F.tanh(self.conv3(out3)) 
        out4=F.max_pool2d(out3,2) # B, 128, 4, 4
        
        out4=torch.flatten(out4,1)
        out=F.relu(self.fc1(out4))
        out=self.fc2(out)
        
        if is_feat:
            return [out1,out2,out3,out4], out
        
        else:        
            return out




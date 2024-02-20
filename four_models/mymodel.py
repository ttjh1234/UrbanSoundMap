from __future__ import absolute_import
'''VGG for CIFAR10. FC layers are removed.
(c) YANG, Wei
'''
import torch.nn as nn
import torch.nn.functional as F
import math
import torch

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}


class VGGmy(nn.Module):

    def __init__(self, cfg, batch_norm=False, num_classes=1000, device='cuda:0'):
        super(VGGmy, self).__init__()
        self.block0 = self._make_layers(cfg[0], batch_norm, 4)
        self.block1 = self._make_layers(cfg[1], batch_norm, cfg[0][-1])
        self.block2 = self._make_layers(cfg[2], batch_norm, cfg[1][-1])
        self.block3 = self._make_layers(cfg[3], batch_norm, cfg[2][-1])
        self.block4 = self._make_layers(cfg[4], batch_norm, cfg[3][-1])

        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.value = nn.Linear(512,1)
        self.query = nn.Linear(512,256)
        self.key = nn.Linear(512,256)        
        self.pos_embedding = nn.Embedding(144, 512)
        self.device= device
        #self.classifier = nn.Linear(144, num_classes)
        self._initialize_weights()

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.block0)
        feat_m.append(self.pool0)
        feat_m.append(self.block1)
        feat_m.append(self.pool1)
        feat_m.append(self.block2)
        feat_m.append(self.pool2)
        feat_m.append(self.block3)
        feat_m.append(self.pool3)
        feat_m.append(self.block4)
        feat_m.append(self.pool4)
        return feat_m

    def get_bn_before_relu(self):
        bn1 = self.block1[-1]
        bn2 = self.block2[-1]
        bn3 = self.block3[-1]
        bn4 = self.block4[-1]
        return [bn1, bn2, bn3, bn4]

    def forward(self, x, is_feat=False, preact=False):
        h = x.shape[2]
        x = F.relu(self.block0(x))
        f0 = x
        x = self.pool0(x)
        x = self.block1(x)
        f1_pre = x
        x = F.relu(x)
        f1 = x
        x = self.pool1(x)
        x = self.block2(x)
        f2_pre = x
        x = F.relu(x)
        f2 = x
        x = self.pool2(x)
        x = self.block3(x)
        f3_pre = x
        x = F.relu(x)
        f3 = x
        if h == 64:
            x = self.pool3(x)
        x = self.block4(x)
        f4_pre = x
        x = F.relu(x)
        f4 = x
            
        # x : B, 512, 12, 12
        # 각각의 픽셀들을 patch라고 생각하자. 그러면, 우리는 총 약 80m x 80을 표현하는 patch들이 있는 것이고, 그것의 차원이 512라고 생각
        # x' : B, 144, 512 에 포지션 embedding을 더해서,   
        pos = torch.arange(0,144).unsqueeze(0).repeat(x.shape[0],1).to(self.device)
    
        x = x.view(x.shape[0],x.shape[1],-1).permute(0,2,1) # B, 144, 512
        x = x + self.pos_embedding(pos)
        query = self.query(x) # B, 144, 256
        key = self.key(x)
        value = self.value(x)
        
        energy = torch.bmm(query, key.permute(0, 2, 1)) / 128**(1/2)
        attention = torch.softmax(energy,dim=-1) # B, 144, 144
                
        x = torch.bmm(attention, value).squeeze(2) # B, 144, 144 x B, 144,1 = B,144,1
        x = torch.mean(x,dim=1,keepdim=True)
        
        #x = self.pool4(f4)
        #x = x.view(x.size(0), -1)
        f5 = x
        #x = self.classifier(x)

        if is_feat:            
            if preact:
                return [f0, f1_pre, f2_pre, f3_pre, f4_pre, f5], x
            else:
                return [f0, f1, f2, f3, f4, f5], x
        else:
            return x
    
    
    @staticmethod
    def _make_layers(cfg, batch_norm=False, in_channels=3):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        layers = layers[:-1]
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()





class MaskVGGmy(nn.Module):

    def __init__(self, cfg, batch_norm=False, num_classes=1000, device='cuda:0'):
        super(MaskVGGmy, self).__init__()
        self.block0 = self._make_layers(cfg[0], batch_norm, 3)
        self.block1 = self._make_layers(cfg[1], batch_norm, cfg[0][-1])
        self.block2 = self._make_layers(cfg[2], batch_norm, cfg[1][-1])
        self.block3 = self._make_layers(cfg[3], batch_norm, cfg[2][-1])
        self.block4 = self._make_layers(cfg[4], batch_norm, cfg[3][-1])

        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.value = nn.Linear(512,1)
        self.query = nn.Linear(512,256)
        self.key = nn.Linear(512,256)        
        self.pos_embedding = nn.Embedding(144, 512)
        self.device= device
        #self.classifier = nn.Linear(144, num_classes)
        self._initialize_weights()

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.block0)
        feat_m.append(self.pool0)
        feat_m.append(self.block1)
        feat_m.append(self.pool1)
        feat_m.append(self.block2)
        feat_m.append(self.pool2)
        feat_m.append(self.block3)
        feat_m.append(self.pool3)
        feat_m.append(self.block4)
        feat_m.append(self.pool4)
        return feat_m

    def get_bn_before_relu(self):
        bn1 = self.block1[-1]
        bn2 = self.block2[-1]
        bn3 = self.block3[-1]
        bn4 = self.block4[-1]
        return [bn1, bn2, bn3, bn4]

    def forward(self, x, mask=None, is_feat=False, preact=False):
        h = x.shape[2]
        x = F.relu(self.block0(x))
        f0 = x
        x = self.pool0(x)
        x = self.block1(x)
        f1_pre = x
        x = F.relu(x)
        f1 = x
        x = self.pool1(x)
        x = self.block2(x)
        f2_pre = x
        x = F.relu(x)
        f2 = x
        x = self.pool2(x)
        x = self.block3(x)
        f3_pre = x
        x = F.relu(x)
        f3 = x
        if h == 64:
            x = self.pool3(x)
        x = self.block4(x)
        f4_pre = x
        x = F.relu(x)
        f4 = x
            
        # x : B, 512, 12, 12
        # 각각의 픽셀들을 patch라고 생각하자. 그러면, 우리는 총 약 80m x 80을 표현하는 patch들이 있는 것이고, 그것의 차원이 512라고 생각
        # x' : B, 144, 512 에 포지션 embedding을 더해서,   
        pos = torch.arange(0,144).unsqueeze(0).repeat(x.shape[0],1).to(self.device)
    
        x = x.view(x.shape[0],x.shape[1],-1).permute(0,2,1) # B, 144, 512
        x = x + self.pos_embedding(pos)
        query = self.query(x) # B, 144, 256
        key = self.key(x)
        value = self.value(x)
        
        energy = torch.bmm(query, key.permute(0, 2, 1)) / 128**(1/2)
        if mask is not None:
            mask = F.adaptive_avg_pool2d(mask, (12,12)) # B, 1, 12, 12
            mask = mask.view(x.shape[0],1,-1)
            energy = energy.masked_fill(mask == 0, -1e10)  # energy : B, 144, 144
        
        attention = torch.softmax(energy,dim=-1) # B, 144, 144
        
        x = torch.bmm(attention, value).squeeze(2) # B, 144, 144 x B, 144,1 = B,144,1
        x = torch.mean(x,dim=1,keepdim=True)
        
        #x = self.pool4(f4)
        #x = x.view(x.size(0), -1)
        f5 = x
        #x = self.classifier(x)

        if is_feat:            
            if preact:
                return [f0, f1_pre, f2_pre, f3_pre, f4_pre, f5], x
            else:
                return [f0, f1, f2, f3, f4, f5], x
        else:
            return x
    
    
    @staticmethod
    def _make_layers(cfg, batch_norm=False, in_channels=3):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        layers = layers[:-1]
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


cfg = {
    'A': [[64], [128], [256, 256], [512, 512], [512, 512]],
    'B': [[64, 64], [128, 128], [256, 256], [512, 512], [512, 512]],
    'D': [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]],
    'E': [[64, 64], [128, 128], [256, 256, 256, 256], [512, 512, 512, 512], [512, 512, 512, 512]],
    'S': [[64], [128], [256], [512], [512]],
}




def myvgg8_bn(**kwargs):
    """VGG 8-layer model (configuration "S")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGGmy(cfg['S'], batch_norm=True, **kwargs)
    return model


def myvgg11_bn(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    model = VGGmy(cfg['A'], batch_norm=True, **kwargs)
    return model

def myvgg13_bn(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = VGGmy(cfg['B'], batch_norm=True, **kwargs)
    return model


def myvgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGGmy(cfg['D'], batch_norm=True, **kwargs)
    return model


def myvgg19_bn(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    model = VGGmy(cfg['E'], batch_norm=True, **kwargs)
    return model


def mymaskvgg8_bn(**kwargs):
    """VGG 8-layer model (configuration "S")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MaskVGGmy(cfg['S'], batch_norm=True, **kwargs)
    return model


def mymaskvgg11_bn(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    model = MaskVGGmy(cfg['A'], batch_norm=True, **kwargs)
    return model

def mymaskvgg13_bn(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = MaskVGGmy(cfg['B'], batch_norm=True, **kwargs)
    return model


def mymaskvgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = MaskVGGmy(cfg['D'], batch_norm=True, **kwargs)
    return model


def mymaskvgg19_bn(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    model = MaskVGGmy(cfg['E'], batch_norm=True, **kwargs)
    return model
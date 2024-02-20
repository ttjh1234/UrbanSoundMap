from __future__ import absolute_import
import torch.nn as nn
import torch.nn.functional as F
import math
import torch



class preserving_spatial(nn.Module):
    def __init__(self):
        super().__init__()
        


class simple_model(nn.Module):
    def __init__(self, cfg, batch_norm=False, num_classes=1000):
        super(simple_model, self).__init__()
        self.block0 = self._make_init_layers(cfg[0], batch_norm, 4, group=4)
        self.block1 = self._make_layers(cfg[1], batch_norm, cfg[0][-1], group=1)
        self.block2 = self._make_layers(cfg[2], batch_norm, cfg[1][-1], group=1)
        self.block3 = self._make_layers(cfg[3], batch_norm, cfg[2][-1], group=1)
        self.block4 = self._make_layers(cfg[4], batch_norm, cfg[3][-1], group=1)

        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))
        #self.pool4 = nn.Conv2d(512, 512, 12, 1)
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.classifier = nn.Linear(512, num_classes)
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

    def forward(self, x, is_feat=False, preact=False, is_grad=False):
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
        if is_grad:
            f4.retain_grad()
        x = self.pool4(f4)
        x = x.view(x.size(0), -1)
        f5 = x
        x = self.classifier(x)
        
        if is_feat:
            if is_grad:
                ind = x.data.max(1)[1]
                grad_out = x.data.clone().fill_(0.0).scatter_(1, ind.unsqueeze(0).t(), 1.0)
                channel_weight = self.cal_grad(x, grad_out, [f4])
            
            if preact:
                if is_grad:
                    return [f0, f1_pre, f2_pre, f3_pre, f4_pre, f5], x, channel_weight
                else:
                    return [f0, f1_pre, f2_pre, f3_pre, f4_pre, f5], x
            else:
                if is_grad:
                    return [f0, f1, f2, f3, f4, f5], x, channel_weight
                else:
                    return [f0, f1, f2, f3, f4, f5], x
        else:
            return x
    
    def cal_grad(self, out, grad_out, feature):
        out.backward(grad_out, retain_graph=True)
        grad = feature[0].grad.clone().detach()

        return grad

    
    @staticmethod
    def _make_layers(cfg, batch_norm=False, in_channels=3, group=1):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, groups=group)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        layers = layers[:-1]
        return nn.Sequential(*layers)

    @staticmethod
    def _make_init_layers(cfg, batch_norm=False, in_channels=3,group=1):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, groups=group)
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
    

   
class GroupVGG(nn.Module):

    def __init__(self, cfg, batch_norm=False, num_classes=1000):
        super(GroupVGG, self).__init__()
        self.block0 = self._make_layers(cfg[0], batch_norm, 4, group=4)
        self.block1 = self._make_layers(cfg[1], batch_norm, cfg[0][-1], group=1)
        self.block2 = self._make_layers(cfg[2], batch_norm, cfg[1][-1], group=1)
        self.block3 = self._make_layers(cfg[3], batch_norm, cfg[2][-1], group=1)
        self.block4 = self._make_layers(cfg[4], batch_norm, cfg[3][-1], group=1)

        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))
        #self.pool4 = nn.Conv2d(512, 512, 12, 1)
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.classifier = nn.Linear(512, num_classes)
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

    def forward(self, x, is_feat=False, preact=False, is_grad=False):
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
        if is_grad:
            f4.retain_grad()
        x = self.pool4(f4)
        x = x.view(x.size(0), -1)
        f5 = x
        x = self.classifier(x)
        
        if is_feat:
            if is_grad:
                ind = x.data.max(1)[1]
                grad_out = x.data.clone().fill_(0.0).scatter_(1, ind.unsqueeze(0).t(), 1.0)
                channel_weight = self.cal_grad(x, grad_out, [f4])
            
            if preact:
                if is_grad:
                    return [f0, f1_pre, f2_pre, f3_pre, f4_pre, f5], x, channel_weight
                else:
                    return [f0, f1_pre, f2_pre, f3_pre, f4_pre, f5], x
            else:
                if is_grad:
                    return [f0, f1, f2, f3, f4, f5], x, channel_weight
                else:
                    return [f0, f1, f2, f3, f4, f5], x
        else:
            return x
    
    def cal_grad(self, out, grad_out, feature):
        out.backward(grad_out, retain_graph=True)
        grad = feature[0].grad.clone().detach()

        return grad

    
    @staticmethod
    def _make_layers(cfg, batch_norm=False, in_channels=3, group=1):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, groups=group)
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

class GroupVGG2(nn.Module):

    def __init__(self, cfg, batch_norm=False, num_classes=1000):
        super(GroupVGG2, self).__init__()
        self.block0 = self._make_layers(cfg[0], batch_norm, 4, group=4)
        self.block1 = self._make_layers(cfg[1], batch_norm, cfg[0][-1], group=4)
        self.block2 = self._make_layers(cfg[2], batch_norm, cfg[1][-1], group=4)
        self.block3 = self._make_layers(cfg[3], batch_norm, cfg[2][-1], group=4)
        self.block4 = self._make_layers(cfg[4], batch_norm, cfg[3][-1], group=4)

        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))
        #self.pool4 = nn.Conv2d(512, 512, 12, 1)
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.classifier = nn.Linear(512, num_classes)
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

    def forward(self, x, is_feat=False, preact=False, is_grad=False):
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
        if is_grad:
            f4.retain_grad()
        x = self.pool4(f4)
        x = x.view(x.size(0), -1)
        f5 = x
        x = self.classifier(x)
        
        if is_feat:
            if is_grad:
                ind = x.data.max(1)[1]
                grad_out = x.data.clone().fill_(0.0).scatter_(1, ind.unsqueeze(0).t(), 1.0)
                channel_weight = self.cal_grad(x, grad_out, [f4])
            
            if preact:
                if is_grad:
                    return [f0, f1_pre, f2_pre, f3_pre, f4_pre, f5], x, channel_weight
                else:
                    return [f0, f1_pre, f2_pre, f3_pre, f4_pre, f5], x
            else:
                if is_grad:
                    return [f0, f1, f2, f3, f4, f5], x, channel_weight
                else:
                    return [f0, f1, f2, f3, f4, f5], x
        else:
            return x
    
    def cal_grad(self, out, grad_out, feature):
        out.backward(grad_out, retain_graph=True)
        grad = feature[0].grad.clone().detach()

        return grad

    
    @staticmethod
    def _make_layers(cfg, batch_norm=False, in_channels=3, group=1):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, groups=group)
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


def vgg8(**kwargs):
    """VGG 8-layer model (configuration "S")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['S'], **kwargs)
    return model


def groupvgg8_bn(**kwargs):
    """VGG 8-layer model (configuration "S")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = GroupVGG(cfg['S'], batch_norm=True, **kwargs)
    return model

def groupvgg8_bn2(**kwargs):
    """VGG 8-layer model (configuration "S")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = GroupVGG2(cfg['S'], batch_norm=True, **kwargs)
    return model


def vgg11(**kwargs):
    """VGG 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['A'], **kwargs)
    return model


def groupvgg11_bn(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    model = GroupVGG(cfg['A'], batch_norm=True, **kwargs)
    return model

def groupvgg11_bn2(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    model = GroupVGG2(cfg['A'], batch_norm=True, **kwargs)
    return model


def vgg13(**kwargs):
    """VGG 13-layer model (configuration "B")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['B'], **kwargs)
    return model


def groupvgg13_bn(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = GroupVGG(cfg['B'], batch_norm=True, **kwargs)
    return model

def groupvgg13_bn2(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = GroupVGG2(cfg['B'], batch_norm=True, **kwargs)
    return model


def vgg16(**kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['D'], **kwargs)
    return model


def groupvgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = GroupVGG(cfg['D'], batch_norm=True, **kwargs)
    return model

def groupvgg16_bn2(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = GroupVGG2(cfg['D'], batch_norm=True, **kwargs)
    return model


def vgg19(**kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['E'], **kwargs)
    return model


def groupvgg19_bn(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    model = GroupVGG(cfg['E'], batch_norm=True, **kwargs)
    return model

def groupvgg19_bn2(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    model = GroupVGG2(cfg['E'], batch_norm=True, **kwargs)
    return model


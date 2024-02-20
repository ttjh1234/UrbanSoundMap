from .resnet import resnet8, resnet14, resnet20, resnet32, resnet44, resnet56, resnet110, resnet8x4, resnet32x4
from .resnetv2 import ResNet18, ResNet50, ResNet34, ResNet101, ResNet152
from .wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2, wrn_28_10
from .vgg import vgg19_bn, vgg16_bn, vgg13_bn, vgg11_bn, vgg8_bn
from .vgg import vgg19_bn2, vgg16_bn2, vgg13_bn2, vgg11_bn2, vgg8_bn2
from .vgg2 import vgg19_fuse2, vgg16_fuse2, vgg13_fuse2, vgg11_fuse2, vgg8_fuse2
from .mobilenetv2 import mobile_half, mobile_light, mobile_heavy
from .mymodel import myvgg8_bn, myvgg11_bn, myvgg13_bn, myvgg16_bn, myvgg19_bn
from .mymodel import mymaskvgg8_bn, mymaskvgg11_bn, mymaskvgg13_bn, mymaskvgg16_bn, mymaskvgg19_bn
from .simple import groupvgg8_bn,groupvgg11_bn,groupvgg13_bn,groupvgg16_bn,groupvgg19_bn
from .simple import groupvgg8_bn2,groupvgg11_bn2,groupvgg13_bn2,groupvgg16_bn2,groupvgg19_bn2
from .simple import simple_model
from .inception_resnetv2 import inception_resnet_v2
from .inception_resnetv2_2 import inception_resnetv2_2


model_dict = {
    'resnet8': resnet8,
    'resnet14': resnet14,
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet44': resnet44,
    'resnet56': resnet56,
    'resnet110': resnet110,
    'resnet8x4': resnet8x4,
    'resnet32x4': resnet32x4,
    'resnet18': ResNet18,
    'resnet34': ResNet34,
    'resnet50': ResNet50,
    'resnet101': ResNet101,
    'resnet152': ResNet152,
    'wrn_16_1': wrn_16_1,
    'wrn_16_2': wrn_16_2,
    'wrn_28_10': wrn_28_10,
    'wrn_40_1': wrn_40_1,
    'wrn_40_2': wrn_40_2,
    'vgg8': vgg8_bn,
    'vgg11': vgg11_bn,
    'vgg13': vgg13_bn,
    'vgg16': vgg16_bn,
    'vgg19': vgg19_bn,
    'vgg8act': vgg8_bn2,
    'vgg11act': vgg11_bn2,
    'vgg13act': vgg13_bn2,
    'vgg16act': vgg16_bn2,
    'vgg19act': vgg19_bn2,
    'vgg8fuse2': vgg8_fuse2,
    'vgg11fuse2': vgg11_fuse2,
    'vgg13fuse2': vgg13_fuse2,
    'vgg16fuse2': vgg16_fuse2,
    'vgg19fuse2': vgg19_fuse2,
    'myvgg8': myvgg8_bn,
    'myvgg11': myvgg11_bn,
    'myvgg13': myvgg13_bn,
    'myvgg16': myvgg16_bn,
    'myvgg19': myvgg19_bn,
    'mymaskvgg8': mymaskvgg8_bn,
    'mymaskvgg11': mymaskvgg11_bn,
    'mymaskvgg13': mymaskvgg13_bn,
    'mymaskvgg16': mymaskvgg16_bn,
    'mymaskvgg19': mymaskvgg19_bn,
    'groupvgg8': groupvgg8_bn,
    'groupvgg11': groupvgg11_bn,
    'groupvgg13': groupvgg13_bn,
    'groupvgg16': groupvgg16_bn,
    'groupvgg19': groupvgg19_bn,
    'groupvgg8_2': groupvgg8_bn2,
    'groupvgg11_2': groupvgg11_bn2,
    'groupvgg13_2': groupvgg13_bn2,
    'groupvgg16_2': groupvgg16_bn2,
    'groupvgg19_2': groupvgg19_bn2,
    'MobileNetV2': mobile_half,
    'simple' : simple_model,
    'MobileNetV2_0.35' : mobile_light,
    'MobileNetV2_1.0' : mobile_heavy,
    'inception_resnetv2' : inception_resnet_v2,
    'inception_resnetv2_2' : inception_resnetv2_2
}
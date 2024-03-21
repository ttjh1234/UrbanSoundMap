from .resnet import resnet8, resnet14, resnet20, resnet32, resnet44, resnet56, resnet110, resnet8x4, resnet32x4
from .resnetv2 import ResNet18, ResNet50, ResNet34, ResNet101, ResNet152
from .wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2, wrn_28_10
from .vgg import vgg19_bn, vgg16_bn, vgg13_bn, vgg11_bn, vgg8_bn
from .mobilenetv2 import mobile_half, mobile_light, mobile_heavy
from .simple import image_classification
from .inception_resnetv2 import inception_resnet_v2
from .inception_resnetv2_2 import inception_resnetv2_2
from .mymodel import myvgg8_bn,myvgg11_bn,myvgg13_bn,myvgg16_bn,myvgg19_bn

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
    'myvgg8': myvgg8_bn,
    'myvgg11': myvgg11_bn,
    'myvgg13': myvgg13_bn,
    'myvgg16': myvgg16_bn,
    'myvgg19': myvgg19_bn,
    'MobileNetV2': mobile_half,
    'simple' : image_classification,
    'MobileNetV2_0.35' : mobile_light,
    'MobileNetV2_1.0' : mobile_heavy,
    'inception_resnetv2' : inception_resnet_v2,
    'inception_resnetv2_2' : inception_resnetv2_2
}
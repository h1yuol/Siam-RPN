import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
import pdb

from siamrpn.SRPN import SiameseRPN

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
       'resnet152']

model_urls = {
  'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
  'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
  'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
  'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
  'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=1, bias=False)


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None, num_groups=32, GN=True):
    super(BasicBlock, self).__init__()
    self.GN = GN
    self.num_groups = num_groups

    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.GroupNorm(num_groups, planes) if self.GN else nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.GroupNorm(num_groups, planes) if self.GN else nn.BatchNorm2d(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None, num_groups=32, GN=True):
    super(Bottleneck, self).__init__()
    self.GN = GN
    self.num_groups = num_groups

    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
    self.bn1 = nn.GroupNorm(num_groups, planes) if self.GN else nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                 padding=1, bias=False)
    self.bn2 = nn.GroupNorm(num_groups, planes) if self.GN else nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
    self.bn3 = nn.GroupNorm(num_groups, planes * 4) if self.GN else nn.BatchNorm2d(planes * 4)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class ResNet(nn.Module):
  def __init__(self, block, layers, num_classes=1000, num_groups=32, GN=True):
    self.GN = GN
    self.num_groups = num_groups
    self.inplanes = 64
    super(ResNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                 bias=False)
    self.bn1 = nn.GroupNorm(num_groups, 64) if self.GN else nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
    # it is slightly better whereas slower to set stride = 1
    # self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
    self.avgpool = nn.AvgPool2d(7)
    self.fc = nn.Linear(512 * block.expansion, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      normModule = nn.GroupNorm(self.num_groups, planes * block.expansion) if self.GN else nn.BatchNorm2d(planes * block.expansion)
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
              kernel_size=1, stride=stride, bias=False),
        normModule,
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample, num_groups=self.num_groups,GN=self.GN))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes, num_groups=self.num_groups, GN=self.GN))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)

    return x


def resnet18(pretrained=False, num_groups=32, GN=True):
  """Constructs a ResNet-18 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [2, 2, 2, 2], num_groups=num_groups, GN=GN)
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
  return model


def resnet34(pretrained=False, num_groups=32, GN=True):
  """Constructs a ResNet-34 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [3, 4, 6, 3], num_groups=num_groups, GN=GN)
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
  return model


def resnet50(pretrained=False, num_groups=32, GN=True):
  """Constructs a ResNet-50 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 6, 3], num_groups=num_groups, GN=GN)
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
  return model


def resnet101(pretrained=False, num_groups=32, GN=True):
  """Constructs a ResNet-101 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 23, 3], num_groups=num_groups, GN=GN)
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
  return model


def resnet152(pretrained=False, num_groups=32, GN=True):
  """Constructs a ResNet-152 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 8, 36, 3], num_groups=num_groups, GN=GN)
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
  return model

class resnet(SiameseRPN):
    def __init__(self, num_layers,pseudo=False, num_groups=32, GN=True):
        self.num_layers = num_layers
        self.channel_depth = {
            18: 256,
            34: 256,
            50: 1024,
            101: 1024,
            152: 1024,
        }[num_layers]
        self.num_groups = num_groups
        self.GN = GN
        SiameseRPN.__init__(self,pseudo=pseudo)

    def _build(self):
        resnet = eval('resnet{}'.format(self.num_layers))(num_groups=self.num_groups, GN=self.GN)
        self.features = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu,
                        resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3)
        if self.pseudo:
            # from copy import deep_copy
            # self.features2 = deep_copy(self.features)
            resnet2 = eval('resnet{}'.format(self.num_layers))(num_groups=self.num_groups, GN=self.GN)
            self.features2 = nn.Sequential(resnet2.conv1, resnet2.bn1,resnet2.relu,
                            resnet2.maxpool,resnet2.layer1,resnet2.layer2,resnet2.layer3)

        self.conv1 = nn.Conv2d(self.channel_depth, 2*self.k*self.channel_depth, kernel_size=3)
        # self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.channel_depth, 4*self.k*self.channel_depth, kernel_size=3)
        # self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(self.channel_depth, self.channel_depth, kernel_size=3)
        # self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(self.channel_depth, self.channel_depth, kernel_size=3)
        # self.relu4 = nn.ReLU(inplace=True)

        self.cconv = nn.Conv2d(self.channel_depth, 2* self.k, kernel_size = 4, bias = False)
        self.rconv = nn.Conv2d(self.channel_depth, 4* self.k, kernel_size = 4, bias = False)

    def reset_params(self):
        pretrained_dict = model_zoo.load_url(model_urls['resnet{}'.format(self.num_layers)])
        model_dict = self.features.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.features.load_state_dict(model_dict)
        if self.pseudo:
            self.features2.load_state_dict(model_dict)

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad=False
        self.features.apply(set_bn_fix)
        if self.pseudo:
            self.features2.apply(set_bn_fix)

        self._init_weights()

    def _fix_layers(self):
        fix_layers = [0,1,4]
        for i in fix_layers:
            layer = self.features[i]
            for param in layer.parameters():
                param.requires_grad = False
            if self.pseudo:
                layer = self.features2[i]
                for param in layer.parameters():
                    param.requires_grad = False


if __name__ == '__main__':
    # for size in [224,299]:
    #   print('='*20)
    #   for layers in [18,34,50,101,152]:
    #       print('-'*10)
    #       print('resnet{}'.format(layers))
    #       resnet = eval('resnet{}'.format(layers))()
    #       x = torch.randn(1,3,size,size)
    #       print('Input:',x.size())
    #       for name,m in [
    #           ('conv1+bn1+relu', nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu)),
    #           ('maxpool', resnet.maxpool),
    #           ('layer1',resnet.layer1),
    #           ('layer2',resnet.layer2),
    #           ('layer3',resnet.layer3),
    #           ('layer4',resnet.layer4)]:
    #           x = m(x)
    #           print('{}:'.format(name), x.size())

    # resnet = resnet50()
    # base = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu,
    #                     resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3)
    # for size in range(306,400,17):
    #     x = torch.randn(1,3,size,size)
    #     x = base(x)
    #     print('size:',size,x.size())

    model = resnet(34)
    template = torch.randn(1,3,96,96)
    detection = torch.randn(1,3,340,340)
    coutput, routput = model(template, detection)
    from IPython import embed
    embed()
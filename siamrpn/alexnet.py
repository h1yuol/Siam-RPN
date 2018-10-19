import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from siamrpn.SRPN import SiameseRPN

model_urls = {'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'}

class alexnet(SiameseRPN):
    def __init__(self):
        self.channel_depth = 256
        SiameseRPN.__init__(self)

    def _build(self):
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3),
        )

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
        pretrained_dict = model_zoo.load_url(model_urls['alexnet'])
        model_dict = self.features.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.features.load_state_dict(model_dict)

        self._init_weights()

    def _fix_layers(self):
        fix_layers = [0,3,6]
        for i in fix_layers:
            layer = self.features[i]
            for param in layer.parameters():
                param.requires_grad = False

if __name__ == '__main__':
    model = alexnet()
    template = torch.randn(1,3,127,127)
    detection = torch.randn(1,3,255,255)
    coutput, routput = model(template, detection)
    from IPython import embed
    embed()


import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseRPN(nn.Module):
    def __init__(self,pseudo,cbam,num_anchors):
        super(SiameseRPN, self).__init__()
        self.k = num_anchors
        self.pseudo = pseudo
        self.cbam = cbam
        self._build()
        self.reset_params()
        self._fix_layers()

    def _build(self):
    	raise NotImplementedError	
        
    def reset_params(self):
        raise NotImplementedError

    def _fix_layers(self):
        raise NotImplementedError

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False, bias=True):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                if bias:
                    m.bias.data.zero_()

        # normal_init(self.conv1, 0, 0.001, False)
        # normal_init(self.conv2, 0, 0.001, False)
        # normal_init(self.conv3, 0, 0.001, False)
        # normal_init(self.conv4, 0, 0.001, False)
        # normal_init(self.cconv, 0, 0.001, False, False)
        # normal_init(self.rconv, 0, 0.001, False, False)
        for m in [self.conv1, self.conv2, self.conv3, self.conv4]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out')

        if self.cbam:
            for m in self.cbamLayer.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
            
#     def forward(self, template, detection, debug=False):
#         if self.pseudo:
#             template = self.features2(template)
#         else:
#             template = self.features(template)
#         detection = self.features(detection)
        
#         ckernal = self.conv1(template)
# #        ckernal = self.relu1(ckernal)
#         ckernal = ckernal.view(2* self.k, self.channel_depth, 4, 4)
#         self.cconv.weight = nn.Parameter(ckernal)
#         cinput = self.conv3(detection)
# #        cinput = self.relu3(cinput)
#         coutput = self.cconv(cinput)
        
#         rkernal = self.conv2(template)
# #        rkernal = self.relu2(rkernal)
#         rkernal = rkernal.view(4* self.k, self.channel_depth, 4, 4)
#         self.rconv.weight = nn.Parameter(rkernal)
#         rinput = self.conv4(detection)
# #        rinput = self.relu4(rinput)
#         routput = self.rconv(rinput)
        
#         if debug:
#             return coutput, routput,ckernal,rkernal,self.conv1.weight,template,cinput,detection
#         return coutput, routput

    def forward(self, template, detection, debug=False):
        if self.pseudo:
            template = self.features2(template)
        else:
            template = self.features(template)
        detection = self.features(detection)
        if self.cbam:
            detection = self.cbamLayer(detection)
        
        N = template.size()[0]

        ckernal = self.conv1(template)
        ckernal = ckernal.view(N*2*self.k, self.channel_depth, 4, 4)
        cinput = self.conv3(detection)
        cinput = cinput.view(1, self.channel_depth*N, 20, 20)
        coutput = F.conv2d(cinput, ckernal, bias=None, groups=N)  # shape (1, 2k*N, 17, 17)
        coutput = coutput.view(N, 2*self.k, 17, 17)

        rkernal = self.conv2(template)
        rkernal = rkernal.view(N*4*self.k, self.channel_depth, 4, 4)
        rinput = self.conv4(detection)
        rinput = rinput.view(1, self.channel_depth*N, 20, 20)
        routput = F.conv2d(rinput, rkernal, bias=None, groups=N)  # shape (1, 4k*N, 17, 17)
        routput = routput.view(N, 4*self.k, 17, 17)

        if debug:
            # raise NotImplementedError
            return coutput, routput, template, detection, ckernal, cinput, rkernal, rinput
        return coutput, routput






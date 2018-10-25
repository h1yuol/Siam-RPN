import torch
import torch.nn as nn

class SiameseRPN(nn.Module):
    def __init__(self,pseudo):
        super(SiameseRPN, self).__init__()
        self.k = 5
        self.pseudo = pseudo
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

        normal_init(self.conv1, 0, 0.001, False)
        normal_init(self.conv2, 0, 0.001, False)
        normal_init(self.conv3, 0, 0.001, False)
        normal_init(self.conv4, 0, 0.001, False)
        # normal_init(self.cconv, 0, 0.001, False, False)
        # normal_init(self.rconv, 0, 0.001, False, False)
            
    def forward(self, template, detection, debug=False):
        if self.pseudo:
            template = self.features2(template)
        else:
            template = self.features(template)
        detection = self.features(detection)
        
        ckernal = self.conv1(template)
#        ckernal = self.relu1(ckernal)
        ckernal = ckernal.view(2* self.k, self.channel_depth, 4, 4)
        self.cconv.weight = nn.Parameter(ckernal)
        cinput = self.conv3(detection)
#        cinput = self.relu3(cinput)
        coutput = self.cconv(cinput)
        
        rkernal = self.conv2(template)
#        rkernal = self.relu2(rkernal)
        rkernal = rkernal.view(4* self.k, self.channel_depth, 4, 4)
        self.rconv.weight = nn.Parameter(rkernal)
        rinput = self.conv4(detection)
#        rinput = self.relu4(rinput)
        routput = self.rconv(rinput)
        
        if debug:
            return coutput, routput,ckernal,rkernal,self.conv1.weight,template,cinput,detection
        return coutput, routput
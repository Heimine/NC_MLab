'''VGG11/13/16/19 in Pytorch.'''
"""Modified from https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py"""
import torch
import torch.nn as nn
import torch.nn.functional as F


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, 
                 in_ch=3,
                 ETF_fc=False, 
                 fixdim=False, num_classes=10):
        super(VGG, self).__init__()
        print(f"Preparing network {vgg_name}! \n")
        self.num_classes = num_classes
        self.in_ch = in_ch
        self.features = self._make_layers(cfg[vgg_name], fixdim)
        self.ETF_fc = ETF_fc
        
        if not fixdim:
            self.classifier = nn.Linear(512, self.num_classes)
        else:
            self.classifier = nn.Linear(self.num_classes, self.num_classes)
            
        # ETF fc
        if ETF_fc:
            weight = torch.sqrt(torch.tensor(num_classes/(num_classes-1)))*(torch.eye(num_classes)-(1/num_classes)*torch.ones((num_classes, num_classes)))
            weight /= torch.sqrt((1/num_classes*torch.norm(weight, 'fro')**2))
            if fixdim:
                self.classifier.weight = nn.Parameter(weight)
            else:
                self.classifier.weight = nn.Parameter(torch.mm(weight, torch.eye(num_classes, 512 * block.expansion)))
            self.classifier.weight.requires_grad_(False)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        features = out
        if self.ETF_fc:
            # out = F.normalize(out)
            out = self.classifier(out)
        else:
            out = self.classifier(out)
        return out, features

    def _make_layers(self, cfg, fixdim):
        layers = []
        in_channels = self.in_ch
        for i,x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if i == len(cfg)-2 and fixdim:
                    print("Creating last convolutional layer to have out_dim=num_classes! \n")
                    layers += [nn.Conv2d(in_channels, self.num_classes, kernel_size=3, padding=1),
                               nn.BatchNorm2d(self.num_classes),
                               nn.ReLU(inplace=True)]
                else:
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                               nn.BatchNorm2d(x),
                               nn.ReLU(inplace=True)]
                in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        layers += [nn.AdaptiveAvgPool2d((1,1))]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())
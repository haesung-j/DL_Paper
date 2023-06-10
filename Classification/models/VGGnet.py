import torch
import torch.nn as nn
from torchinfo import summary

configurations = {'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
                  'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
                  'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
                  'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
                 }


class VGGnet(nn.Module):
    def __init__(self,
                 configuration: list,
                 bn: bool,
                 num_classes: int = 1000,
                 init_weights: bool = True,
                 drop_p: float = 0.5):
        super().__init__()

        self.features = self._make_layers(configuration, bn)
        self.avg_pool = nn.AdaptiveAvgPool2d((7,7))    # make(force) output size 7x7
        self.classifier = nn.Sequential(nn.Linear(512*7*7, 4096),
                                        nn.ReLU(),
                                        nn.Dropout(drop_p),
                                        nn.Linear(4096, 4096),
                                        nn.ReLU(),
                                        nn.Dropout(drop_p),
                                        nn.Linear(4096, num_classes))

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _make_layers(self, configuration, bn):
        layers = []
        in_channels = 3
        #  'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        for layer in configuration:
            if type(layer) == int:
                if bn:
                    layers += [nn.Conv2d(in_channels, layer, 3, stride=1, padding=1),
                               nn.BatchNorm2d(layer),
                               nn.ReLU()]
                else:
                    layers += [nn.Conv2d(in_channels, layer, 3, stride=1, padding=1),
                               nn.ReLU()]

                in_channels = layer
            elif type(layer) == str:
                layers += [nn.MaxPool2d(2)]

        return nn.Sequential(*layers)


if __name__ == '__main__':
    avgpool = nn.AdaptiveAvgPool2d((7,7))
    # x1 = torch.randn(3,2, 111, 111)
    # x2 = torch.randn(3, 2, 22, 22)
    x3 = torch.randn(3, 2, 1, 1)
    # print(avgpool(x1).shape)
    # print(avgpool(x2).shape)
    print(avgpool(x3).shape)
    print(avgpool(x3))
    # model = VGGnet(configurations['D'], True)
    # summary(model, input_size=(1, 3, 224, 224))

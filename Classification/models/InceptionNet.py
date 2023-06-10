import torch
import torch.nn as nn
from torchinfo import summary


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, **kwargs):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_ch, out_ch, bias=False, **kwargs),
                                   nn.BatchNorm2d(out_ch, eps=1e-3),
                                   nn.ReLU())

    def forward(self, x):
        x = self.block(x)
        return x


class Inception(nn.Module):
    def __init__(self, in_ch, out1_1, out3_3_red, out3_3, out5_5_red, out5_5, out_pool):
        super().__init__()

        self.conv1_1 = ConvBlock(in_ch, out1_1, kernel_size=1)
        self.conv3_3 = nn.Sequential(ConvBlock(in_ch, out3_3_red, kernel_size=1),
                                     ConvBlock(out3_3_red, out3_3, kernel_size=3, padding=1))
        self.conv5_5 = nn.Sequential(ConvBlock(in_ch, out5_5_red, kernel_size=1),
                                     ConvBlock(out5_5_red, out5_5, kernel_size=5, padding=2))
        self.pool = nn.Sequential(nn.MaxPool2d(kernel_size=(3,3), stride=1, padding=1),
                                  ConvBlock(in_ch, out_pool, kernel_size=1))

    def forward(self, x):
        x1 = self.conv1_1(x)
        x2 = self.conv3_3(x)
        x3 = self.conv5_5(x)
        x4 = self.pool(x)
        out = torch.cat([x1, x2, x3, x4], dim=1)
        return out


class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_ch, num_classes=1000, dropout=0.7):
        super().__init__()
        self.conv = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(4,4)),
                                  ConvBlock(in_ch, 128, kernel_size=1))
        self.classifier = nn.Sequential(nn.Linear(128*4*4, 1024),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(1024, num_classes))

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


class InceptionNet(nn.Module):
    def __init__(self, init_weight=True, use_auxiliary=True, aux_dropout=0.7, dropout=0.4, num_classes=1000):
        super().__init__()
        self.init_weight = init_weight
        self.use_auxiliary = use_auxiliary
        self.stem = nn.Sequential(ConvBlock(3, 64, kernel_size=7, stride=2, padding=3),
                                  nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                  ConvBlock(64, 64, kernel_size=1),
                                  ConvBlock(64, 192, kernel_size=3, stride=1, padding=1),
                                  nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.inception_3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception_3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception_4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception_4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception_4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception_4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception_4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception_5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception_5b = Inception(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(1024, num_classes)

        if self.use_auxiliary:
            self.aux1 = AuxiliaryClassifier(512, num_classes, aux_dropout)
            self.aux2 = AuxiliaryClassifier(528, num_classes, aux_dropout)

        if self.init_weight:
            self._init_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.maxpool1(x)
        x = self.inception_4a(x)
        if self.use_auxiliary and self.training:
            output1 = self.aux1(x)
        else:
            output1 = None
        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)
        if self.use_auxiliary and self.training:
            output2 = self.aux2(x)
        else:
            output2 = None
        x = self.inception_4e(x)
        x = self.maxpool2(x)
        x = self.inception_5a(x)
        x = self.inception_5b(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        if self.use_auxiliary and self.training:
            return x, output1, output2
        else:
            return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    m = InceptionNet(use_auxiliary=True)
    # x = torch.randn(1, 3, 224, 224)
    summary(m, input_size=(1, 3, 224, 224))
    # print(m.__class__.__name__ == 'InceptionNet')
    # print(m(x).size())

import torch
import torch.nn as nn
from torchinfo import summary

class ResNet(nn.Module):
    def __init__(self, block, block_list: list, num_classes: int = 1000, zero_init_residual: bool = True):
        """
        Args:
            block: ResNet 깊이에 따라 BasicBlock 혹은 BottleNeckBlock
            block_list: conv layer 개수를 담은 리스트
            num_classes: output dimension
            zero_init_residual: 각 block의 마지막 BN층을 0으로 초기화하여 초반 학습 시 앞쪽 layer 훈련 집중
        """
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2 = self._make_layers(block, 64, block_list[0], stride=1)
        self.conv3 = self._make_layers(block, 128, block_list[1],
                                       stride=2)  # 각 layer 첫번째 block의 1x1은 stride=2로 downsampling
        self.conv4 = self._make_layers(block, 256, block_list[2], stride=2)
        self.conv5 = self._make_layers(block, 512, block_list[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, block):
                    nn.init.constant_(m.residual[-1].weight, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

    def _make_layers(self, block, inner_channels, num_blocks, stride=1):
        if stride != 1 or self.in_channels != inner_channels * block.expansion:
            # stride가 1이어도, conv2 layer의 첫번째 BottleNeckBlock의 채널 수를 맞춰줘야 함
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, inner_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(inner_channels * block.expansion)
            )
        else:
            downsample = None

        layers = []
        layers += [block(self.in_channels, inner_channels, stride, downsample)]
        self.in_channels = inner_channels * block.expansion
        for _ in range(1, num_blocks):
            layers += [block(self.in_channels, inner_channels)]  # layer의 첫번째 block이 아니면 downsample은 없음

        return nn.Sequential(*layers)


class BasicBlock(nn.Module):
    expansion = 1  # Basic or BottleNeck 구분 짓는 용도 및 out_channel 조절

    def __init__(self, in_channels, inner_channels, stride=1, downsample=None):
        # downsample: 채널 수 조절 혹은 이미지 사이즈 조절 시 사용
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, inner_channels * self.expansion, kernel_size=3, stride=stride, padding=1,
                      bias=False),
            nn.BatchNorm2d(inner_channels * self.expansion)
        )
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.residual(x)

        if self.downsample is not None:
            shortcut = self.downsample(x)
        else:
            shortcut = x

        out = residual + shortcut
        return self.relu(out)


class BottleNeckBlock(nn.Module):
    expansion = 4  # Basic or BottleNeck 구분 짓는 용도 및 out_channel 조절

    def __init__(self, in_channels, inner_channels, stride=1, downsample=None):
        # downsample: 채널 수 조절 혹은 이미지 사이즈 조절 시 사용
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, inner_channels * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channels * self.expansion)
        )
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.residual(x)

        if self.downsample is not None:
            shortcut = self.downsample(x)
        else:
            shortcut = x

        out = residual + shortcut
        return self.relu(out)


if __name__ == '__main__':
    model = ResNet(BottleNeckBlock, [3, 4, 6, 3])
    summary(model, input_size=(2, 3, 224, 224))
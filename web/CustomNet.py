import mindspore.numpy as mnp
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import constexpr
from mindspore import Tensor, Parameter
import mindspore.common.initializer as initializer
import mindcv

class CustomNet(nn.Cell):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.feature = mindcv.create_model('resnet34', pretrained=True)

        self.output_new = nn.SequentialCell([
            nn.Dense(1000, 512),
            nn.ReLU(),
            nn.Dropout(p=0.8),
            nn.Dense(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.8),
            nn.Dense(256, 120)
        ])

        for name, cell in self.output_new.cells_and_names():
            if isinstance(cell, nn.Dense):
                k = 1 / cell.in_channels
                k = k ** 0.5

                cell.weight.set_data(
                    initializer.initializer(initializer.Uniform(k), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(
                        initializer.initializer(initializer.Uniform(k), cell.bias.shape, cell.bias.dtype))

    def construct(self, x):
        x = self.feature(x)
        x = self.output_new(x)
        return x

def get_net(devices):
    net = CustomNet()

    # 冻结参数
    for param in net.feature.get_parameters():
        param.requires_grad = False

    return net

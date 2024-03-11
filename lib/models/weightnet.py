from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.torch_api.nn_api.nn as x2ms_nn

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


class WeightNet(nn.Cell):
    def __init__(self, cfg, **kwargs):
        self.N = cfg.WEIGHT.NUM_JOINTS       

        super(WeightNet, self).__init__()
        self.conv1 = x2ms_nn.Conv2d(2*self.N, 2*self.N, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = x2ms_nn.BatchNorm2d(2*self.N, momentum=BN_MOMENTUM)
        self.relu = x2ms_nn.ReLU(inplace=True)
        self.conv2 = x2ms_nn.Conv2d(2*self.N, 2*self.N, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = x2ms_nn.BatchNorm2d(2*self.N, momentum=BN_MOMENTUM)
        self.avgpool = x2ms_nn.AdaptiveAvgPool2d((1))

        self.fc1 = x2ms_nn.Linear(5*self.N, 3*self.N)
        self.fc2 = x2ms_nn.Linear(3*self.N, 3)


    def construct(self, l1, l2, l3, s, t):
        x = x2ms_adapter.cat((s,t),dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.avgpool(x)

        x = x2ms_adapter.cat((x2ms_adapter.tensor_api.squeeze(x2ms_adapter.tensor_api.squeeze(x, -1), -1), l1, l2, l3),dim=1)

        x = self.fc1(x)
        x = self.fc2(x)

        out = x2ms_adapter.exp(x)
        return out

    def init_weights(self):
        logger.info('=>Weightnet init weights from normal distribution')
        for m in x2ms_adapter.nn_cell.modules(self):
            if isinstance(m, x2ms_nn.Conv2d):
                # nn.init.constant_(m.weight, 0)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                x2ms_adapter.nn_init.normal_(m.weight, std=0.001)
                # nn.init.constant_(m.bias, 0)
            elif isinstance(m, x2ms_nn.BatchNorm2d):
                x2ms_adapter.nn_init.constant_(m.weight, 1)
                x2ms_adapter.nn_init.constant_(m.bias, 0)
            elif isinstance(m, x2ms_nn.Linear):
                x2ms_adapter.nn_init.constant_(m.weight, 0)
                x2ms_adapter.nn_init.constant_(m.bias, 0)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.normal_(m.weight, std=0.001)

def get_weight_net(cfg, **kwargs):

    model = WeightNet(cfg, **kwargs)
    model.init_weights()

    return model

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .utils import channel_shuffle, load_checkpoint
import mindspore
import mindspore.nn as nn
# import x2ms_adapter
# import x2ms_adapter.torch_api.nn_api.nn as x2ms_nn
import x2ms_adapter.torch_api.torch_utils as x2ms_torch_util
# from x2ms_adapter.auto_static import auto_static
import copy
import logging
# from mmcv.cnn import ConvModule, constant_init, normal_init
class ConvModule(nn.Cell):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 kernel_size,
                 stride=1, 
                 padding=0,
                 groups=1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):
        super().__init__()
        if padding>0:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,pad_mode='pad', padding=padding,group=groups)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,group=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act_cfg=act_cfg
        if act_cfg!=None:
            self.relu = nn.ReLU()
        
    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act_cfg!=None:
            x = self.relu(x)
        return x



BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

class InvertedResidual(nn.Cell):
    """InvertedResidual block for ShuffleNetV2 backbone.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 with_cp=False):
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        act_cfg = copy.deepcopy(act_cfg)
        super().__init__()
        self.stride = stride
        self.with_cp = with_cp

        branch_features = out_channels // 2
        if self.stride == 1:
            assert in_channels == branch_features * 2, (
                f'in_channels ({in_channels}) should equal to '
                f'branch_features * 2 ({branch_features * 2}) '
                'when stride is 1')

        if in_channels != branch_features * 2:
            assert self.stride != 1, (
                f'stride ({self.stride}) should not equal 1 when '
                f'in_channels != branch_features * 2')

        if self.stride > 1:
            self.branch1 = nn.SequentialCell(
                ConvModule(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=self.stride,
                    padding=1,
                    groups=in_channels,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=None),
                ConvModule(
                    in_channels,
                    branch_features,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg),
            )

        self.branch2 = nn.SequentialCell(
            ConvModule(
                in_channels if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                branch_features,
                branch_features,
                kernel_size=3,
                stride=self.stride,
                padding=1,
                groups=branch_features,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None),
            ConvModule(
                branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))

    def construct(self, x):

        def _inner_forward(x):
            if self.stride > 1:
                t1=self.branch1(x)
                t2=self.branch2(x)
                out = mindspore.ops.cat((t1, t2), axis=1)
            else:
                x1, x2 = x.chunk(2, axis=1)
                out = mindspore.ops.cat((x1, self.branch2(x2)), axis=1)

            out = channel_shuffle(out, 2)

            return out

        if self.with_cp and x.requires_grad:
            out = x2ms_torch_util.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out


class ShuffleNetV2(nn.Cell):
    """ShuffleNetV2 backbone.

    Args:
        widen_factor (float): Width multiplier - adjusts the number of
            channels in each layer by this amount. Default: 1.0.
        out_indices (Sequence[int]): Output from which stages.
            Default: (0, 1, 2, 3).
        frozen_stages (int): Stages to be frozen (all param fixed).
            Default: -1, which means not freezing any parameters.
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 cfg,
                 widen_factor=1.0,
                 out_indices=(3, ),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 norm_eval=False,
                 with_cp=False):
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        act_cfg = copy.deepcopy(act_cfg)
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS

        super().__init__()
        self.stage_blocks = [4, 8, 4]
        for index in out_indices:
            if index not in range(0, 4):
                raise ValueError('the item in out_indices must in '
                                 f'range(0, 4). But received {index}')

        if frozen_stages not in range(-1, 4):
            raise ValueError('frozen_stages must be in range(-1, 4). '
                             f'But received {frozen_stages}')
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        if widen_factor == 0.5:
            channels = [48, 96, 192, 1024]
            self.inplanes = 1024
        elif widen_factor == 1.0:
            channels = [116, 232, 464, 1024]
            self.inplanes = 1024
        elif widen_factor == 1.5:
            channels = [176, 352, 704, 1024]
            self.inplanes = 1024
        elif widen_factor == 2.0:
            channels = [244, 488, 976, 2048]
            self.inplanes = 2048
        else:
            raise ValueError('widen_factor must be in [0.5, 1.0, 1.5, 2.0]. '
                             f'But received {widen_factor}')

        self.in_channels = 24
        self.conv1 = ConvModule(
            in_channels=3,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1,pad_mode='pad')

        self.layers = nn.CellList()
        for i, num_blocks in enumerate(self.stage_blocks):
            layer = self._make_layer(channels[i], num_blocks)
            self.layers.append(layer)

        output_channels = channels[-1]
        self.layers.append(
            ConvModule(
                in_channels=self.in_channels,
                out_channels=output_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))
        
        # self.inplanes = 1024
        self.deconv_layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS,
            extra.NUM_DECONV_FILTERS,
            extra.NUM_DECONV_KERNELS,
        )

        self.final_layer = nn.Conv2d(
            in_channels=extra.NUM_DECONV_FILTERS[-1],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0,
            has_bias=True
        )
    
    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.Conv2dTranspose(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    pad_mode='pad',
                    padding=padding,
                    has_bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=1-BN_MOMENTUM))
            layers.append(nn.ReLU())
            self.inplanes = planes

        return nn.SequentialCell(*layers)

    def _make_layer(self, out_channels, num_blocks):
        """Stack blocks to make a layer.

        Args:
            out_channels (int): out_channels of the block.
            num_blocks (int): number of blocks.
        """
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            layers.append(
                InvertedResidual(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    with_cp=self.with_cp))
            self.in_channels = out_channels

        return nn.SequentialCell(*layers)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.conv1.get_parameters():
                param.requires_grad = False

        for i in range(self.frozen_stages):
            m = self.layers[i]
            m.set_train(False)
            for param in m.get_parameters():
                param.requires_grad = False

    # def init_weights(self, pretrained=None):
    #     if isinstance(pretrained, str):
    #         load_checkpoint(self, pretrained, strict=False)
    #         logger.info('=> init weights from {}'.format(pretrained))
    #     elif pretrained is None:
    #         for name, m in x2ms_adapter.named_modules(self):
    #             if isinstance(m, x2ms_nn.Conv2d):
    #                 if 'conv1' in name:
    #                     # normal_init(m, mean=0, std=0.01)
    #                 else:
    #                     # normal_init(m, mean=0, std=1.0 / m.weight.shape[1])
    #             elif isinstance(m, (x2ms_nn._BatchNorm, x2ms_nn.GroupNorm)):
    #                 # constant_init(m.weight, val=1, bias=0.0001)
    #                 if isinstance(m, x2ms_nn._BatchNorm):
    #                     if m.running_mean is not None:
    #                         x2ms_adapter.nn_init.constant_(m.running_mean, 0)
    #     else:
    #         logger.info('=> init weights from normal distribution')
    #         for m in x2ms_adapter.nn_cell.modules(self):
    #             if isinstance(m, x2ms_nn.Conv2d):
    #                 # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #                 x2ms_adapter.nn_init.normal_(m.weight, std=0.001)
    #                 # nn.init.constant_(m.bias, 0)
    #             elif isinstance(m, x2ms_nn.BatchNorm2d):
    #                 x2ms_adapter.nn_init.constant_(m.weight, 1)
    #                 x2ms_adapter.nn_init.constant_(m.bias, 0)
    #             elif isinstance(m, x2ms_nn.ConvTranspose2d):
    #                 x2ms_adapter.nn_init.normal_(m.weight, std=0.001)
    #                 if self.deconv_with_bias:
    #                     x2ms_adapter.nn_init.constant_(m.bias, 0)

    def construct(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        for i, layer in enumerate(self.layers):
            x = layer(x)
        
        x = self.deconv_layers(x)
        x = self.final_layer(x)

        return x


def get_pose_net(cfg, is_train, **kwargs):
    
    model = ShuffleNetV2(cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model

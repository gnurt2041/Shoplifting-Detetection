import torch.nn as nn
import numpy as np
import torch
import importlib.util
import math

def build_model(config: dict) -> 'Recognizer3D':
    """Build a model dynamically from a config dictionary."""
    model_type = config.get('type', None)
    backbone_cfg = config.get('backbone', None)
    cls_head_cfg = config.get('cls_head', None)
    test_cfg = config.get('test_cfg', None)

    if model_type == 'Recognizer3D':
        backbone = build_backbone(backbone_cfg)
        cls_head = build_head(cls_head_cfg)
        model = Recognizer3D(backbone=backbone, cls_head=cls_head, test_cfg=test_cfg)
        return model
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def build_backbone(backbone_cfg: dict) -> 'nn.Module':
    """Build the backbone based on the config."""
    backbone_type = backbone_cfg.get('type')

    if backbone_type == 'C3D':
        return C3D(
            in_channels=backbone_cfg['in_channels'],
            base_channels=backbone_cfg['base_channels'],
            num_stages=backbone_cfg['num_stages'],
            temporal_downsample=backbone_cfg['temporal_downsample']
        )

    elif backbone_type == 'X3D':
        return X3D(
            gamma_d=backbone_cfg.get('gamma_d', 1),
            in_channels=backbone_cfg.get('in_channels', 17),
            base_channels=backbone_cfg.get('base_channels', 24),
            num_stages=backbone_cfg.get('num_stages', 3),
            se_ratio=backbone_cfg.get('se_ratio', None),
            use_swish=backbone_cfg.get('use_swish', False),
            stage_blocks=backbone_cfg.get('stage_blocks', (2, 5, 3)),
            spatial_strides=backbone_cfg.get('spatial_strides', (2, 2, 2))
        )

    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")


def build_head(cls_head_cfg: dict) -> 'nn.Module':
    """Build the classification head based on the config."""
    head_type = cls_head_cfg.get('type')

    if head_type == 'I3DHead':
        return I3DHead(
            num_classes=cls_head_cfg['num_classes'],
            in_channels=cls_head_cfg['in_channels'],
            dropout=cls_head_cfg['dropout']
        )

    else:
        raise ValueError(f"Unknown head type: {head_type}")

def build_optimizer_and_lr_scheduler(model, config):
    # Extract parameters from config
    optimizer_config = config.optimizer
    optimizer_type = optimizer_config.get('type', 'SGD')  # Default to SGD
    lr = optimizer_config.get('lr', 0.02)
    momentum = optimizer_config.get('momentum', 0.9)
    weight_decay = optimizer_config.get('weight_decay', 0.0003)

    # Build the optimizer based on its type
    if optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    # Define the learning rate scheduler
    lr_schedule_config = config.lr_config
    total_epochs = lr_schedule_config.get('total_epochs', 24)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=0)

    return optimizer, lr_scheduler

def load_config(config_path):
    """Load a config from a Python file."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


class SimpleHead(nn.Module):
    """ A simplified classification head.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input features.
        dropout (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for weight initialization. Default: 0.01.
        mode (str): Determines the pooling mode ('3D', 'GCN', or '2D').
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 dropout=0.5,
                 init_std=0.01,
                 mode='3D'):
        super(SimpleHead, self).__init__()

        self.dropout_ratio = dropout
        self.init_std = init_std
        self.mode = mode

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        self.in_c = in_channels
        self.fc_cls = nn.Linear(self.in_c, num_classes)

        # Initialize weights of the final fully connected layer
        self.init_weights()

    def init_weights(self):
        """Initialize the weights of the classification layer."""
        nn.init.normal_(self.fc_cls.weight, 0, self.init_std)
        if self.fc_cls.bias is not None:
            nn.init.constant_(self.fc_cls.bias, 0)

    def forward(self, x):
        """Defines the forward computation.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: Classification scores.
        """
        if isinstance(x, list):
            # Case where x is a list of tensors, reduce each to its mean along dim 0
            x = [item.mean(dim=0) for item in x]
            x = torch.stack(x)

        if len(x.shape) != 2:
            # 2D, 3D, or GCN mode with different types of pooling
            if self.mode == '2D':
                N, S, C, H, W = x.shape
                x = x.view(N * S, C, H, W)
                x = nn.AdaptiveAvgPool2d(1)(x)
                x = x.view(N, S, C).mean(dim=1)
            elif self.mode == '3D':
                x = nn.AdaptiveAvgPool3d(1)(x)
                x = x.view(x.shape[:2])  # Collapse spatial dimensions to get (N, C)
            elif self.mode == 'GCN':
                N, M, C, T, V = x.shape
                x = x.view(N * M, C, T, V)
                x = nn.AdaptiveAvgPool2d(1)(x)
                x = x.view(N, M, C).mean(dim=1)

        if self.dropout is not None:
            x = self.dropout(x)

        # Compute class scores
        cls_score = self.fc_cls(x)
        return cls_score

class I3DHead(SimpleHead):
    """I3D Head inheriting from SimpleHead for 3D mode."""
    def __init__(self, num_classes, in_channels, dropout=0.5, init_std=0.01):
        super(I3DHead, self).__init__(num_classes, in_channels, dropout=dropout, init_std=init_std, mode='3D')

class GCNHead(SimpleHead):
    """GCN Head inheriting from SimpleHead for GCN mode."""
    def __init__(self, num_classes, in_channels, dropout=0., init_std=0.01):
        super(GCNHead, self).__init__(num_classes, in_channels, dropout=dropout, init_std=init_std, mode='GCN')

class TSNHead(SimpleHead):
    """TSN Head inheriting from SimpleHead for 2D mode."""
    def __init__(self, num_classes, in_channels, dropout=0.5, init_std=0.01):
        super(TSNHead, self).__init__(num_classes, in_channels, dropout=dropout, init_std=init_std, mode='2D')

class SlowFastHead(I3DHead):
    """SlowFast Head inheriting from I3DHead."""
    pass

class Recognizer3D(nn.Module):
    def __init__(self, backbone, cls_head, test_cfg=None):
        super(Recognizer3D, self).__init__()
        self.backbone = backbone
        self.cls_head = cls_head
        self.test_cfg = test_cfg

    def forward(self, x):
        x = self.backbone(x)
        x = self.cls_head(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, norm=True, activation=True):
        super(ConvBlock, self).__init__()

        layers = []

        # Convolution layer
        layers.append(
            nn.Conv3d(
                in_channels, out_channels, kernel_size, stride=stride,
                padding=padding, dilation=dilation, groups=groups, bias=bias
            )
        )

        # Optional normalization layer (BatchNorm3d)
        if norm:
            layers.append(nn.BatchNorm3d(out_channels))

        # Optional activation layer (ReLU)
        if activation:
            layers.append(nn.ReLU(inplace=True))

        # Combine layers into a sequential block
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class C3D(nn.Module):
    """C3D backbone, without flatten and mlp.

    Args:
        pretrained (str | None): Name of pretrained model.
    """

    def __init__(self,
                 in_channels=3,
                 base_channels=64,
                 num_stages=4,
                 temporal_downsample=True,
                 pretrained=None):
        super().__init__()

        self.pretrained = pretrained
        self.in_channels = in_channels
        self.base_channels = base_channels
        assert num_stages in [3, 4]
        self.num_stages = num_stages
        self.temporal_downsample = temporal_downsample

        pool_kernel, pool_stride = 2, 2
        if not self.temporal_downsample:
            pool_kernel, pool_stride = (1, 2, 2), (1, 2, 2)

        # C3D convolution parameters
        c3d_conv_param = dict(kernel_size=3, padding=1)

        # Define the convolutional layers using the simplified ConvBlock
        self.conv1a = ConvBlock(self.in_channels, self.base_channels, **c3d_conv_param)
        self.pool1 = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2a = ConvBlock(self.base_channels, self.base_channels * 2, **c3d_conv_param)
        self.pool2 = nn.AvgPool3d(kernel_size=pool_kernel, stride=pool_stride)

        self.conv3a = ConvBlock(self.base_channels * 2, self.base_channels * 4, **c3d_conv_param)
        self.conv3b = ConvBlock(self.base_channels * 4, self.base_channels * 4, **c3d_conv_param)
        self.pool3 = nn.AvgPool3d(kernel_size=pool_kernel, stride=pool_stride)

        self.conv4a = ConvBlock(self.base_channels * 4, self.base_channels * 8, **c3d_conv_param)
        self.conv4b = ConvBlock(self.base_channels * 8, self.base_channels * 8, **c3d_conv_param)

        if self.num_stages == 4:
            self.pool4 = nn.AvgPool3d(kernel_size=pool_kernel, stride=pool_stride)
            self.conv5a = ConvBlock(self.base_channels * 8, self.base_channels * 8, **c3d_conv_param)
            self.conv5b = ConvBlock(self.base_channels * 8, self.base_channels * 8, **c3d_conv_param)

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from scratch."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data. The size of x is (num_batches, 3, 16, 112, 112).

        Returns:
            torch.Tensor: The feature of the input samples extracted by the backbone.
        """
        x = self.conv1a(x)
        x = self.pool1(x)

        x = self.conv2a(x)
        x = self.pool2(x)

        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.pool3(x)

        x = self.conv4a(x)
        x = self.conv4b(x)

        if self.num_stages == 3:
            return x

        x = self.pool4(x)
        x = self.conv5a(x)
        x = self.conv5b(x)

        return x





class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.bottleneck = self._round_width(channels, reduction)
        self.fc1 = nn.Conv3d(channels, self.bottleneck, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(self.bottleneck, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def _round_width(width, multiplier, min_width=8, divisor=8):
        width *= multiplier
        width_out = max(min_width, int(width + divisor / 2) // divisor * divisor)
        if width_out < 0.9 * width:
            width_out += divisor
        return int(width_out)

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class BlockX3D(nn.Module):
    """BlockX3D 3D building block for X3D.

    Args:
        inplanes (int): Number of channels for the input in the first ConvBlock.
        planes (int): Number of channels produced by intermediate ConvBlock layers.
        outplanes (int): Number of channels produced by the final ConvBlock layer.
        spatial_stride (int): Spatial stride in the ConvBlock layer. Default: 1.
        downsample (nn.Module | None): Downsample layer. Default: None.
        se_ratio (float | None): The reduction ratio of squeeze and excitation unit.
            If set as None, it means not using SE unit. Default: None.
        use_swish (bool): Whether to use Swish as the activation function
            before and after the 3x3x3 conv. Default: True.
    """

    def __init__(self,
                 inplanes,
                 planes,
                 outplanes,
                 spatial_stride=1,
                 downsample=None,
                 se_ratio=None,
                 use_swish=True):
        super().__init__()

        self.inplanes = inplanes
        self.planes = planes
        self.outplanes = outplanes
        self.spatial_stride = spatial_stride
        self.downsample = downsample
        self.se_ratio = se_ratio
        self.use_swish = use_swish

        # First ConvBlock (1x1x1 Conv)
        self.conv1 = ConvBlock(
            in_channels=inplanes,
            out_channels=planes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            norm=True,
            activation=True)  # ReLU activation

        # Second ConvBlock (3x3x3 Depthwise Conv)
        self.conv2 = ConvBlock(
            in_channels=planes,
            out_channels=planes,
            kernel_size=3,
            stride=(1, self.spatial_stride, self.spatial_stride),
            padding=1,
            groups=planes,  # Depthwise convolution
            bias=False,
            norm=True,
            activation=False)  # No activation here, as Swish is applied later

        # Swish activation (or Identity if not used)
        self.swish = Swish() if self.use_swish else nn.Identity()

        # Third ConvBlock (1x1x1 Conv, no activation)
        self.conv3 = ConvBlock(
            in_channels=planes,
            out_channels=outplanes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            norm=True,
            activation=False)  # No activation after this layer

        # Squeeze-and-Excitation (SE) module if se_ratio is provided
        if self.se_ratio is not None:
            self.se_module = SEModule(planes, self.se_ratio)

        # Final ReLU activation
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Defines the computation performed at every call."""

        identity = x

        # First ConvBlock
        out = self.conv1(x)

        # Second ConvBlock (Depthwise Conv)
        out = self.conv2(out)

        # Apply SE module if defined
        if self.se_ratio is not None:
            out = self.se_module(out)

        # Apply Swish (or Identity if Swish is not used)
        out = self.swish(out)

        # Third ConvBlock
        out = self.conv3(out)

        # Apply downsampling if defined
        if self.downsample is not None:
            identity = self.downsample(x)

        # Add the residual (identity) connection
        out += identity

        # Final ReLU activation
        out = self.relu(out)

        return out

class X3D(nn.Module):
    def __init__(self,
                 gamma_w=1.0,
                 gamma_b=2.25,
                 gamma_d=2.2,
                 pretrained=None,
                 in_channels=3,
                 base_channels=24,
                 num_stages=4,
                 stage_blocks=(1, 2, 5, 3),
                 spatial_strides=(2, 2, 2, 2),
                 frozen_stages=-1,
                 se_style='half',
                 se_ratio=1 / 16,
                 use_swish=True,
                 norm_eval=False,
                 zero_init_residual=True,
                 **kwargs):
        super().__init__()
        self.gamma_w = gamma_w
        self.gamma_b = gamma_b
        self.gamma_d = gamma_d
        self.pretrained = pretrained
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.stage_blocks = stage_blocks

        # Apply parameters gamma_w and gamma_d
        self.base_channels = self._round_width(self.base_channels, self.gamma_w)
        self.stage_blocks = [self._round_repeats(x, self.gamma_d) for x in self.stage_blocks]

        self.num_stages = num_stages
        assert 1 <= num_stages <= 4
        self.spatial_strides = spatial_strides
        assert len(spatial_strides) == num_stages
        self.frozen_stages = frozen_stages
        self.se_style = se_style
        self.se_ratio = se_ratio
        self.use_swish = use_swish
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual

        self.block = BlockX3D
        self.stage_blocks = self.stage_blocks[:num_stages]
        self.layer_inplanes = self.base_channels
        self._make_stem_layer()

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            spatial_stride = spatial_strides[i]
            inplanes = self.base_channels * 2**i
            planes = int(inplanes * self.gamma_b)

            res_layer = self.make_res_layer(
                self.block,
                self.layer_inplanes,
                inplanes,
                planes,
                num_blocks,
                spatial_stride=spatial_stride,
                se_style=self.se_style,
                se_ratio=self.se_ratio,
                use_swish=self.use_swish,
                **kwargs)
            self.layer_inplanes = inplanes
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.feat_dim = self.base_channels * 2**(len(self.stage_blocks) - 1)
        self.conv5 = ConvBlock(
            self.feat_dim,
            int(self.feat_dim * self.gamma_b),
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            norm=True,
            activation=True)
        self.feat_dim = int(self.feat_dim * self.gamma_b)

    @staticmethod
    def _round_width(width, multiplier, min_depth=8, divisor=8):
        if not multiplier:
            return width
        width *= multiplier
        min_depth = min_depth or divisor
        new_filters = max(min_depth, int(width + divisor / 2) // divisor * divisor)
        if new_filters < 0.9 * width:
            new_filters += divisor
        return int(new_filters)

    @staticmethod
    def _round_repeats(repeats, multiplier):
        if not multiplier:
            return repeats
        return int(math.ceil(multiplier * repeats))

    def _make_stem_layer(self):
        self.conv1_s = ConvBlock(
            self.in_channels,
            self.base_channels,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
            bias=False,
            norm=False,
            activation=False)
        self.conv1_t = ConvBlock(
            self.base_channels,
            self.base_channels,
            kernel_size=(5, 1, 1),
            stride=(1, 1, 1),
            padding=(2, 0, 0),
            groups=self.base_channels,
            bias=False,
            norm=True,
            activation=True)

    def make_res_layer(self,
                       block,
                       layer_inplanes,
                       inplanes,
                       planes,
                       blocks,
                       spatial_stride=1,
                       se_style='half',
                       se_ratio=None,
                       use_swish=True,
                       **kwargs):
        downsample = None
        if spatial_stride != 1 or layer_inplanes != inplanes:
            downsample = ConvBlock(
                layer_inplanes,
                inplanes,
                kernel_size=1,
                stride=(1, spatial_stride, spatial_stride),
                padding=0,
                bias=False,
                norm=True,
                activation=False)

        use_se = [False] * blocks
        if self.se_style == 'all':
            use_se = [True] * blocks
        elif self.se_style == 'half':
            use_se = [i % 2 == 0 for i in range(blocks)]
        else:
            raise NotImplementedError

        layers = []
        layers.append(
            block(
                layer_inplanes,
                planes,
                inplanes,
                spatial_stride=spatial_stride,
                downsample=downsample,
                se_ratio=se_ratio if use_se[0] else None,
                use_swish=use_swish,
                **kwargs))

        for i in range(1, blocks):
            layers.append(
                block(
                    inplanes,
                    planes,
                    inplanes,
                    spatial_stride=1,
                    se_ratio=se_ratio if use_se[i] else None,
                    use_swish=use_swish,
                    **kwargs))

        return nn.Sequential(*layers)
    
    def _freeze_stages(self):
        """Prevent all the parameters from being optimized before
        ``self.frozen_stages``."""
        if self.frozen_stages >= 0:
            # Freeze the stem layers (conv1_s and conv1_t)
            self.conv1_s.eval()
            self.conv1_t.eval()
            for param in self.conv1_s.parameters():
                param.requires_grad = False
            for param in self.conv1_t.parameters():
                param.requires_grad = False

        # Freeze the stages based on the frozen_stages parameter
        for i in range(1, self.frozen_stages + 1):
            res_layer = getattr(self, f'layer{i}')
            res_layer.eval()  # Set the layer to evaluation mode
            for param in res_layer.parameters():
                param.requires_grad = False


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, BlockX3D):
                    nn.init.constant_(m.conv3.bn.weight, 0)

        if isinstance(self.pretrained, str):
            self.load_state_dict(torch.load(self.pretrained))

    def forward(self, x):
        x = self.conv1_s(x)
        x = self.conv1_t(x)
        for layer_name in self.res_layers:
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
        x = self.conv5(x)
        return x

    def train(self, mode=True):
        super(X3D, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm3d):
                    m.eval()
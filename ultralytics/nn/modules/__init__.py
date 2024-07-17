# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules. Visualize with:

from ultralytics.nn.modules import *
import torch
import os

x = torch.ones(1, 128, 40, 40)
m = Conv(128, 128)
f = f'{m._get_name()}.onnx'
torch.onnx.export(m, x, f)
os.system(f'onnxsim {f} {f} && open {f}')
"""

from .block import (C1, C2, C3, C3TR, DFL, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x, GhostBottleneck,
                    HGBlock, HGStem, Proto, RepC3, C2f_Bottleneck_ATT, CARAFE, C2f_DCNv3, SimFusion_4in, SimFusion_3in,
                    IFM, InjectionMultiSum_Auto_pool, PyramidPoolAgg, AdvPoolFusion, TopBasicLayer, C2f_iRMB, C2f_Ghost
, SPPFCSPC, GhostMobile, C2f_Sc, C2f_Sc2,C2f_ATT,RepBlock,Pool,C2ELAN, C2f_PKI)
from .conv import (CBAM, ChannelAttention, Concat, Conv, Conv2, ConvTranspose, DWConv, DWConvTranspose2d, Focus,
                   GhostConv, LightConv, RepConv, SpatialAttention, GAM_Attention, MPConv,Conv_ATT,ADown, Cat)
from .head import Classify, Detect, Pose, RTDETRDecoder, Segment
from .transformer import (AIFI, MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer, LayerNorm2d,
                          MLPBlock, MSDeformAttn, TransformerBlock, TransformerEncoderLayer, TransformerLayer)
from .dysample import (DySample)
from .DCNv3 import (DCNv3_pytorch)
from .EVC import UP_EVC
from .AFPN import YOLOv5AFPN
from .AFPN4 import AFPN
from .BiFPN import BiFPN_Add2,BiFPN_Add3
from .fasternet import BasicStage
from .shufflenetv2 import BasicUnit, DownsampleUnit
from .StarNet import Star, ConvBN
__all__ = ('Conv', 'Conv2', 'LightConv', 'RepConv', 'DWConv', 'DWConvTranspose2d', 'ConvTranspose', 'Focus', 'Cat',
           'GhostConv', 'ChannelAttention', 'SpatialAttention', 'CBAM', 'Concat', 'TransformerLayer', 'Star',
           'TransformerBlock', 'MLPBlock', 'LayerNorm2d', 'DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3',
           'C2f', 'C3x', 'C3TR', 'C3Ghost', 'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'Detect',
           'Segment', 'Pose', 'Classify', 'TransformerEncoderLayer', 'RepC3', 'RTDETRDecoder', 'AIFI', 'ConvBN',
           'DeformableTransformerDecoder', 'DeformableTransformerDecoderLayer', 'MSDeformAttn', 'MLP', 'GAM_Attention',
           'DySample', 'DCNv3_pytorch', 'C2f_DCNv3', 'SimFusion_3in','ADown','C2f_PKI', 'BasicStage', 'BasicUnit', 'DownsampleUnit',
           'SimFusion_4in', 'IFM', 'InjectionMultiSum_Auto_pool', 'PyramidPoolAgg', 'AdvPoolFusion', 'TopBasicLayer',
           'C2f_iRMB', 'C2f_Ghost', 'SPPFCSPC', 'UP_EVC', 'GhostMobile', 'C2f_Bottleneck_ATT', 'CARAFE', 'YOLOv5AFPN',
           'AFPN', 'MPConv','C2f_Sc2','C2f_Sc','Conv_ATT', 'RepBlock', 'BiFPN_Add2', 'BiFPN_Add3','Pool','C2ELAN')

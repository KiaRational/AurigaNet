import torchinfo
import torch
from torch import nn
import math

from modules import (ConvBNSiLU, DownsampleConv, BilinearUpsampleConv1x1,
                     UpsampleConvTranspose, BottleNeck, ResidualBottleneckBlock, BasicBlock)

from modules import (CBL, C3, SPPF)


class BackBone(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, resize_module=False):
        super(BackBone, self).__init__()
        self.layer1 = ConvBNSiLU(
            in_channels=in_channels, n_filters=out_channels//16, k_size=7, padding=3, stride=1)
        self.bottleneck1 = BottleNeck(in_channels=out_channels//16, out_channels=out_channels //
                                      8, kernel_size=3, padding=1, stride=1, resize_module=True)
        self.bottleneck2 = BottleNeck(in_channels=out_channels//8, out_channels=out_channels //
                                      4, kernel_size=3, padding=1, stride=1, resize_module=True)
        self.bottleneck3 = BottleNeck(in_channels=out_channels//4, out_channels=out_channels //
                                      2, kernel_size=3, padding=1, stride=1, resize_module=True)
        self.bottleneck4 = BottleNeck(in_channels=out_channels//2, out_channels=out_channels,
                                      kernel_size=3, padding=1, stride=1, resize_module=True)

        # added by sonOfAnton for object detection
        self.bottleneck5 = BottleNeck(in_channels=out_channels, out_channels=out_channels*2,
                                      kernel_size=3, padding=1, stride=1, resize_module=True)

    def forward(self, x):
        x = self.layer1(x)
        Half = self.bottleneck1(x)
        Quarter = self.bottleneck2(Half)
        Octant = self.bottleneck3(Quarter)
        One_sixteenth = self.bottleneck4(Octant)
        # added by sonOfAnton for object detection
        One_thirtysecond = self.bottleneck5(One_sixteenth)

        return Half, Quarter, Octant, One_sixteenth, One_thirtysecond


class SegNeck(nn.Module):
    def __init__(self):
        super(SegNeck, self).__init__()
        # BasicBlockB_S B is number of branch and S is number of stage
        self.BasicBlock1_1 = BasicBlock(in_channels=64, out_channels=64)
        self.BasicBlock1_2 = BasicBlock(in_channels=64, out_channels=64)
        self.BasicBlock1_3 = BasicBlock(in_channels=64, out_channels=64)
        self.BasicBlock1_4 = BasicBlock(in_channels=64, out_channels=64)
        self.BasicBlock2_2 = BasicBlock(in_channels=128, out_channels=128)
        self.BasicBlock2_3 = BasicBlock(in_channels=128, out_channels=128)
        self.BasicBlock2_4 = BasicBlock(in_channels=128, out_channels=128)
        self.BasicBlock3_3 = BasicBlock(in_channels=256, out_channels=256)
        self.BasicBlock3_4 = BasicBlock(in_channels=256, out_channels=256)
        # self.BasicBlock4_4 = BasicBlock(in_channels = 512 , out_channels = 512)

        # DownsampleB_S_R B is number of branch and S is number of stage and R is ratio size
        self.Downsample1_2_2 = DownsampleConv(
            in_channels=64, out_channels=128, downsample_ratio=2)
        # self.Downsample1_2_2 = DownsampleConv(in_channels = 64 , out_channels = 128 , downsample_ratio=2)
        # self.Downsample2_2_2 = DownsampleConv(in_channels = 128 , out_channels = 256 , downsample_ratio=2)
        self.Downsample1_3_2 = DownsampleConv(
            in_channels=64, out_channels=128, downsample_ratio=2)
        self.Downsample1_3_4 = DownsampleConv(
            in_channels=64, out_channels=256, downsample_ratio=4)
        self.Downsample2_3_2 = DownsampleConv(
            in_channels=128, out_channels=256, downsample_ratio=2)
        # self.Downsample3_3_2 = DownsampleConv(in_channels = 256 , out_channels = 512 , downsample_ratio=2)
        # self.Downsample1_4_2 = DownsampleConv(in_channels = 64 , out_channels = 128 , downsample_ratio=2)
        # self.Downsample1_4_4 = DownsampleConv(in_channels = 64 , out_channels = 256 , downsample_ratio=4)
        # self.Downsample1_4_8 = DownsampleConv(in_channels = 64 , out_channels = 512 , downsample_ratio=8)
        # self.Downsample2_4_2 = DownsampleConv(in_channels = 128 , out_channels = 256 , downsample_ratio=2)
        # self.Downsample2_4_4 = DownsampleConv(in_channels = 128 , out_channels = 512 , downsample_ratio=4)
        # self.Downsample3_4_2 = DownsampleConv(in_channels = 256 , out_channels = 512 , downsample_ratio=2)

        # UpsampleB_S_R B is number of branch and S is number of stage and R is ratio size
        self.Upsample2_2_2 = BilinearUpsampleConv1x1(
            in_channels=128, out_channels=64, upsample_ratio=2)
        self.Upsample2_3_2 = BilinearUpsampleConv1x1(
            in_channels=128, out_channels=64, upsample_ratio=2)
        self.Upsample3_3_2 = BilinearUpsampleConv1x1(
            in_channels=256, out_channels=128, upsample_ratio=2)
        self.Upsample3_3_4 = BilinearUpsampleConv1x1(
            in_channels=256, out_channels=64, upsample_ratio=4)
        # self.Upsample4_4_2 = BilinearUpsampleConv1x1(in_channels = 512 , out_channels= 256 , upsample_ratio=2)
        # self.Upsample4_4_4 = BilinearUpsampleConv1x1(in_channels = 512 , out_channels= 128 , upsample_ratio=4)
        # self.Upsample4_4_8 = BilinearUpsampleConv1x1(in_channels = 512 , out_channels= 64 , upsample_ratio=8)
        # self.Upsample3_4_2 = BilinearUpsampleConv1x1(in_channels = 256 , out_channels= 128 , upsample_ratio=2)
        # self.Upsample3_4_4 = BilinearUpsampleConv1x1(in_channels = 256 , out_channels= 64 , upsample_ratio=4)
        # self.Upsample2_4_2 = BilinearUpsampleConv1x1(in_channels = 128 , out_channels= 64 , upsample_ratio=2)

    def forward(self, Half, Quarter, Octant, One_sixteenth):
        # -----------------------Stage 1------------------------
        Branch1 = self.BasicBlock1_1(Half)

        # transisition

        # -----------------------Stage 2------------------------

        Branch2 = self.BasicBlock2_2(Quarter)

        Branch1_2_2 = self.Downsample1_2_2(Branch1)

        Branch2_1_2 = self.Upsample2_2_2(Branch2)
        # Fuse
        Branch1 += Branch2_1_2
        Branch2 += Branch1_2_2

        # transisition

        # -----------------------Stage 3------------------------

        Branch1 = self.BasicBlock1_3(Branch1)
        Branch2 = self.BasicBlock2_3(Branch2)
        Branch3 = self.BasicBlock3_3(Octant)

        Branch1_2_3 = self.Downsample1_3_2(Branch1)
        Branch1_3_3 = self.Downsample1_3_4(Branch1)
        Branch2_3_3 = self.Downsample2_3_2(Branch2)

        Branch3_2_3 = self.Upsample3_3_2(Branch3)
        Branch3_1_3 = self.Upsample3_3_4(Branch3)
        Branch2_1_3 = self.Upsample2_3_2(Branch2)

        # Fuse
        Branch1 += Branch3_1_3 + Branch2_1_3
        Branch2 += Branch1_2_3 + Branch3_2_3
        Branch3 += Branch1_3_3 + Branch2_3_3

        # -----------------------Stage 4------------------------

        # Branch1 = self.BasicBlock1_4(Branch1)
        # Branch2 = self.BasicBlock2_4(Branch2)
        # Branch3 = self.BasicBlock3_4(Branch3)
        # Branch4 = self.BasicBlock4_4(One_sixteenth)

        # Branch1_2_4 = self.Downsample1_4_2(Branch1)
        # Branch1_3_4 = self.Downsample1_4_4(Branch1)
        # Branch1_4_4 = self.Downsample1_4_8(Branch1)
        # Branch2_3_4 = self.Downsample2_4_2(Branch2)
        # Branch2_4_4 = self.Downsample2_4_4(Branch2)
        # Branch3_4_4 = self.Downsample3_4_2(Branch3)
        # Branch4_3_4 = self.Upsample4_4_2(Branch4)
        # Branch4_2_4 = self.Upsample4_4_4(Branch4)
        # Branch4_1_4 = self.Upsample4_4_8(Branch4)
        # Branch3_2_4 = self.Upsample3_4_2(Branch3)
        # Branch3_1_4 = self.Upsample3_4_4(Branch3)
        # Branch2_1_4 = self.Upsample2_4_2(Branch2)
        # print(Branch1_4_4.shape)
        # Branch1 += Branch3_1_4 + Branch2_1_4 + Branch4_1_4
        # Branch2 += Branch4_2_4 + Branch3_2_4 + Branch1_2_4
        # Branch3 += Branch1_3_4 + Branch2_3_4 + Branch4_3_4
        # Branch4 += Branch1_4_4 + Branch2_4_4 + Branch3_4_4

        return Branch1, Branch2, Branch3  # ,Branch4


class Yolov5(nn.Module):
    def __init__(self, first_out):
        super(Yolov5, self).__init__()

        # create the neck of the model
        # self.backbone = nn.ModuleList()
        # self.backbone = BackBone(in_channels=3, out_channels=512)

        # the backbone for yolo model
        # self.backbone += [
        #     CBL(in_channels=3, out_channels=first_out,
        #         kernel_size=6, stride=2, padding=2),
        #     CBL(in_channels=first_out, out_channels=first_out *
        #         2, kernel_size=3, stride=2, padding=1),
        #     C3(in_channels=first_out*2, out_channels=first_out *
        #        2, width_multiple=0.5, depth=2),
        #     CBL(in_channels=first_out*2, out_channels=first_out *
        #         4, kernel_size=3, stride=2, padding=1),
        #     C3(in_channels=first_out*4, out_channels=first_out *
        #        4, width_multiple=0.5, depth=4),
        #     CBL(in_channels=first_out*4, out_channels=first_out *
        #         8, kernel_size=3, stride=2, padding=1),
        #     C3(in_channels=first_out*8, out_channels=first_out *
        #        8, width_multiple=0.5, depth=6),
        #     CBL(in_channels=first_out*8, out_channels=first_out *
        #         16, kernel_size=3, stride=2, padding=1),
        #     C3(in_channels=first_out*16, out_channels=first_out *
        #        16, width_multiple=0.5, depth=2),
        #     SPPF(in_channels=first_out*16, out_channels=first_out*16)
        # ]

        # create the neck of the model
        self.neck = nn.ModuleList()

        self.neck += [
            CBL(in_channels=first_out*16, out_channels=first_out *
                8, kernel_size=1, stride=1, padding=0),
            C3(in_channels=first_out*16, out_channels=first_out *
               8, width_multiple=0.25, depth=2, backbone=False),
            CBL(in_channels=first_out*8, out_channels=first_out *
                4, kernel_size=1, stride=1, padding=0),
            C3(in_channels=first_out*8, out_channels=first_out *
               4, width_multiple=0.25, depth=2, backbone=False),
            CBL(in_channels=first_out*4, out_channels=first_out *
                4, kernel_size=3, stride=2, padding=1),
            C3(in_channels=first_out*8, out_channels=first_out *
               8, width_multiple=0.5, depth=2, backbone=False),
            CBL(in_channels=first_out*8, out_channels=first_out *
                8, kernel_size=3, stride=2, padding=1),
            C3(in_channels=first_out*16, out_channels=first_out *
               16, width_multiple=0.5, depth=2, backbone=False)
        ]

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, Octant, One_sixteenth, One_thirtysecond):
        # assert x.shape[2] % 32 == 0 and x.shape[3] % 32 == 0, "if Width and Height aren't divisible by 32! raise and error"

        backbone_connection = [Octant, One_sixteenth, One_thirtysecond]
        # neck_connection = []
        # outputs = []

        # self.cbl_1(One_thirtysecond)
        cbl_1_out = self.neck[0](One_thirtysecond)
        up_out = self.upsample(cbl_1_out)
        up_out = torch.cat((up_out, One_sixteenth), dim=1)

        c3_1_out = self.neck[1](up_out)  # self.c3_1(up_out)

        cbl_2_out = self.neck[2](c3_1_out)  # self.cbl_2(c3_1_out)
        up_out = self.upsample(cbl_2_out)
        up_out = torch.cat((up_out, Octant), dim=1)

        H_80 = self.neck[3](up_out)  # 80*80 head  # self.c3_2(up_out)

        cbl_3_out = self.neck[4](H_80)  # self.cbl_3(H80)
        cbl_3_out = torch.cat((cbl_3_out, cbl_2_out), dim=1)

        H_40 = self.neck[5](cbl_3_out)  # 40*40 head # self.c3_3(cbl_3_out)

        cbl_4_out = self.neck[6](H_40)  # self.cbl_4(H_40)
        cbl_4_out = torch.cat((cbl_4_out, cbl_1_out), dim=1)

        H_20 = self.neck[7](cbl_4_out)  # 20*20 head  #self.c3_4(cbl_4_out)

        return H_80, H_40, H_20

        # for idx, layer in enumerate(self.neck):
        #     if idx in [0, 2]:
        #         x = layer(x)
        #         neck_connection.append(x)
        #         x = Resize([x.shape[2]*2, x.shape[3]*2],
        #                    interpolation=InterpolationMode.NEAREST)(x)
        #         x = torch.cat([x, backbone_connection.pop(-1)], dim=1)

        #     elif idx in [4, 6]:
        #         x = layer(x)
        #         x = torch.cat([x, neck_connection.pop(-1)], dim=1)

        #     elif isinstance(layer, C3) and idx > 2:
        #         x = layer(x)
        #         outputs.append(x)

        #     else:
        #         x = layer(x)

        # return self.head(outputs)


class YoloHeads(nn.Module):
    def __init__(self, nc=10, anchors=(), ch=()):  # detection layer
        super(YoloHeads, self).__init__()
        self.nc = nc  # number of classes
        self.nl = len(anchors)  # number of detection layers
        self.naxs = len(anchors[0])  # number of anchors per scale
        self.stride = [8, 16, 32]

        # anchors are divided by the stride (anchors_for_head_1/8, anchors_for_head_1/16 etc.)
        anchors_ = torch.tensor(anchors).float().view(
            self.nl, -1, 2) / torch.tensor(self.stride).repeat(6, 1).T.reshape(3, 3, 2)
        self.register_buffer('anchors', anchors_)

        self.out_convs = nn.ModuleList()
        
        for in_channels in ch:
            self.out_convs += [
                nn.Conv2d(in_channels=in_channels, out_channels=(
                    5+self.nc) * self.naxs, kernel_size=1)
            ]

    def forward(self, x):
        for i in range(self.nl):
            x[i] = self.out_convs[i](x[i])
            bs, _, grid_y, grid_x = x[i].shape
            x[i] = x[i].view(bs, self.naxs, (5+self.nc), grid_y,
                             grid_x).permute(0, 1, 3, 4, 2).contiguous()

        return x


ANCHORS = (
    [10, 13, 16, 30, 33, 23],  # P3/8
    [30, 61, 62, 45, 59, 119],  # P4/16
    [116, 90, 156, 198, 373, 326] # P5/32
)


class MultiNet(nn.Module):
    def __init__(self):
        super(MultiNet, self).__init__()
        self.BackBone = BackBone(in_channels=3, out_channels=512)

        # self.Seg = SegNeck()

        self.ObjNeck = Yolov5(first_out=64)
        self.ObjHead = YoloHeads(nc=10, anchors=ANCHORS, ch=(256, 512, 1024))

    def forward(self, x):
        Half, Quarter, Octant, One_sixteenth, One_thirtysecond = self.BackBone(
            x)

        # Branch1, Branch2, Branch3 = self.Seg(
        #     Half, Quarter, Octant, One_sixteenth)

        h1, h2, h3 = self.ObjNeck(Octant, One_sixteenth, One_thirtysecond)
        pred = self.ObjHead([h1, h2, h3])

        return pred
        # return Branch1, Branch2, Branch3  # ,Branch4


# %pip install torchinfo
model = MultiNet()
# torchinfo.summary(upsample_conv,(1,512, 20, 20))
torchinfo.summary(model, (1, 3, 640, 640))

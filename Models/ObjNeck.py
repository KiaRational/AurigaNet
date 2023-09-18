import torch 
from torch import nn
from .BackBone import BottleNeck


class CBL(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=padding, stride=stride, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class ObjBottleneck(nn.Module):
    """
    Parameters:
        in_channels (int): number of channel of the input tensor
        out_channels (int): number of channel of the output tensor
        width_multiple (float): it controls the number of channels (and weights)
                                of all the convolutions beside the
                                first and last one. If closer to 0,
                                the simpler the modelIf closer to 1,
                                the model becomes more complex
    """

    def __init__(self, in_channels, out_channels, width_multiple=1):
        super(ObjBottleneck, self).__init__()
        c_ = int(width_multiple*in_channels)
        self.c1 = CBL(in_channels, c_, kernel_size=1, stride=1, padding=0)
        self.c2 = CBL(c_, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.c2(self.c1(x)) + x



class C3(nn.Module):
    """
    Parameters:
        in_channels (int): number of channel of the input tensor
        out_channels (int): number of channel of the output tensor
        width_multiple (float): it controls the number of channels (and weights)
                                of all the convolutions beside the
                                first and last one. If closer to 0,
                                the simpler the modelIf closer to 1,
                                the model becomes more complex
        depth (int): it controls the number of times the bottleneck (residual block)
                        is repeated within the C3 block
        backbone (bool): if True, self.seq will be composed by bottlenecks 1, if False
                            it will be composed by bottlenecks 2 (check in the image linked below)
    """

    def __init__(self, in_channels, out_channels, width_multiple=1, depth=1, backbone=True):
        super(C3, self).__init__()
        c_ = int(width_multiple*in_channels)

        self.c1 = CBL(in_channels, c_, kernel_size=1, stride=1, padding=0)
        self.c_skipped = CBL(
            in_channels,  c_, kernel_size=1, stride=1, padding=0)
        if backbone:
            self.seq = nn.Sequential(
                *[ObjBottleneck(c_, c_, width_multiple=1) for _ in range(depth)]
            )
        else:
            self.seq = nn.Sequential(
                *[nn.Sequential(
                    CBL(c_, c_, 1, 1, 0),
                    CBL(c_, c_, 3, 1, 1)
                ) for _ in range(depth)]
            )
        self.c_out = CBL(c_ * 2, out_channels,
                         kernel_size=1, stride=1, padding=0)
                         

    def forward(self, x):
        x = torch.cat([self.seq(self.c1(x)), self.c_skipped(x)], dim=1)
        return self.c_out(x)


class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SPPF, self).__init__()

        c_ = int(in_channels//2)

        self.c1 = CBL(in_channels, c_, 1, 1, 0)
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.c_out = CBL(c_ * 4, out_channels, 1, 1, 0)

    def forward(self, x):
        x = self.c1(x)
        pool1 = self.pool(x)
        pool2 = self.pool(pool1)
        pool3 = self.pool(pool2)

        return self.c_out(torch.cat([x, pool1, pool2, pool3], dim=1))



class YoloNeck(nn.Module):
    def __init__(self, out_channels_list):
        super(YoloNeck, self).__init__()

        self.out_channels = out_channels_list # for example [32, 64, 128, 256]
        # create the neck of the model
        input_channels = out_channels_list[-1] * 2
        print('\n\n', input_channels, '\n\n')
        # self.base_cbl = CBL(in_channels=input_channels, out_channels=input_channels,
        #      kernel_size=3, stride=2, padding=1)

        self.back = nn.ModuleList()

        self.back += [
            CBL(in_channels = input_channels//2, out_channels=input_channels,
                kernel_size=3, stride=2, padding=1),
            SPPF(input_channels, input_channels) # spff layer
        ]

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest') # upsampling 

        # create the neck of the model
        self.neck = nn.ModuleList()

        self.neck += [
            CBL(in_channels = input_channels, out_channels=input_channels//2,
             kernel_size=1, stride=1, padding=0),

            C3(in_channels=input_channels, out_channels=input_channels//2,
             width_multiple=0.25, depth=2, backbone=False),

            CBL(in_channels=input_channels//2, out_channels=input_channels//4,
             kernel_size=1, stride=1, padding=0),

            C3(in_channels=input_channels//2, out_channels=input_channels//4,
             width_multiple=0.25, depth=2, backbone=False),

            CBL(in_channels=input_channels//4, out_channels=input_channels//4,
             kernel_size=3, stride=2, padding=1),

            C3(in_channels=input_channels//2, out_channels=input_channels//2,
             width_multiple=0.5, depth=2, backbone=False),

            CBL(in_channels=input_channels//2, out_channels=input_channels//2,
             kernel_size=3, stride=2, padding=1),

            C3(in_channels=input_channels, out_channels=input_channels,
             width_multiple=0.5, depth=2, backbone=False)
        ]



    def forward(self, x):
        Quarter, Octant, One_sixteenth = x # NOTE: no use for Quarter (160x160) output
        # assert x.shape[2] % 32 == 0 and x.shape[3] % 32 == 0, "if Width and Height aren't divisible by 32! raise and error"

        # backbone_connection = [Quarter, Octant, One_sixteenth]
        # neck_connection = []
        # outputs = []

        base_cbl_out = self.back[0](One_sixteenth) # NOTE: this makes 20x20 output

        spff_out  = self.back[1](base_cbl_out)

        cbl_1_out = self.neck[0](spff_out)
        up_out    = self.upsample(cbl_1_out)
        up_out    = torch.cat((up_out, One_sixteenth), dim=1)

        c3_1_out  = self.neck[1](up_out)  # self.c3_1(up_out)

        cbl_2_out = self.neck[2](c3_1_out)  # self.cbl_2(c3_1_out)
        up_out    = self.upsample(cbl_2_out)
        up_out    = torch.cat((up_out, Octant), dim=1)

        H_80      = self.neck[3](up_out)  # 80*80 head  # self.c3_2(up_out)

        cbl_3_out = self.neck[4](H_80)  # self.cbl_3(H80)
        cbl_3_out = torch.cat((cbl_3_out, cbl_2_out), dim=1)

        H_40      = self.neck[5](cbl_3_out)  # 40*40 head # self.c3_3(cbl_3_out)

        cbl_4_out = self.neck[6](H_40)  # self.cbl_4(H_40)
        cbl_4_out = torch.cat((cbl_4_out, cbl_1_out), dim=1)

        H_20      = self.neck[7](cbl_4_out)  # 20*20 head  # self.c3_4(cbl_4_out)

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
        anchors_ = torch.tensor(anchors).float().view(self.nl, -1, 2)
        anchors_ /= torch.tensor(self.stride).repeat(6, 1).T.reshape(3, 3, 2)
        # print("################anchors##################", anchors_) # NOTE: for test and will be deleted
        self.register_buffer('anchors', anchors_)

        self.out_convs = nn.ModuleList()
        
        for in_channels in ch:
            self.out_convs += [
                nn.Conv2d(in_channels=in_channels, out_channels=(
                    5 + self.nc) * self.naxs, kernel_size=1)
            ]


    def forward(self, x):
        for i in range(self.nl):
            x[i] = self.out_convs[i](x[i])
            bs, _, grid_y, grid_x = x[i].shape
            x[i] = x[i].view(bs, self.naxs, (5 + self.nc), grid_y,
                             grid_x) # the example tensor shape for grid size 160: [1, 3, 15, 160, 160]
            x[i] = x[i].permute(0, 1, 3, 4, 2).contiguous() # # the example tensor shape for grid size 160: ([1, 3, 160, 160, 15])
            print(f"\n the shape of a layer {x[i].shape} \n")

        return x


class Object(nn.Module):
    def __init__(self, hyp, anchors, out_channels_list):
        super(Object, self).__init__()

        self.hyp = hyp 

        self.ObjNeck = YoloNeck(out_channels_list=out_channels_list) # 512
        self.ObjHead = YoloHeads(nc=10, anchors=anchors, ch=(2*oc for oc in out_channels_list[1:]))


    def forward(self, x):

        h1, h2, h3 = self.ObjNeck([x[0], x[1], x[2]])

        preds = self.ObjHead([h1, h2, h3])

        return preds
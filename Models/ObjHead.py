import torch 
from torch import nn
from .BackBone import BottleNeck


import os 
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from seg_utils.Parameters import Parameters


class ConvBNSiLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
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
        self.c1 = ConvBNSiLU(in_channels, c_, kernel_size=1, stride=1, padding=0)
        self.c2 = ConvBNSiLU(c_, out_channels, kernel_size=3, stride=1, padding=1)

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

        self.c1 = ConvBNSiLU(in_channels, c_, kernel_size=1, stride=1, padding=0)
        self.c_skipped = ConvBNSiLU(
            in_channels,  c_, kernel_size=1, stride=1, padding=0)
        if backbone:
            self.seq = nn.Sequential(
                *[ObjBottleneck(c_, c_, width_multiple=1) for _ in range(depth)]
            )
        else:
            self.seq = nn.Sequential(
                *[nn.Sequential(
                    ConvBNSiLU(c_, c_, 1, 1, 0),
                    ConvBNSiLU(c_, c_, 3, 1, 1)
                ) for _ in range(depth)]
            )
        self.c_out = ConvBNSiLU(c_ * 2, out_channels,
                         kernel_size=1, stride=1, padding=0)
                         

    def forward(self, x):
        x = torch.cat([self.seq(self.c1(x)), self.c_skipped(x)], dim=1)
        return self.c_out(x)


class ObjHead(nn.Module):
    def __init__(self,out_channels_list,w,r,d):
        super(ObjHead, self).__init__()
        out_channels_list = [64,128,256,512]
        self.cbl_1 = ConvBNSiLU(in_channels=out_channels_list[2]//w,out_channels=out_channels_list[2]//w,kernel_size=3,stride=2,padding=1)
        self.c3_1 = C3(in_channels=(out_channels_list[2]+out_channels_list[3])//w,out_channels=(out_channels_list[3])//w,depth=3//d)
        self.cbl_2 = ConvBNSiLU(in_channels=out_channels_list[3]//w,out_channels=out_channels_list[3]//w,kernel_size=3,stride=2,padding=1)
        self.c3_2 = C3(in_channels=(out_channels_list[3])*(1+r)//w,out_channels=(out_channels_list[3])*r//w,depth=3//d)

    def forward(self,Octant_in , One_sixteenth_in , One_thirty_second_in):
        
        One_sixteenth_temp = self.cbl_1(Octant_in)

        One_sixteenth_Head = self.c3_1(torch.cat((One_sixteenth_in,One_sixteenth_temp),dim=1))

        One_thirty_second_temp = self.cbl_2(One_sixteenth_Head)

        One_thirty_second_Head = self.c3_2(torch.cat((One_thirty_second_temp,One_thirty_second_in),dim=1))

        return Octant_in , One_sixteenth_Head , One_thirty_second_Head

class ObjDetect(nn.Module):

    def __init__(self, nc=10, anchors=(), ch=()):  # detection layer

        super(ObjDetect, self).__init__()
        self.nc = nc  # number of classes
        self.nl = len(anchors)  # number of detection layers
        self.naxs = len(anchors[0])  # number of anchors per scale
        self.stride = [8, 16, 32]

        # anchors are divided by the stride (anchors_for_head_1/8, anchors_for_head_1/16 etc.)
        anchors_ = anchors.clone().detach().float().view(self.nl, -1, 2)
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

        return x

class Object(nn.Module):
    def __init__(self, anchors, out_channels_list,w,r,d):
        super(Object, self).__init__()

        self.ObjHead = ObjHead(out_channels_list=out_channels_list,w=w,r=r,d=d)
        self.ObjDetect = ObjDetect(nc=10, anchors=anchors, ch=(r*oc//w for oc in out_channels_list[1:]))


    def forward(self,Octant_in , One_sixteenth_in , One_thirty_second_in ):

        Octant_out , One_sixteenth_out , One_thirty_second_out = self.ObjHead(Octant_in , One_sixteenth_in , One_thirty_second_in)

        preds = self.ObjDetect([Octant_out , One_sixteenth_out , One_thirty_second_out])
        return preds
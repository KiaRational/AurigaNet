import torch 
from torch import nn


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
        self.na = len(anchors[0])  # number of anchors per scale
        self.stride = [8, 16, 32]
        self.dynamic = False
        # anchors are divided by the stride (anchors_for_head_1/8, anchors_for_head_1/16 etc.)
        anchors_ = anchors.clone().detach().float().view(self.nl, -1, 2).to("cuda")
        anchors_ /= torch.tensor(self.stride,device="cuda").repeat(6, 1).T.reshape(3, 3, 2)
        # print("################anchors##################", anchors_) # NOTE: for test and will be deleted
        self.grid = [torch.empty(0,device = "cuda") for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0 ,  device = "cuda") for _ in range(self.nl)]  # init anchor grid
        self.anchors = anchors_
        self.out_convs = nn.ModuleList()
        
        for in_channels in ch:
            self.out_convs += [
                nn.Conv2d(in_channels=in_channels, out_channels=(
                    5 + self.nc) * self.na, kernel_size=1)
            ]


    def forward(self, x):
        batch_boxes = []
        for i, conv in enumerate(self.out_convs):
            x[i] = conv(x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.nc + 5, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

            layer_prediction = x[i].sigmoid()
            obj = layer_prediction[..., 4:5]
            xy = (2 * layer_prediction[..., :2] + self.grid[i]) * self.stride[i]
            wh = ((2 * layer_prediction[..., 2:4]) ** 2) * self.anchor_grid[i]
            best_class = torch.argmax(layer_prediction[..., 5:], dim=-1).unsqueeze(-1)

            # Concatenate predictions along the last dimension and reshape
            scale_bboxes = torch.cat((best_class, obj, xy, wh), dim=-1).view(bs, -1, 5 + self.nc)
            batch_boxes.append(scale_bboxes)

        return [x, torch.cat(batch_boxes, dim=1)]
    # def forward(self, x):
    #     batch_boxes = []
    #     for i, conv in enumerate(self.out_convs):
    #         x[i] = conv(x[i])
    #         bs, _, ny, nx = x[i].shape
    #         x[i] = x[i].view(bs, self.na, self.nc + 5, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

    #         if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
    #             self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

    #         xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
    #         obj = conf[..., 0:1]
    #         best_class = torch.argmax(conf[..., 1:], dim=-1).unsqueeze(-1)

    #         xy = (2 * xy + self.grid[i] - 0.5) * self.stride[i]
    #         wh = ((2 * wh) ** 2) * self.anchor_grid[i]

    #         # Concatenate predictions along the last dimension and reshape
    #         scale_bboxes = torch.cat((best_class, obj, xy, wh), dim=-1).view(bs, -1, 5 + self.nc)
    #         batch_boxes.append(scale_bboxes)

    #     return [x, torch.cat(batch_boxes, dim=1)]
    def _make_grid(self, nx=20, ny=20, i=0):
        """Generates a mesh grid for anchor boxes with optional compatibility for torch versions < 1.10."""
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing="ij")  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid

class Object(nn.Module):
    def __init__(self, anchors, out_channels_list,w,r,d):
        super(Object, self).__init__()
        self.Parameters = Parameters()
        self.ObjHead = ObjHead(out_channels_list=out_channels_list,w=w,r=r,d=d)
        self.ObjDetect = ObjDetect(nc=len(self.Parameters.bdd), anchors=anchors, ch=(out_channels_list[2]//w,out_channels_list[3]//w,(out_channels_list[3]*r)//w))


    def forward(self,Octant_in , One_sixteenth_in , One_thirty_second_in ):

        Octant_out , One_sixteenth_out , One_thirty_second_out = self.ObjHead(Octant_in , One_sixteenth_in , One_thirty_second_in)
        preds = self.ObjDetect([Octant_out , One_sixteenth_out , One_thirty_second_out])
        
        return preds

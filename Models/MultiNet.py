import torch
from torch import nn
import os
import sys 
import torchinfo
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from Models.BackBone import BackBone
from Models.Neck import Neck
from Models.SegHead import SegHead
from Models.ObjHead import Object

ANCHORS = torch.tensor([
    [[10, 13], [16, 30], [33, 23]],  # P3/8
    [[30, 61], [62, 45], [59, 119]],  # P4/16
    [[116, 90], [156, 198], [373, 326]] # P5/32
]).detach().requires_grad_(False).float()


class MultiNet(nn.Module):
    """
    A multi-network architecture composed of a Backbone and a Segmentation Neck.

    This network takes an input tensor and processes it through both a backbone and a segmentation neck.

    Attributes:
        BackBone (BackBone): The backbone module.
        Seg (SegNeck): The segmentation neck module.
    """
    def __init__(self): # TODO object_hyp will be moved to somewhere else in the future
        super(MultiNet, self).__init__()

        out_channels_list = [64,128,256,512]
        w = 4 #width coefficient
        r = 2 #ratio
        d = 3 #depth

        self.BackBone = BackBone(in_channels=3, out_channels_list=out_channels_list , w=w , r=r , d=d)
        self.Neck = Neck(out_channels_list,w=w,r=r,d=d)
        # self.Seg = SegHead(out_channels_list,w=w,r=r,d=d,class_number=2)

        # self.Seg = SegNeck(out_channels_list , class_number = 2)

        # ###### for object ######
        # self.nl = 3 # number of detection layers
        # self.na = 3
        # self.anchors = ANCHORS

        # # self.Seg = SegNeck()

        # self.Obj = Object(AzNCHORS, out_channels_list=out_channels_list,w=w,r=r,d=d)

    def forward(self, x):
        """
        Forward pass of the MultiNet.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor representing the segmented predictions.
        """
        Half, Quarter, Octant, One_sixteenth  = self.BackBone(x)
        Octant_out , One_sixteenth_out , One_thirty_second_out = self.Neck(Octant,One_sixteenth)
        # Out = self.Seg(Half, Quarter, Octant_out, One_sixteenth_out)

        # Out = self.Obj(Octant_out , One_sixteenth_out , One_thirty_second_out) # object predictions
        # Out = [Half, Quarter, Octant, One_sixteenth]
        return Out

if __name__ == "__main__":

    model = MultiNet()
    # torchinfo.summary(upsample_conv,(1,512, 20, 20))
    torchinfo.summary(model, (1, 3, 640, 640))

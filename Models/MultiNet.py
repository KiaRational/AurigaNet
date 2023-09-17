import torch
from torch import nn
import os
import sys 
import torchinfo
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from Models.BackBone import BackBone
from Models.SegNeck import SegNeck
from Models.ObjNeck import Object

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
    def __init__(self, object_hyp): # TODO object_hyp will be moved to somewhere else in the future
        super(MultiNet, self).__init__()
        out_channels_list = [32,64,128,256]
        self.BackBone = BackBone(in_channels=3, out_channels_list=out_channels_list)
        self.Seg = SegNeck(out_channels_list , class_number = 2)

        ###### for object ######
        self.nl = 3 # number of detection layers
        self.na = 3
        self.anchors = ANCHORS

        # self.Seg = SegNeck()

        self.Obj = Object(object_hyp, ANCHORS, out_channels_list=out_channels_list)
    
    def forward(self, x):
        """
        Forward pass of the MultiNet.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor representing the segmented predictions.
        """
        Half, Quarter, Octant, One_sixteenth  = self.BackBone(x)
        # Out = self.Seg(Half, Quarter, Octant, One_sixteenth)

        Out = self.Obj([Quarter, Octant, One_sixteenth]) # object predictions

        return Out

if __name__ == "__main__":

    model = MultiNet({})
    # torchinfo.summary(upsample_conv,(1,512, 20, 20))
    torchinfo.summary(model, (1, 3, 640, 640))

import torch
from torch import nn
import os
import sys 
import torchinfo
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from Models.BackBone import BackBone
from Models.SegNeck import SegNeck
from Models.ObjNeck import ObjHead , ObjNeck

ANCHORS = (
    [10, 13, 16, 30, 33, 23],  # P3/8
    [30, 61, 62, 45, 59, 119],  # P4/16
    [116, 90, 156, 198, 373, 326] # P5/32
)



class MultiNet(nn.Module):
    """
    A multi-network architecture composed of a Backbone and a Segmentation Neck.

    This network takes an input tensor and processes it through both a backbone and a segmentation neck.

    Attributes:
        BackBone (BackBone): The backbone module.
        Seg (SegNeck): The segmentation neck module.
    """
    def __init__(self):
        super(MultiNet, self).__init__()
        self.BackBone = BackBone(in_channels=3, out_channels=512)
        self.Seg = SegNeck(Resize=2)
        self.ObjNeck = ObjNeck(first_out=64)
        self.ObjHead = ObjHead(nc=10, anchors=ANCHORS, ch=(256, 512, 1024))

    def forward(self, x):
        """
        Forward pass of the MultiNet.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor representing the segmented predictions.
        """
        Half, Quarter, Octant, One_sixteenth, One_thirtysecond  = self.BackBone(x)
        Out = self.Seg(Half, Quarter, Octant, One_sixteenth)
        h1, h2, h3 = self.ObjNeck(Octant, One_sixteenth, One_thirtysecond)
        pred = self.ObjHead([h1, h2, h3])

        return pred
        # return Branch1, Branch2, Branch3  # ,Branch4

        return Out


model = MultiNet()
# torchinfo.summary(upsample_conv,(1,512, 20, 20))
torchinfo.summary(model, (1, 3, 640, 640))
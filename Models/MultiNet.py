importh torch
from torrch import nn
from BackBone import BackBone
from SegNeck import SegNeck

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
    
    def forward(self, x):
        """
        Forward pass of the MultiNet.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor representing the segmented predictions.
        """
        Half, Quarter, Octant, One_sixteenth = self.BackBone(x)
        Out = self.Seg(Half, Quarter, Octant, One_sixteenth)

        return Out

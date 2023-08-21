from torch import nn
import torch

class ConvBNSiLU(nn.Sequential):
    """
    A sequential module consisting of a convolutional layer, batch normalization, and SiLU activation.
    
    Args:
        in_channels (int): Number of input channels.
        n_filters (int): Number of filters in the convolutional layer.
        k_size (int): Size of the convolutional kernel.
        padding (int): Padding added to the input during convolution.
        stride (int): Stride used for the convolution operation.
        bias (bool, optional): Whether to include bias in the convolutional layer. Default is True.
    """
    def __init__(self, in_channels, n_filters, k_size, padding, stride, bias=True):
        super().__init__(
            nn.Conv2d(in_channels, n_filters, k_size, padding=padding, stride=stride, bias=bias),
            nn.BatchNorm2d(n_filters),
            nn.SiLU(inplace=False),
        )

class BottleNeck(nn.Module):
    """
    Residual bottleneck module consisting of multiple convolutional layers with batch normalization and SiLU activation.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Size of the convolutional kernels. Default is 3.
        padding (int, optional): Padding added to the input during convolutions. Default is 1.
        stride (int, optional): Stride used for the convolution operations. Default is 1.
        resize_module (bool, optional): Whether the module is used for resizing. Default is False.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, resize_module=False):
        super(BottleNeck, self).__init__()

        if resize_module:
            stride1 = 2
        else:
            stride1 = stride

        self.conv_res = ConvBNSiLU(in_channels=in_channels, n_filters=out_channels, k_size=1, padding=0, stride=stride1)
        self.conv1 = ConvBNSiLU(in_channels=in_channels, n_filters=out_channels, k_size=1, padding=0, stride=stride1)
        self.conv2 = ConvBNSiLU(in_channels=out_channels, n_filters=out_channels, k_size=kernel_size, padding=padding, stride=stride)
        self.conv3 = ConvBNSiLU(in_channels=out_channels, n_filters=out_channels, k_size=1, padding=0, stride=1)

    def forward(self, x):
        """
        Forward pass of the bottleneck module.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor.
        """
        res = self.conv_res(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + res
        return x

class BackBone(nn.Module):
    """
    Backbone module consisting of stacked convolutional and bottleneck layers.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Size of the convolutional kernels. Default is 3.
        padding (int, optional): Padding added to the input during convolutions. Default is 1.
        stride (int, optional): Stride used for the convolution operations. Default is 1.
        resize_module (bool, optional): Whether the module is used for resizing. Default is False.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, resize_module=False):
        super(BackBone, self).__init__()
        self.layer1 = ConvBNSiLU(in_channels=in_channels, n_filters=out_channels // 16, k_size=7, padding=3, stride=1)
        self.bottleneck1 = BottleNeck(in_channels=out_channels // 16, out_channels=out_channels // 8,
                                      kernel_size=3, padding=1, stride=1, resize_module=True)
        self.bottleneck2 = BottleNeck(in_channels=out_channels // 8, out_channels=out_channels // 4,
                                      kernel_size=3, padding=1, stride=1, resize_module=True)
        self.bottleneck3 = BottleNeck(in_channels=out_channels // 4, out_channels=out_channels // 2,
                                      kernel_size=3, padding=1, stride=1, resize_module=True)
        self.bottleneck4 = BottleNeck(in_channels=out_channels // 2, out_channels=out_channels,
                                      kernel_size=3, padding=1, stride=1, resize_module=True)

    def forward(self, x):
        """
        Forward pass of the backbone module.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensors at different stages of the backbone.
        """
        x = self.layer1(x)
        Half = self.bottleneck1(x)
        Quarter = self.bottleneck2(Half)
        Octant = self.bottleneck3(Quarter)
        One_sixteenth = self.bottleneck4(Octant)

        return Half, Quarter, Octant, One_sixteenth

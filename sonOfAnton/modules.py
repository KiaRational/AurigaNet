import torch
from torch import nn

'''
segmentation modules
'''

def generate_sequence(first_num, goal_num, num_steps):
    if num_steps == 1:
        return [first_num]

    ratio = math.pow(goal_num / first_num, 1 / (num_steps - 1))
    sequence = [round(first_num * math.pow(ratio, i))
                for i in range(num_steps - 1)]
    sequence.append(goal_num)

    # Round the sequence elements to the closest power of 2 number
    sequence = [int(2 ** round(math.log2(num))) for num in sequence]

    return sequence


class ConvBNSiLU(nn.Sequential):
    def __init__(self, in_channels, n_filters, k_size, padding, stride, bias=True):
        super().__init__(
            nn.Conv2d(in_channels, n_filters, k_size,
                      padding=padding, stride=stride, bias=bias),
            nn.BatchNorm2d(n_filters),
            nn.SiLU(inplace=True),
        )


class DownsampleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, downsample_ratio=2):
        super().__init__()
        ratio_log2 = int(torch.log2(torch.tensor(downsample_ratio)).item())
        temp_channels = generate_sequence(
            in_channels, out_channels, 2**ratio_log2)
        # temp_channels.append(temp_channels[len(temp_channels)-1])
        # print(ratio_log2)
        for i in range(ratio_log2):
            if i == 0:
                temp_channels = in_channels
            else:
                temp_channels = out_channels
            self.add_module(f'Conv2d_{i}', ConvBNSiLU(
                in_channels=temp_channels, n_filters=out_channels//2, k_size=3, padding=1, stride=2))
            self.add_module(f'ConvBNSiLU_{i}', ConvBNSiLU(
                in_channels=out_channels//2, n_filters=out_channels, k_size=1, padding=0, stride=1))


class BilinearUpsampleConv1x1(nn.Sequential):
    def __init__(self, in_channels, out_channels, upsample_ratio=2):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 1, padding=0, stride=1)
        )
        for i in range(int(torch.log2(torch.tensor(upsample_ratio)).item())):
            self.add_module(f'Upsample_{i}', nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True))
            self.add_module(f'BatchNorm_{i}', nn.BatchNorm2d(out_channels))
            self.add_module(f'BatchNorm_{i}', nn.ReLU(inplace=True))


class UpsampleConvTranspose(nn.Sequential):
    def __init__(self, in_channels, out_channels, upsample_ratio=2):
        super().__init__()
        ratio_log2 = int(torch.log2(torch.tensor(upsample_ratio)).item())
        # temp_channels = generate_sequence(in_channels,out_channels,ratio_log2)
        # temp_channels.append(temp_channels[len(temp_channels)-1])
        # print(temp_channels)
        for i in range(ratio_log2):
            self.add_module(f'ConvTranspose2d_{i}', nn.ConvTranspose2d(
                in_channels, out_channels, 3, 2, 1, 1))
            self.add_module(f'Conv1x1_{i}', ConvBNSiLU(
                in_channels=out_channels, n_filters=out_channels, k_size=1, padding=0, stride=1))


class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, resize_module=False):
        super(BottleNeck, self).__init__()

        # Conditionally adjust stride for resizing the output size
        if resize_module:
            stride1 = 2
        else:
            stride1 = stride
        # Convolutional layers with batch normalization and SiLU activation
        self.conv_res = ConvBNSiLU(
            in_channels=in_channels, n_filters=out_channels, k_size=1, padding=0, stride=stride1)
        self.conv1 = ConvBNSiLU(
            in_channels=in_channels, n_filters=out_channels, k_size=1, padding=0, stride=stride1)
        self.conv2 = ConvBNSiLU(in_channels=out_channels, n_filters=out_channels,
                                k_size=kernel_size, padding=padding, stride=stride)
        self.conv3 = ConvBNSiLU(
            in_channels=out_channels, n_filters=out_channels, k_size=1, padding=0, stride=1)

    def forward(self, x):
        # Store the input tensor for the residual connection
        res = self.conv_res(x)
        # Pass the input through the convolutional layers
        x = self.conv1(x)

        x = self.conv2(x)
        x = self.conv3(x)
        # Add the residual connection
        x = x + res

        return x


class ResidualBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(ResidualBottleneckBlock, self).__init__()
        self.layer1 = ConvBNSiLU(
            in_channels=in_channels, n_filters=out_channels, k_size=1, padding=0, stride=1)
        self.layer2 = ConvBNSiLU(in_channels=in_channels, n_filters=out_channels,
                                 k_size=kernel_size, padding=padding, stride=stride)
        self.layer3 = ConvBNSiLU(
            in_channels=out_channels, n_filters=out_channels, k_size=1, padding=0, stride=1)

    def forward(self, x):
        res = x
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x + res


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, repeat_num=4):
        super(BasicBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(repeat_num):
            if i == 0:
                self.layers.append(ResidualBottleneckBlock(
                    in_channels, out_channels, kernel_size, padding, stride))
            else:
                self.layers.append(ResidualBottleneckBlock(
                    out_channels, out_channels, kernel_size, padding, stride))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


'''
object detection blocks
'''


class CBL(nn.Module):
    '''
    Conv => BatchNorm => SiLU
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(CBL, self).__init__()

        conv = nn.Conv2d(in_channels, out_channels,
                         kernel_size, stride, padding, bias=False)
        bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03)

        self.cbl = nn.Sequential(
            conv,
            bn,
            # https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.cbl(x)


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
        https://user-images.githubusercontent.com/31005897/172404576-c260dcf9-76bb-4bc8-b6a9-f2d987792583.png
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

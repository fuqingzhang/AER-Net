import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter
from torch.nn import Softmax

"""
    构造下采样模块--右边特征融合基础模块    
"""


class ExternalAttention(nn.Module):
    def __init__(self, in_planes, S=8):
        super().__init__()

        self.mk = nn.Linear(in_planes, S, bias=False)
        self.mv = nn.Linear(S, in_planes, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        b, c, h, w = x.size()
        n = h * w
        queries = x.view(b, c, n)  # 即bs,n,d_model
        queries = queries.permute(0, 2, 1)
        attn = self.mk(queries)  # bs,n,S
        attn = self.softmax(attn)  # bs,n,S
        attn = attn / (1e-9 + torch.sum(attn, dim=2, keepdim=True))  # bs,n,S
        attn = self.mv(attn)  # bs,n,d_model
        attn = attn.permute(0, 2, 1)
        x_attn = attn.view(b, c, h, w)
        x = x + x_attn
        x = F.relu(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Res_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ea = ExternalAttention(out_channels, S=8)
        self.ca = ChannelAttention(out_channels)

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ea(out)
        out = self.ca(out) * out
        out += residual
        out = self.relu(out)
        return out


"""
    构造上采样模块--左边特征提取基础模块    
"""


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


###
class conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, padding='same',
                 bias=False, bn=True, relu=False):
        super(conv, self).__init__()
        if '__iter__' not in dir(kernel_size):
            kernel_size = (kernel_size, kernel_size)
        if '__iter__' not in dir(stride):
            stride = (stride, stride)
        if '__iter__' not in dir(dilation):
            dilation = (dilation, dilation)

        if padding == 'same':
            width_pad_size = kernel_size[0] + (kernel_size[0] - 1) * (dilation[0] - 1)
            height_pad_size = kernel_size[1] + (kernel_size[1] - 1) * (dilation[1] - 1)
        elif padding == 'valid':
            width_pad_size = 0
            height_pad_size = 0
        else:
            if '__iter__' in dir(padding):
                width_pad_size = padding[0] * 2
                height_pad_size = padding[1] * 2
            else:
                width_pad_size = padding * 2
                height_pad_size = padding * 2

        width_pad_size = width_pad_size // 2 + (width_pad_size % 2 - 1)
        height_pad_size = height_pad_size // 2 + (height_pad_size % 2 - 1)
        pad_size = (width_pad_size, height_pad_size)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_size, dilation, groups, bias=bias)
        self.reset_parameters()

        if bn is True:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None

        if relu is True:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv.weight)


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,
                                                                                                                 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,
                                                                                                                 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width,
                                                                                                     height,
                                                                                                     height).permute(0,
                                                                                                                     2,
                                                                                                                     1,
                                                                                                                     3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        # print(concate)
        # print(att_H)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        # print(out_H.size(),out_W.size())
        return self.gamma * (out_H + out_W) + x


class CCA_kernel(nn.Module):
    def __init__(self, in_channel, out_channel, receptive_size=3):
        super(CCA_kernel, self).__init__()
        self.conv0 = conv(in_channel, out_channel, 1)
        self.conv1 = conv(out_channel, out_channel, kernel_size=(1, receptive_size))
        self.conv2 = conv(out_channel, out_channel, kernel_size=(receptive_size, 1))
        self.conv3 = conv(out_channel, out_channel, 3, dilation=receptive_size)

        self.ccation = CrissCrossAttention(out_channel)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.ccation(x)

        return x


class DCM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DCM, self).__init__()
        self.relu = nn.ReLU(True)

        self.branch0 = conv(in_channel, out_channel, 1)
        self.branch1 = CCA_kernel(in_channel, out_channel, 3)
        self.branch2 = CCA_kernel(in_channel, out_channel, 5)
        self.branch3 = CCA_kernel(in_channel, out_channel, 7)

        self.conv_cat = conv(4 * out_channel, out_channel, 3)
        self.conv_res = conv(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.relu(x_cat + self.conv_res(x))

        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Refinement_Module(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Refinement_Module, self).__init__()
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(out_channel, out_channel))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        fusion = g + x
        att_1 = self.mlp(
            F.avg_pool2d(fusion, (fusion.size(2), fusion.size(3)), stride=(fusion.size(2), fusion.size(3))))
        att_2 = self.mlp(
            F.max_pool2d(fusion, (fusion.size(2), fusion.size(3)), stride=(fusion.size(2), fusion.size(3))))
        scale = torch.sigmoid(att_1 + att_2).unsqueeze(2).unsqueeze(3).expand_as(x)
        out = self.relu(x * scale)

        return out


"""
    模型主架构
"""


class Network(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, dim=16, deep_supervision=True, **kwargs):
        super(Network, self).__init__()
        self.deep_supervision = deep_supervision

        # 卷积参数设置
        n1 = dim
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        num_blocks = [2, 2, 2, 2]
        block = Res_block

        # 最大池化层
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 左边特征提取卷积层
        self.Conv1 = self._make_layer(block, in_ch, filters[0])
        self.Conv2 = self._make_layer(block, filters[0], filters[1], num_blocks[0])
        self.Conv3 = self._make_layer(block, filters[1], filters[2], num_blocks[1])
        self.Conv4 = self._make_layer(block, filters[2], filters[3], num_blocks[2])
        self.Conv5 = self._make_layer(block, filters[3], filters[4], num_blocks[3])

        # 右边特征融合反卷积层
        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = self._make_layer(block, filters[3] * 2, filters[3], num_blocks[0])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = self._make_layer(block, filters[2] * 2, filters[2], num_blocks[0])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = self._make_layer(block, filters[1] * 2, filters[1], num_blocks[0])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = self._make_layer(block, filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        ###
        self.DCM3 = DCM(filters[2], filters[2])
        self.DCM4 = DCM(filters[3], filters[3])
        self.DCM5 = DCM(filters[4], filters[4])
        self.RM1 = Refinement_Module(filters[0], filters[0])
        self.RM2 = Refinement_Module(filters[1], filters[1])
        self.ce5 = nn.Conv2d(filters[4], 1, kernel_size=1, stride=1)
        self.cd5 = nn.Conv2d(filters[3], 1, kernel_size=1, stride=1)
        self.cd4 = nn.Conv2d(filters[2], 1, kernel_size=1, stride=1)
        self.cd3 = nn.Conv2d(filters[1], 1, kernel_size=1, stride=1)  ###

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

        # 前向计算，输出一张与原图相同尺寸的图片矩阵

    def forward(self, x):

        x_size = x.size()
        e1 = self.Conv1(x)
        e2 = self.Conv2(self.Maxpool1(e1))
        e3 = self.Conv3(self.Maxpool2(e2))
        e4 = self.Conv4(self.Maxpool3(e3))
        e5 = self.Conv5(self.Maxpool4(e4))


        e3 = self.DCM3(e3)
        e4 = self.DCM4(e4)
        e5 = self.DCM5(e5)

        out5 = F.interpolate(self.ce5(e5), x_size[2:], mode='bilinear', align_corners=False)  ###

        d5 = torch.cat((e4, self.Up5(e5)), dim=1)  # 将e4特征图与d5特征图横向拼接
        d5 = self.Up_conv5(d5)

        out4 = F.interpolate(self.cd5(d5), x_size[2:], mode='bilinear', align_corners=False)  ###

        d4 = torch.cat((e3, self.Up4(d5)), dim=1)  # 将e3特征图与d4特征图横向拼接
        d4 = self.Up_conv4(d4)

        out3 = F.interpolate(self.cd4(d4), x_size[2:], mode='bilinear', align_corners=False)  ###

        d3 = self.Up3(d4)
        d3 = torch.cat((self.RM2(g=d3, x=e2), d3), dim=1)  # 将e2特征图与d3特征图横向拼接
        d3 = self.Up_conv3(d3)

        out2 = F.interpolate(self.cd3(d3), x_size[2:], mode='bilinear', align_corners=False)  ###

        d2 = self.Up2(d3)
        d2 = torch.cat((self.RM1(g=d2, x=e1), d2), dim=1)  # 将e1特征图与d1特征图横向拼接
        d2 = self.Up_conv2(d2)

        out1 = self.Conv(d2)

        return [out5, out4, out3, out2, out1]


if __name__ == '__main__':
    model = Network(dim=16)

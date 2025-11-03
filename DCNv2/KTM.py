import torch
import torch.nn as nn
from torch.nn import Softmax
from dcn_v2 import DCN
import torch.nn.functional as F
from PIL import Image
from datetime import datetime

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

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
        # return self.sigmoid(x)
        return x




# DFOM Module
class DFOM(nn.Module):
    def __init__(self, channel=32):
        super(DFOM, self).__init__()

        self.query_conv = nn.Conv2d(channel, channel // 2, kernel_size=1)
        self.key_conv = nn.Conv2d(channel, channel // 2, kernel_size=1)
        self.conv_x1_sum = nn.Conv2d(channel, channel, kernel_size=1)
        self.conv_x1_mul = nn.Conv2d(channel, channel, kernel_size=1)
        self.conv_x2 = nn.Conv2d(channel, channel * 2, kernel_size=1)
        self.conv_di2 = nn.Conv2d(channel * 2, channel, kernel_size=1)
        self.value_conv_2 = nn.Conv2d(channel, channel, kernel_size=1)
        self.value_conv_3 = nn.Conv2d(channel, channel, kernel_size=1)
        self.gamma_2 = nn.Parameter(torch.zeros(1))
        self.gamma_3 = nn.Parameter(torch.zeros(1))

        self.Dcn_sum = DCN(channel, channel, kernel_size=(3, 3), stride=1, padding=1)
        self.Dcn_mul = DCN(channel, channel, kernel_size=(3, 3), stride=1, padding=1)

        self.softmax = Softmax(dim=-1)

        # following DANet
        self.conv_2 = nn.Sequential(BasicConv2d(channel, channel, 3, padding=1),
                                    nn.ReLU(),
                                    nn.Dropout2d(0.1, False),
                                    nn.Conv2d(channel, channel, 1)
                                    )
        self.conv_3 = nn.Sequential(BasicConv2d(channel, channel, 3, padding=1),
                                    nn.ReLU(),
                                    nn.Dropout2d(0.1, False),
                                    nn.Conv2d(channel, channel, 1)
                                    )

        self.conv_out = nn.Sequential(nn.Dropout2d(0.1, False),
                                      nn.Conv2d(channel, channel, 1)
                                      )
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.SA = SpatialAttention()

        self.topdown = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channel, channel // 4, 1, 1, 0),
            nn.BatchNorm2d(channel // 4),
            nn.ReLU(True),

            nn.Conv2d(channel // 4, channel, 1, 1, 0),
            nn.BatchNorm2d(channel),
            nn.Sigmoid(),
        )

        self.bottomup = nn.Sequential(
            nn.Conv2d(channel, channel // 4, 1, 1, 0),
            nn.BatchNorm2d(channel // 4),
            nn.ReLU(True),

            nn.Conv2d(channel // 4, channel, 1, 1, 0),
            nn.BatchNorm2d(channel),
            nn.Sigmoid(),
        )


        """
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // 4, kernel_size=1),
            nn.BatchNorm2d(channel // 4),
            nn.ReLU(True),

            nn.Conv2d(channel // 4, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.Sigmoid(),
        )
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 4, kernel_size=1),
            nn.BatchNorm2d(channel // 4),
            nn.ReLU(True),

            nn.Conv2d(channel // 4, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.Sigmoid(),
        )
        """
    def prin_show(self, img, name):
        y1 = F.interpolate(img, size=(256, 256), mode='bilinear', align_corners=False)
        #
        f = 1
        if f == 0:
            yy = y1[0, 0, :, :]
            # print(yy.shape)
            # ���������ŵ� 0-255 �ķ�Χ���Ա㱣��Ϊͼ��
            scaled_tensor = ((yy - yy.min()) / (
                    yy.max() - yy.min()) * 255).byte()

            # ���� PIL Image ����
            image = Image.fromarray(scaled_tensor.cpu().numpy())

            current_datetime = datetime.now()
            formatted_datetime = current_datetime.strftime("%H-%M-%S")
            # ����ͼ��
            image.save("/root/autodl-tmp/ABC-add/image/" + str(name) + str(formatted_datetime) + ".png")
        else:
            b, _, _, _ = y1.shape
            for i in range(b):
                yy = y1[i, 0, :, :]
                # print(yy.shape)
                # ���������ŵ� 0-255 �ķ�Χ���Ա㱣��Ϊͼ��
                scaled_tensor = ((yy - yy.min()) / (
                        yy.max() - yy.min()) * 255).byte()

                # ���� PIL Image ����
                image = Image.fromarray(scaled_tensor.cpu().numpy())

                current_datetime = datetime.now()
                formatted_datetime = current_datetime.strftime("%H-%M-%S")
                # ����ͼ��

                image.save("/root/autodl-tmp/ABC-add/image/" + str(name) + str(formatted_datetime) + str(i) + ".png")

    def forward(self, x2, x3):  # V

        x_sum = x2 + x3  # Q
        x_mul = x2 * x3  # K


        proj_query_pr = self.value_conv_2(x_sum)
        proj_key_pr = self.value_conv_3(x_mul)

        # dcn
        x_sum_x1 = self.conv_x1_sum(x_sum)
        x_mul_x1 = self.conv_x1_mul(x_mul)
        dcn_sum = self.Dcn_sum(x_sum_x1)
        dcn_mul = self.Dcn_mul(x_mul_x1)

        proj_query = torch.cat((dcn_sum, proj_query_pr), dim=1)
        proj_key = torch.cat((dcn_mul, proj_key_pr), dim=1)
        proj_query = self.conv_di2(proj_query)
        proj_key = self.conv_di2(proj_key)

        #proj_query = self.pa(proj_query)
        #proj_key = self.pa(proj_key)
        proj_query = self.bottomup(proj_query)
        proj_key = self.bottomup(proj_key)


        x_out_1 = x2 * proj_query
        x_out_1 = x_out_1 * proj_key

        x_out_sa = self.topdown(x_out_1)
        x_out_ca = self.topdown(proj_key_pr)
        x_out = 2 * x_out_sa * proj_query_pr + 2 * x_out_ca * proj_key_pr
        x_out = self.conv_x2(x_out)
        return x_out




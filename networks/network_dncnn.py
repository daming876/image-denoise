import torch.nn as nn
# import models.basicblock as B
import networks.basicblock as B
from torch.autograd import Variable
import functions

"""
# --------------------------------------------
# DnCNN (20 conv layers)
# FDnCNN (20 conv layers)
# IRCNN (7 conv layers)
# --------------------------------------------
# References:
@article{zhang2017beyond,
  title={Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising},
  author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
  journal={IEEE Transactions on Image Processing},
  volume={26},
  number={7},
  pages={3142--3155},
  year={2017},
  publisher={IEEE}
}
@article{zhang2018ffdnet,
  title={FFDNet: Toward a fast and flexible solution for CNN-based image denoising},
  author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
  journal={IEEE Transactions on Image Processing},
  volume={27},
  number={9},
  pages={4608--4622},
  year={2018},
  publisher={IEEE}
}
# --------------------------------------------
"""


# --------------------------------------------
# DnCNN原作者的DnCNN,不用了。------使用最下面的，从ffdnet修改来的（加了noise level map）
# --------------------------------------------
class DnCNNyuanshibuyong(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64, nb=17, act_mode='BR'):
        """
        # ------------------------------------
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        Batch normalization and residual learning are
        beneficial to Gaussian denoising (especially
        for a single noise level).
        The residual of a noisy image corrupted by additive white
        Gaussian noise (AWGN) follows a constant
        Gaussian distribution which stablizes batch
        normalization during training.
        # ------------------------------------
        """
        super(DnCNN, self).__init__()
        # R = ReLU（）； L = LeakyReLU（）
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'  # 提示激活函数可以为R，L，BR，BL，IR，IL
        bias = True

        # m_head = B.conv(in_nc+3, nc, mode='C' + act_mode[-1], bias=bias)
        m_head = B.conv(in_nc, nc, mode='C' + act_mode[-1], bias=bias)
        m_body = [B.conv(nc, nc, mode='C' + act_mode, bias=bias) for _ in range(nb - 2)]
        m_tail = B.conv(nc, out_nc, mode='C', bias=bias)

        self.model = B.sequential(m_head, *m_body, m_tail)

    # def forward(self, x):
    #     #print("-----------x.shape----------:",x.shape)      #torch.Size([128, 3, 44, 44])
    #     n = self.model(x)
    #     return x-n

    def forward(self, x, noise_sigma):
        # 给图片加噪声(将noise_map 和 downsampledfeatures拼接在一起)
        concat_noise_x = functions.dncnn_concatenate_input_noise_map( \
            x.data, noise_sigma.data)
        # print('-----concat_noise_x-1------', concat_noise_x.shape)      #torch.Size([128, 6, 44, 44])
        concat_noise_x = Variable(concat_noise_x)  # Variable()表示可变的参数
        # h_dncnn = self.intermediate_dncnn(concat_noise_x)
        print("--------concat_noise_x----", concat_noise_x.shape)
        n = self.model(concat_noise_x)
        print("--------dncnn-n----", n.shape)
        return concat_noise_x - n


# --------------------------------------------
# IRCNN denoiser
# --------------------------------------------
class IRCNN(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64):
        """
        # ------------------------------------
        denoiser of IRCNN
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        Batch normalization and residual learning are
        beneficial to Gaussian denoising (especially
        for a single noise level).
        The residual of a noisy image corrupted by additive white
        Gaussian noise (AWGN) follows a constant
        Gaussian distribution which stablizes batch
        normalization during training.
        # ------------------------------------
        """
        super(IRCNN, self).__init__()
        L = []
        L.append(
            nn.Conv2d(in_channels=in_nc, out_channels=nc, kernel_size=3, stride=1, padding=1, dilation=1, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=2, dilation=2, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=3, dilation=3, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=4, dilation=4, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=3, dilation=3, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=2, dilation=2, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(
            nn.Conv2d(in_channels=nc, out_channels=out_nc, kernel_size=3, stride=1, padding=1, dilation=1, bias=True))
        self.model = B.sequential(*L)

    def forward(self, x):
        n = self.model(x)
        return x - n


# --------------------------------------------
# FDnCNN
# --------------------------------------------
# Compared with DnCNN, FDnCNN has three modifications:
# 1) add noise level map as input
# 2) remove residual learning and BN
# 3) train with L1 loss
# may need more training time, but will not reduce the final PSNR too much.
# --------------------------------------------
class FDnCNN(nn.Module):
    def __init__(self, in_nc=2, out_nc=1, nc=64, nb=20, act_mode='R'):
        """
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        """
        super(FDnCNN, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True

        m_head = B.conv(in_nc, nc, mode='C' + act_mode[-1], bias=bias)
        m_body = [B.conv(nc, nc, mode='C' + act_mode, bias=bias) for _ in range(nb - 2)]
        m_tail = B.conv(nc, out_nc, mode='C', bias=bias)

        self.model = B.sequential(m_head, *m_body, m_tail)

    def forward(self, x):
        x = self.model(x)
        return x


class UpSampleFeatures(nn.Module):
    r"""Implements the last layer of FFDNet （上采样，实现 FFDNet 的最后一层）；DnCNN里不用这个函数
    """

    def __init__(self):
        super(UpSampleFeatures, self).__init__()

    def forward(self, x):
        return functions.upsamplefeatures(x)


class IntermediateDnCNN(nn.Module):
    r"""Implements the middel part of the FFDNet architecture, which
    is basically a DnCNN net，（基于DnCNN实现FFDNet的中间层）
    """

    def __init__(self, input_features, middle_features, num_conv_layers):
        super(IntermediateDnCNN, self).__init__()
        self.kernel_size = 3
        self.padding = 1  ##填充范围
        self.input_features = input_features
        self.num_conv_layers = num_conv_layers
        self.middle_features = middle_features
        # if self.input_features == 5:
        #     self.output_features = 4  # Grayscale image
        # elif self.input_features == 15:
        #     self.output_features = 12  # RGB image
        if self.input_features == 2:
            self.output_features = 1  # Grayscale image
        elif self.input_features == 6:
            self.output_features = 3  # RGB image
        else:
            raise Exception('Invalid number of input features')
        # 先设置网络层为空列表
        layers = []
        # 再追加网络层。   __init__下面的类属性值，如：self.kernel_size = 3，self.padding = 1 等在下面append里被调用
        layers.append(nn.Conv2d(in_channels=self.input_features, \
                                out_channels=self.middle_features, \
                                kernel_size=self.kernel_size, \
                                padding=self.padding, \
                                bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(self.num_conv_layers - 2):
            layers.append(nn.Conv2d(in_channels=self.middle_features, \
                                    out_channels=self.middle_features, \
                                    kernel_size=self.kernel_size, \
                                    padding=self.padding, \
                                    bias=False))
            layers.append(nn.BatchNorm2d(self.middle_features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=self.middle_features, \
                                out_channels=self.output_features, \
                                kernel_size=self.kernel_size, \
                                padding=self.padding, \
                                bias=False))
        self.itermediate_dncnn = nn.Sequential(*layers)  # 定义一个Sequential容器，式子右边函数参数传入上面layers里的网络层

    def forward(self, x):
        out = self.itermediate_dncnn(x)
        return out


class DnCNN(nn.Module):
    r"""Implements the FFDNet architecture，，，ffdnet修改来的（加了noise level map）
    """

    def __init__(self, num_input_channels):
        super(DnCNN, self).__init__()
        self.num_input_channels = num_input_channels
        if self.num_input_channels == 1:  # 灰度图
            # Grayscale image
            self.num_feature_maps = 64  # 特征图数
            self.num_conv_layers = 15  # 卷积15层
            #self.downsampled_channels = 5  # 下采样通道5
            self.downsampled_channels = 2  # 下采样通道5
            #self.output_features = 4
            self.output_features = 1
        elif self.num_input_channels == 3:  # 彩色图
            # RGB image
            self.num_feature_maps = 96  # 特征图数
            self.num_conv_layers = 12  # 卷积12层
            #self.downsampled_channels = 15  # 下采样通道15
            self.downsampled_channels = 6  # 下采样通道15
            #self.output_features = 12
            self.output_features = 3
        else:
            raise Exception('Invalid number of input features')

        self.intermediate_dncnn = IntermediateDnCNN( \
            input_features=self.downsampled_channels, \
            middle_features=self.num_feature_maps, \
            num_conv_layers=self.num_conv_layers)
        #self.upsamplefeatures = UpSampleFeatures()

    def forward(self, x, noise_sigma):
        # 给图片加噪声(将noise_map 和 downsampledfeatures拼接在一起)
            # print('-----x------', x.shape)	#torch.Size([128, 3, 44, 44])
            # print('-----noise_sigma------',noise_sigma.shape)		#torch.Size([128])
        concat_noise_x = functions.dncnn_concatenate_input_noise_map( \
            x.data, noise_sigma.data)
            # print('-----concat_noise_x-1------', concat_noise_x.shape)	#torch.Size([128, 15, 22, 22])
        concat_noise_x = Variable(concat_noise_x)  # Variable()表示可变的参数
            # print('-----concat_noise_x-2------', concat_noise_x.shape)		#torch.Size([128, 15, 22, 22])
        h_dncnn = self.intermediate_dncnn(concat_noise_x)
            # print('-----h_dncnn------', h_dncnn.shape)		# torch.Size([128, 12, 22, 22])
            #pred_noise = self.upsamplefeatures(h_dncnn)  # 上采样后网络s预测的噪声
            # print('-----pred_noise------', pred_noise.shape)	#torch.Size([128, 3, 44, 44])
        return h_dncnn



if __name__ == '__main__':
    from utils import utils_model
    import torch

    model1 = DnCNN(in_nc=1, out_nc=1, nc=64, nb=20, act_mode='BR')
    print(utils_model.describe_model(model1))

    model2 = FDnCNN(in_nc=2, out_nc=1, nc=64, nb=20, act_mode='R')
    print(utils_model.describe_model(model2))

    x = torch.randn((1, 1, 240, 240))
    x1 = model1(x)
    print(x1.shape)

    x = torch.randn((1, 2, 240, 240))
    x2 = model2(x)
    print(x2.shape)

    #  run models/network_dncnn.py


import torch.nn as nn
from torch.autograd import Variable
import functions
import cv2
import numpy as np
import os


class UpSampleFeatures(nn.Module):
    def __init__(self):
        super(UpSampleFeatures, self).__init__()

    def forward(self, x):
        return functions.upsamplefeatures(x)


class IntermediateDnCNN(nn.Module):


    def __init__(self, input_features, middle_features, num_conv_layers):
        super(IntermediateDnCNN, self).__init__()
        self.kernel_size = 3
        self.padding = 1
        self.input_features = input_features
        self.num_conv_layers = num_conv_layers
        self.middle_features = middle_features
        if self.input_features == 5:
            self.output_features = 4  # Grayscale image
        else:
            raise Exception('Invalid number of input features')

        layers = []
        # add network
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
        self.itermediate_dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.itermediate_dncnn(x)
        return out


class FFDNet(nn.Module):
    r"""Implements the FFDNet architecture
    """

    def __init__(self, num_input_channels):
        super(FFDNet, self).__init__()
        self.num_input_channels = num_input_channels
        if self.num_input_channels == 1:
            # Grayscale image
            self.num_feature_maps = 64
            self.num_conv_layers = 15
            self.downsampled_channels = 5
            self.output_features = 4
        else:
            raise Exception('Invalid number of input features')

        self.intermediate_dncnn = IntermediateDnCNN( \
            input_features=self.downsampled_channels, \
            middle_features=self.num_feature_maps, \
            num_conv_layers=self.num_conv_layers)
        self.upsamplefeatures = UpSampleFeatures()

    def forward(self, x, noise_sigma):

        # print('-----x------', x.shape)	#torch.Size([128, 3, 44, 44])
        # print('-----noise_sigma------',noise_sigma.shape)		#torch.Size([128])
        concat_noise_x = functions.concatenate_input_noise_map( \
            x.data, noise_sigma.data)



        concat_noise_x = Variable(concat_noise_x)
        # print('-----concat_noise_x-2------', concat_noise_x.shape)

        h_dncnn = self.intermediate_dncnn(concat_noise_x)
        # print('-----h_dncnn------', h_dncnn.shape)		# torch.Size([128, 12, 22, 22])
        pred_noise = self.upsamplefeatures(h_dncnn)

        return pred_noise

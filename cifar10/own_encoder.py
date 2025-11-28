import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np


# class AutoEncoder(nn.Module):
#     def __init__(self):
#         super(AutoEncoder, self).__init__()
#
#         self.encoder = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=3,  # input height
#                 out_channels=16,  # n_filters
#                 kernel_size=3,  # filter size
#                 stride=1,  # filter movement/step
#                 padding=1,
#             ),
#             nn.LeakyReLU(),  # activation
#             nn.Conv2d(
#                 in_channels=16,  # input height
#                 out_channels=32,  # n_filters
#                 kernel_size=3,  # filter size
#                 stride=1,  # filter movement/step
#                 padding=1,
#             ),
#             nn.LeakyReLU(),  # activation
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(
#                 in_channels=32,  # input height
#                 out_channels=32,  # n_filters
#                 kernel_size=5,  # filter size
#                 stride=1,  # filter movement/step
#                 padding=2,
#             ),
#             nn.LeakyReLU(),  # activation
#             nn.Conv2d(
#                 in_channels=32,  # input height
#                 out_channels=64,  # n_filters
#                 kernel_size=5,  # filter size
#                 stride=1,  # filter movement/step
#                 padding=2,
#             ),
#             nn.LeakyReLU(),  # activation
#             nn.MaxPool2d(kernel_size=2),
#         )
#
#         # Bottleneck
#         self.fc1 = nn.Linear(64 * 8 * 8, 10)  # [batch, 10]
#         self.fc2 = nn.Linear(10, 64 * 8 * 8)  # [batch, 256]
#
#
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(
#                 in_channels=64,  # input height
#                 out_channels=32,  # n_filters
#                 kernel_size=2,  # filter size
#                 stride=2,  # filter movement/step
#                 padding=0,
#             ),
#             nn.LeakyReLU(),  # activation
#             nn.Conv2d(
#                 in_channels=32,  # input height
#                 out_channels=32,  # n_filters
#                 kernel_size=5,  # filter size
#                 stride=1,  # filter movement/step
#                 padding=2,
#             ),
#             nn.LeakyReLU(),  # activation
#             nn.ConvTranspose2d(
#                 in_channels=32,  # input height
#                 out_channels=16,  # n_filters
#                 kernel_size=5,  # filter size
#                 stride=1,  # filter movement/step
#                 padding=2,
#             ),
#             nn.Conv2d(
#                 in_channels=16,  # input height
#                 out_channels=16,  # n_filters
#                 kernel_size=5,  # filter size
#                 stride=1,  # filter movement/step
#                 padding=2,
#             ),
#             nn.LeakyReLU(),  # activation
#             nn.ConvTranspose2d(
#                 in_channels=16,  # input height
#                 out_channels=16,  # n_filters
#                 kernel_size=2,  # filter size
#                 stride=2,  # filter movement/step
#                 padding=0,
#             ),
#             nn.LeakyReLU(),
#             nn.Conv2d(
#                 in_channels=16,  # input height
#                 out_channels=16,  # n_filters
#                 kernel_size=3,  # filter size
#                 stride=1,  # filter movement/step
#                 padding=1,
#             ),
#             nn.LeakyReLU(),  # activation
#             nn.ConvTranspose2d(
#                 in_channels=16,  # input height
#                 out_channels=3,  # n_filters
#                 kernel_size=5,  # filter size
#                 stride=1,  # filter movement/step
#                 padding=2,
#             ),
#             nn.Conv2d(
#                 in_channels=3,  # input height
#                 out_channels=3,  # n_filters
#                 kernel_size=3,  # filter size
#                 stride=1,  # filter movement/step
#                 padding=1,
#             ),
#             nn.Sigmoid(),  # activation
#         )
#
#     def forward(self, x):
#         x = self.encoder(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = x.view(x.size(0), 64, 8, 8)
#         decoded = self.decoder(x)
#         return decoded
#     def getFeature(self,x):
#         x = self.encoder(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         return x
#
import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexAutoencoder(nn.Module):
    def __init__(self):
        super(ComplexAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # [batch, 16, 16, 16]
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # [batch, 32, 8, 8]
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)  # [batch, 64, 2, 2]
        )

        # Bottleneck
        self.fc1 = nn.Linear(64 * 2 * 2, 10)  # [batch, 10]
        self.fc2 = nn.Linear(10, 64 * 2 * 2)  # [batch, 256]

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # 使用 Sigmoid 激活函数以输出在 [0, 1] 范围的像素值
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def getFeature(self,x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

    def getReverse(self,x):
        x = self.fc2(x)
        x = x.view(x.size(0), 64, 2, 2)
        x = self.decoder(x)
        return x

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=3,  # input height
                out_channels=16,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,
            ),
            nn.LeakyReLU(),  # activation
            nn.Conv2d(
                in_channels=16,  # input height
                out_channels=32,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,
            ),
            nn.LeakyReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=32,  # input height
                out_channels=32,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
            ),
            nn.LeakyReLU(),  # activation
            nn.Conv2d(
                in_channels=32,  # input height
                out_channels=64,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
            ),
            nn.LeakyReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64,  # input height
                out_channels=32,  # n_filters
                kernel_size=2,  # filter size
                stride=2,  # filter movement/step
                padding=0,
            ),
            nn.LeakyReLU(),  # activation
            nn.Conv2d(
                in_channels=32,  # input height
                out_channels=32,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
            ),
            nn.LeakyReLU(),  # activation
            nn.ConvTranspose2d(
                in_channels=32,  # input height
                out_channels=16,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
            ),
            nn.Conv2d(
                in_channels=16,  # input height
                out_channels=16,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
            ),
            nn.LeakyReLU(),  # activation
            nn.ConvTranspose2d(
                in_channels=16,  # input height
                out_channels=16,  # n_filters
                kernel_size=2,  # filter size
                stride=2,  # filter movement/step
                padding=0,
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=16,  # input height
                out_channels=16,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,
            ),
            nn.LeakyReLU(),  # activation
            nn.ConvTranspose2d(
                in_channels=16,  # input height
                out_channels=3,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
            ),
            nn.Conv2d(
                in_channels=3,  # input height
                out_channels=3,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,
            ),
            nn.ReLU(),  # activation
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_feature(self,x):
        encoded = self.encoder(x)
        return encoded

    def reverse_img(self,x):
        decoded = self.decoder(x)
        return decoded
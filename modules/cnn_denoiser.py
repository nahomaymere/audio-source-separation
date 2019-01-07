
import torch.nn as nn
from torch.nn.functional import relu
from torch.nn import Module, Linear
from torch.nn.functional import relu
import torch

class CNNDenoiser(Module):
    def __init__(self, input_dim):
        """The FNN enc and FNN dec of the Denoiser.

        :param input_dim: The input dimensionality.
        :type input_dim: int
        """
        super(CNNDenoiser, self).__init__()
        self._input_dim = input_dim
        kernel_conv = [5, 5]
        kernel_pool = [1, 2]

        strides_conv = [1, 4]
        strides_pool = [1, 2]

        padding_conv = [2, 1]
        padding_pool = [0, 0]

        self.conv1 = nn.Conv2d(16, 64, kernel_conv,strides_conv,padding_conv)
        self.conv2 = nn.Conv2d(64, 512, kernel_conv,strides_conv,padding_conv)
        self.conv3 = nn.Conv2d(512, 2049, kernel_conv,strides_conv,padding_conv)
        self.conv4 = nn.Conv2d(512, 2049, kernel_conv,strides_conv,padding_conv)

        self.pool = nn.MaxPool2d(kernel_pool,strides_pool,padding_pool)

        self.fc = Linear(2049*40*1, self._input_dim)






    def forward(self, v_j_filt_prime):
        """The forward pass.

        :param v_j_filt_prime: The output of the Masker.
        :type v_j_filt_prime: torch.autograd.variable.Variable
        :return: The output of the Denoiser
        :rtype: torch.autograd.variable.Variable
        """

        x = v_j_filt_prime.view(1,16,40,2049)
        x = (relu(self.conv1(x)))
        x = (relu(self.conv2(x)))
        x = (relu(self.conv3(x)))
        x = self.pool(x)
        x = x.view(16,40,2049)
        v_j_filt = x.mul(v_j_filt_prime)


        return v_j_filt


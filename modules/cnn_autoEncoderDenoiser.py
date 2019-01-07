
from torch.nn import Module, Linear
from torch.nn.init import xavier_normal
import torch.nn as nn

class CDAE(Module):
    def __init__(self, input_dim):
        """The FNN enc and FNN dec of the Denoiser.

        :param input_dim: The input dimensionality.
        :type input_dim: int
        """
        super(CDAE, self).__init__()
        self._input_dim = input_dim
        kernel_conv = [5, 5]
        kernel_pool = [1, 5]

        strides_conv = [1, 5]
        strides_pool = [1, 5]

        padding_conv = [2, 2]
        padding_pool = [0, 0]

        kernel_deConv = [5, 9]
        strides_deConv = [1, 5]
        padding_deConv = [2, 2]
        self.cnn_model= nn.Sequential(

            nn.Conv2d(input_dim, 410, kernel_conv,strides_conv,padding_conv),
            nn.ReLU(),
            nn.MaxPool2d((kernel_pool,strides_pool,padding_pool)),
            nn.ConvTranspose2d(82,410,kernel_deConv,strides_deConv,padding_deConv),
            nn.ReLU(),
            nn.ConvTranspose2d(410,2049,[5, 8],strides_deConv,padding_deConv),
            nn.ReLU()
        )


    def forward(self, v_j_filt_prime):
        """The forward pass.

        :param v_j_filt_prime: The output of the Masker.
        :type v_j_filt_prime: torch.autograd.variable.Variable
        :return: The output of the Denoiser
        :rtype: torch.autograd.variable.Variable
        """
        cnn_auto_encoder_output = self.cnn_model(v_j_filt_prime)
        v_j_filt = cnn_auto_encoder_output.mul(v_j_filt_prime)

        return v_j_filt

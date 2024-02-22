import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchcubicspline
import camb
from pixell import enmap, utils
from .utils import unnormalize_phi
from utils import get_colored_noise_2d
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ColoredPS(nn.Module):

    def __init__(self, norm_input_phi = 'compact', shape = (3, 256, 256)):
        super().__init__()
        shape = tuple(shape) if isinstance(shape, (list, tuple)) else (shape, )
        ndim = len(shape) - 1

        N = shape[-1]

        assert len(shape) == 3 # (C, H, W)
        assert shape[1] == shape[2] # (C, H, W)

        # Build an array of isotropic wavenumbers
        wn = torch.fft.fftfreq(N).reshape((N, 1)).to(device)

        S = torch.zeros((1,)+shape).to(device)
        for i in range(ndim):
            S += torch.moveaxis(wn, 0, i).pow(2)
        self.S = torch.sqrt(S)

        self.S[:,:, 0, 0] = 1
        self.norm_input_phi = norm_input_phi
    
    def forward(self, phi):
        '''Generates a power spectrum S(k) ~ k^alpha
        alpha: tensor of alpha of size (batch_size, phi_dim)
        '''
        phi = unnormalize_phi(phi, mode=self.norm_input_phi)
        
        S = self.S.repeat(phi.shape[0], 1, 1, 1)
        S = torch.pow(S, phi.reshape(-1, 1, 1, 1))
        S = S/torch.mean(S, dim=(2, 3), keepdim=True)
        return S
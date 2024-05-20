import torch
import numpy as np
import skimage.metrics as skmetrics


def psnr(img1, img2, data_range=1.0):
    '''img1: clean image
        img2: denoised image
        Assume images of size BxCxWxH'''
    if not img1.ndim == img2.ndim:
        raise ValueError('Input images must have the same number of dimensions.')
    if img1.ndim == 3 or img2.ndim == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    img1_np = img1.cpu().numpy()
    img2_np = img2.cpu().numpy()

    psnr_list = []
    for i in range(img1_np.shape[0]):
        psnr_list.append(skmetrics.peak_signal_noise_ratio(img1_np[i], img2_np[i], data_range=data_range))
    return torch.tensor(np.array(psnr_list)).to(img1.device)

def ssim(img1, img2):
    '''img1: clean image
       img2: denoised image
       Assume images of size BxCxWxH'''
    if not img1.ndim == img2.ndim:
        raise ValueError('Input images must have the same number of dimensions.')
    if img1.ndim == 3 or img2.ndim == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    img1_np = img1.cpu().numpy()
    img2_np = img2.cpu().numpy()

    ssim_list = []
    for i in range(img1.shape[0]):
        ssim_list.append(skmetrics.structural_similarity(img1_np[i], img2_np[i], data_range=1.0, channel_axis=0))
    return torch.tensor(np.array(ssim_list)).to(img1.device)

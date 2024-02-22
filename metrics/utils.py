### implement PSNR and SSIM metrics on 3 channel images

import torch
import torch.nn.functional as F
import numpy as np
from math import log10

import skimage.metrics as skmetrics
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from utils import power_spectrum_iso

def psnr(img1, img2, data_range=1.0):
    '''img1: clean image
        img2: denoised image
        Assume images of size BxCxWxH'''
    if not img1.dim() == img2.dim():
        raise ValueError('Input images must have the same dimensions.')
    if img1.dim() == 3 or img2.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    
    mse = torch.mean((img1.float() - img2.float()) ** 2, dim=(-1, -2, -3))
    return 10 * torch.log10((data_range**2)/mse) 

def psnr_skimage(img1, img2, data_range=1.0):
    '''img1: clean image
        img2: denoised image
        Assume images of size BxCxWxH'''
    if not img1.dim() == img2.dim():
        raise ValueError('Input images must have the same dimensions.')
    if img1.dim() == 3 or img2.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    img1_np = img1.cpu().numpy()
    img2_np = img2.cpu().numpy()

    psnr_list = []
    for i in range(img1_np.shape[0]):
        psnr_list.append(skmetrics.peak_signal_noise_ratio(img1_np[i], img2_np[i], data_range=data_range))
    return torch.tensor(np.array(psnr_list)).to(img1.device)

def psnr_uint8(img1, img2):
    '''img1: clean image
       img2: denoised image
       Assume images of size BxCxWxH'''
    img1 = (img1.float().clamp(0,1) * 255.0).round().int().float()
    img2 = (img2.float().clamp(0,1) * 255.0).round().int().float()
    if not img1.dim() == img2.dim():
        raise ValueError('Input images must have the same dimensions.')
    if img1.dim() == 3 or img2.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    mse = F.mse_loss(img1, img2, reduction='none').mean(dim=(-1,-2,-3))
    
    return 10 * torch.log10((255.0)**2 / mse) # = 10 * torch.log10(1/mse) and 1 is the max value of the image since they are normalized inside

def ssim(img1, img2):
    '''img1: clean image
       img2: denoised image
       Assume images of size BxCxWxH'''
    
    ssim = StructuralSimilarityIndexMeasure(reduction=None, data_range=1.0)
    return ssim(img2, img1)

def ssim_skimage(img1, img2):
    '''img1: clean image
       img2: denoised image
       Assume images of size BxCxWxH'''
    if not img1.dim() == img2.dim():
        raise ValueError('Input images must have the same dimensions.')
    if img1.dim() == 3 or img2.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    img1_np = img1.cpu().numpy()
    img2_np = img2.cpu().numpy()

    ssim_list = []
    for i in range(img1.shape[0]):
        ssim_list.append(skmetrics.structural_similarity(img1_np[i], img2_np[i], data_range=1.0, channel_axis=0))
    return torch.tensor(np.array(ssim_list)).to(img1.device)

def VIF(img1, img2):
    '''img1: clean image
       img2: denoised image
       Assume images of size BxCxWxH'''

    from torchmetrics.image import VisualInformationFidelity
    
    vif = VisualInformationFidelity().to(img1.device)
    return vif(img2, img1)

def FID_score(original, denoised, n_features_Inception = 64, normalized_images01 = True):
    if not original.dim() == denoised.dim():
        raise ValueError('Input images must have the same dimensions.')
    if original.dim() == 3:
        original = original.unsqueeze(0)
    if denoised.dim() == 3:
        denoised = denoised.unsqueeze(0)
    fid = FrechetInceptionDistance(feature=n_features_Inception, normalize = normalized_images01).to(original.device)
    fid.update(original.clamp(0, 1.0), real=True)
    fid.update(denoised.clamp(0, 1.0), real=False)
    fid_score = fid.compute()
    fid.reset()
    return fid_score

def integrated_error_PS(image, denoised_image):
    assert image.shape == denoised_image.shape
    assert image.dim() == 4
    image_np = image.cpu().numpy()
    denoised_image_np = denoised_image.cpu().numpy()
    bins = np.linspace(0, np.pi, 100)
    relative_error = []
    for i in range(image.shape[0]):
        rel = 0
        for j in range(image.shape[1]):
            bins, ps_mean, ps_std = power_spectrum_iso(image_np[i,j], bins=bins)
            bins, ps_mean_denoised, ps_std = power_spectrum_iso(denoised_image_np[i,j], bins=bins)
            rel += np.linalg.norm((ps_mean - ps_mean_denoised) /ps_mean)
        relative_error.append(rel / image.shape[1])
    return torch.tensor(np.array(relative_error)).to(image.device)

# #SSIM from DnCNN
# import cv2
# def calculate_ssim(img1, img2, border=0):
#     #The convention here is channel last
#     '''calculate SSIM
#     the same outputs as MATLAB's
#     img1, img2: [0, 255]
#     '''
#     #img1 = img1.squeeze()
#     #img2 = img2.squeeze()
#     if not img1.shape == img2.shape:
#         raise ValueError('Input images must have the same dimensions.')
#     h, w = img1.shape[:2]
#     img1 = img1[border:h-border, border:w-border]
#     img2 = img2[border:h-border, border:w-border]

#     if img1.ndim == 2:
#         return ssim(img1, img2)
#     elif img1.ndim == 3:
#         if img1.shape[2] == 3:
#             ssims = []
#             for i in range(3):
#                 ssims.append(ssim(img1[:,:,i], img2[:,:,i]))
#             return np.array(ssims).mean()
#         elif img1.shape[2] == 1:
#             return ssim(np.squeeze(img1), np.squeeze(img2))
#     else:
#         raise ValueError('Wrong input image dimensions.')


# def ssim(img1, img2):
#     C1 = (0.01 * 255)**2
#     C2 = (0.03 * 255)**2

#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)
#     kernel = cv2.getGaussianKernel(11, 1.5)
#     window = np.outer(kernel, kernel.transpose())

#     mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
#     mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
#     mu1_sq = mu1**2
#     mu2_sq = mu2**2
#     mu1_mu2 = mu1 * mu2
#     sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
#     sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
#     sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
#                                                             (sigma1_sq + sigma2_sq + C2))
#     return ssim_map.mean()

# def ms_ssim(img1, img2):
#     '''img1: clean image
#        img2: denoised image
#        Assume images of size BxCxWxH'''
    
#     ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(reduction='none')
#     return ms_ssim(img1, img2)



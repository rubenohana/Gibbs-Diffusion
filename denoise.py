import os, sys
import numpy as np
import torch
import bm3d
import pickle
from torch.utils.data import DataLoader, RandomSampler
from gdiff_utils.data import GDiff_dataset
from tqdm import tqdm
from gibbs_diffusion.model import load_model
from gdiff_utils.utils import get_colored_noise_2d

from metrics.utils import integrated_error_PS, FID_score, psnr_skimage, ssim_skimage

from gdiff_utils.network_dncnn import DnCNN as net

device = "cuda" if torch.cuda.is_available() else "cpu"
save_images = True
output_name = "output/denoising_imagenet_results_v4_script_high_noise.pkl"

#
# Load models
#

# DDPM
model_gdiff = load_model(diffusion_steps=10000, device=device)
model_gdiff.eval();

# DnCNN, downloaded from https://github.com/cszn/KAIR/tree/master/model_zoo
model_path = 'model_checkpoints/dncnn/dncnn_color_blind.pth'
model_dncnn = net(in_nc=3, out_nc=3, nc=64, nb=20, act_mode='R')
model_dncnn.load_state_dict(torch.load(model_path), strict=True)
model_dncnn.eval();
for k, v in model_dncnn.named_parameters():
    v.requires_grad = False
model_dncnn = model_dncnn.to(device)

#
# Sigmas and phis
#

sigmas = [0.2]
sigmas = [model_gdiff.get_closest_timestep(torch.tensor([s]).to(device), ret_sigma=True)[1].item() for s in sigmas]
sigmas_uint8 = [int(s*255) for s in sigmas]
phis = [-1, 0, 1]
print("Sigmas: ", sigmas)
print("Sigmas (in uint8 unit)", sigmas_uint8)
print("phis: ", phis)

#
# Dataset and batch size
#

num_samples = 50
batch_size = 25

dataset = GDiff_dataset(dataset="cbsd68")

sampler = RandomSampler(dataset, replacement=False, num_samples=num_samples)
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, sampler=sampler)

algos = ["ddpm_blind", "ddpm_pmean", "bm3d", "dncnn"]
metrics = ["psnr", "ssim", "IPS"] 

res_dict = {}

for algo in algos:
    res_dict[algo] = {}
    for metric in metrics:
        res_dict[algo][metric] = torch.zeros(len(sigmas), len(phis), num_samples)
    if save_images:
        res_dict[algo]["x_denoised"] = torch.zeros(len(sigmas), len(phis), num_samples, 3, 256, 256)
if save_images:
    res_dict["data"] = {}
    res_dict["data"]["y"] = torch.zeros(len(sigmas), len(phis), num_samples, 3, 256, 256)
    res_dict["data"]["x"] = torch.zeros(len(sigmas), len(phis), num_samples, 3, 256, 256)

for sigma_idx, sigma in enumerate(sigmas):
    for phi_idx, phi in tqdm(enumerate(phis)):
        print("Sigma: ", sigma)
        print("phi: ", phi)

        for x_id, (x, _) in enumerate(dataloader):
            print(x_id, x.shape)
            x = x.to(device)

            # Make test data and auxilary variables
            sigma_timestep = model_gdiff.get_closest_timestep(torch.tensor([sigma]).to(device)).item()
            alpha_bar_t = model_gdiff.alpha_bar_t[sigma_timestep].reshape(-1, 1, 1, 1).to(device)
            eps, psd =  get_colored_noise_2d(x.shape, phi, ret_psd=True)
            eps = torch.sqrt(1 - alpha_bar_t)/torch.sqrt(alpha_bar_t) * eps.to(device)
            y = x + eps # Noisy image
            psd = (torch.sqrt(1 - alpha_bar_t)/torch.sqrt(alpha_bar_t))**2 * psd.to(device) * x.shape[-1] * x.shape[-2]
            yt = torch.sqrt(alpha_bar_t) * y # Noisy image normalized for the diffusion model

            if save_images:
                res_dict["data"]["y"][sigma_idx, phi_idx, x_id*batch_size:x_id*batch_size+y.shape[0]] = y
                res_dict["data"]["x"][sigma_idx, phi_idx, x_id*batch_size:x_id*batch_size+x.shape[0]] = x

            x_denoised_pmean = None # For ddpm_mean if ddpm_blind was called before (keep in memory for the right iteration)
            for algo in algos:
                #
                # Denoising
                #
                print("Denoising with ", algo)
                x_denoised = None
                if algo == "ddpm":
                    x_denoised = model_gdiff.denoise_samples_batch_time(yt, torch.tensor(sigma_timestep).unsqueeze(0).repeat(yt.shape[0]), phi_ps=phi)
                elif algo == "ddpm_pmean":
                    if "ddpm_blind" in algos and x_denoised_pmean is not None:
                        print("Using pmean computed on ddpm_blind samples")
                        x_denoised = x_denoised_pmean.clone()
                        x_denoised_pmean = None
                    else:
                        repeat = 20
                        x_denoised = model_gdiff.denoise_samples_batch_time(yt.repeat(repeat, 1, 1, 1), torch.tensor(sigma_timestep).unsqueeze(0).repeat(yt.shape[0]*repeat), phi_ps=phi)
                        x_denoised = x_denoised.reshape(repeat, yt.shape[0], 3, yt.shape[2], yt.shape[3]).mean(dim=0)
                elif algo == "ddpm_blind":
                    num_chains_per_sample = 5
                    phi_all, x_all = model_gdiff.blind_denoising(y, yt, num_chains_per_sample=num_chains_per_sample)
                    x_denoised = x_all[:yt.shape[0], -1] # We take the last samples of the first series of chains
                    x_denoised_pmean = x_all[:, -10:].reshape(num_chains_per_sample, -1, 10, 3, 256, 256).mean(dim=(0, 2))
                elif algo == "bm3d":
                    x_denoised_list = []
                    for i in range(y.shape[0]):
                        x_denoised_list.append(bm3d.bm3d_rgb(y[i].cpu().permute(1,2,0).numpy(),
                                                             psd[i, 0].cpu().numpy()))
                    x_denoised = torch.tensor(np.array(x_denoised_list)).permute(0, 3, 1, 2).to(device)
                elif algo == "dncnn":
                    x_denoised = model_dncnn(y)
                else:
                    raise ValueError("Unknown algorithm")
                if save_images:
                    res_dict[algo]["x_denoised"][sigma_idx, phi_idx, x_id*batch_size:x_id*batch_size+x.shape[0]] = x_denoised

                # Compute metrics
                for metric in metrics:
                    score = None
                    if metric == "psnr":
                        score = psnr_skimage(x, x_denoised)
                    elif metric == "ssim":
                        score = ssim_skimage(x, x_denoised)
                    elif metric == "fid": # TOFIX: infs or NaN
                        score = FID_score(x, x_denoised)
                    elif metric == "IPS":
                        score = integrated_error_PS(x, x_denoised)
                    else:
                        raise ValueError("Unknown metric")
                    res_dict[algo][metric][sigma_idx, phi_idx, x_id*batch_size:x_id*batch_size+score.shape[0]] = score

# Save results
with open(output_name, "wb") as f:
    pickle.dump(res_dict, f)

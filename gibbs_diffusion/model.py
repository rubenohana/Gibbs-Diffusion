import torch
import torch.nn as nn
import lightning as pl
import math
import os, glob
from .modules import DoubleConv, Down, Up, OutConv, phi_embedding, SAWrapper
from gdiff_utils.utils import get_colored_noise_2d
import gdiff_utils.utils_hmc as iut
import sys
sys.path.append('../')
from gdiff_utils.data import get_noise_level_estimate
from tqdm import tqdm
from gdiff_utils.hmc import HMC


class GDiff(pl.LightningModule):
    def __init__(self,
                 in_size, 
                 diffusion_steps, 
                 img_depth = 3, 
                 lr = 2e-4,
                 weight_decay = 0):
        """
        Gibbs-Diffusion model. 
        The LightningModule allows for Data Parallel training easily. 
        Training is done on Imagenet for 100 epochs, and takes about 40 hours a single node of 8 H100s GPUs. 
        See train.py for details.
        """

        super().__init__()
        self.diffusion_steps = diffusion_steps
        self.beta_small = 0.1 / self.diffusion_steps
        self.beta_large = 20 / self.diffusion_steps
        self.timesteps = torch.arange(0, self.diffusion_steps)
        self.beta_t = self.beta_small + (self.timesteps / self.diffusion_steps) * (self.beta_large - self.beta_small)
        self.alpha_t = 1 - self.beta_t
        self.alpha_bar_t = torch.cumprod(self.alpha_t, dim=0) 

        self.in_size = in_size
        H = math.sqrt(in_size)
        self.img_depth = img_depth
        
        phi_dim = 1 #number of dimensions of phi
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()

        self.inc = DoubleConv(self.img_depth, 64) #img_depth is the number of channels
        nc_1 = 128
        self.down1 = Down(64, nc_1) #double the number of channels, but H/2 and W/2
        self.alpha_embed1 = phi_embedding(phi_dim, in_dim = 100, out_dim=nc_1)
        nc_2 = 256
        self.down2 = Down(128, nc_2)
        self.alpha_embed2 = phi_embedding(phi_dim, in_dim = 100, out_dim=nc_2)
        nc_3 = 512 
        self.down3 = Down(256, nc_3)
        self.alpha_embed3 = phi_embedding(phi_dim, in_dim = 100, out_dim=nc_3)
        self.sa3 = SAWrapper(512, int(H/8))
        nc_4 = 1024
        self.down4 = Down(512, nc_4)
        self.alpha_embed4 = phi_embedding(phi_dim, in_dim = 100, out_dim=nc_4)
        self.sa4 = SAWrapper(1024, int(H/16)) #if H = 256: bootleneck dim = 16
        
        self.bottleneck = DoubleConv(1024, 1024)
        self.att_bottleneck = SAWrapper(1024, int(H/16))
        nc_5 = 512
        self.up1 = Up(1024, nc_5)
        self.alpha_embed5 = phi_embedding(phi_dim, in_dim = 100, out_dim=nc_5)
        nc_6 = 256
        self.up2 = Up(512, 256)
        self.alpha_embed6 = phi_embedding(phi_dim, in_dim = 100, out_dim=nc_6)
        nc_7 = 128
        self.up3 = Up(256, 128)
        self.alpha_embed7 = phi_embedding(phi_dim, in_dim = 100, out_dim=nc_7)
        nc_8 = 64
        self.up4 = Up(128, 64)
        self.alpha_embed8 = phi_embedding(phi_dim, in_dim = 100, out_dim=nc_8)
        self.outc = OutConv(64, self.img_depth)

    def pos_encoding(self, t, channels, embed_size):
        """
        Positinal encoding of time, as in the original transformer paper. 
        """
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc.view(-1, channels, 1, 1)

    def forward(self, x, t, phi_ps=None):
        """
        The model is a U-Net with added positional encodings, embedding of \varphi  and self-attention layers. The total number of parameters is ~70M.
        """
        if phi_ps is None:
            phi_ps = torch.zeros(x.shape[0],1).to(self.device).float()
            print('Careful, no alpha was given, set to 0')
        elif isinstance(phi_ps, int) or isinstance(phi_ps, float):
            phi_ps = torch.tensor(phi_ps).reshape(-1,1).to(self.device).float()
        else:
            pass
        bs, n_channels, H, W = x.shape 
        x1 = self.inc(x) #dim = 256, n_channel = 64
        x2 = self.down1(x1) + self.pos_encoding(t, 128, int(H/2)) + self.alpha_embed1(phi_ps).unsqueeze(-1).unsqueeze(-1) #dim = 128, n_channel = 128, unsqueeze for broadcasting on HxW
        x3 = self.down2(x2) + self.pos_encoding(t, 256, int(H/4)) + self.alpha_embed2(phi_ps).unsqueeze(-1).unsqueeze(-1) #dim = 64, n_channel = 256
        x4 = self.down3(x3) + self.pos_encoding(t, 512, int(H/8)) + self.alpha_embed3(phi_ps).unsqueeze(-1).unsqueeze(-1)#dim = 32, n_channel = 512
        x4 = self.sa3(x4) #dim = 32, n_channel = 512 
        x5 = self.down4(x4) + self.pos_encoding(t, 1024, int(H/16)) + self.alpha_embed4(phi_ps).unsqueeze(-1).unsqueeze(-1) #dim = 16, n_channel = 1024
        x5 = self.sa4(x5) #dim = 16, n_channel = 1024 
        x_bottleneck = self.bottleneck(x5) + self.pos_encoding(t, 1024, int(H/16)) #dim = 16, n_channel = 1024 
        x_bottleneck = self.att_bottleneck(x_bottleneck) #dim = 16, n_channel = 1024
        x = self.up1(x_bottleneck, x4) + self.pos_encoding(t, 512, int(H/8)) + self.alpha_embed5(phi_ps).unsqueeze(-1).unsqueeze(-1) #dim = 32, n_channel = 512
        x = self.up2(x, x3) + self.pos_encoding(t, 256, int(H/4)) + self.alpha_embed6(phi_ps).unsqueeze(-1).unsqueeze(-1) #dim = 64, n_channel = 256
        x = self.up3(x, x2) + self.pos_encoding(t, 128, int(H/2)) + self.alpha_embed7(phi_ps).unsqueeze(-1).unsqueeze(-1) #dim = 128, n_channel = 128
        x = self.up4(x, x1) + self.pos_encoding(t, 64, int(H)) + self.alpha_embed8(phi_ps).unsqueeze(-1).unsqueeze(-1)#dim = 256, n_channel = 64
        output = self.outc(x) #dim = 256, n_channel = 3

        return output

    def get_loss(self, batch, batch_idx, phi_ps = None):
        """
        Corresponds to Algorithm 1 from (Ho et al., 2020), but with colored noise.
        """
        #phi should be a tensor of the size of the batch_size, we want a phi different for each batch element
        #if batch is a list (the case of ImageFolder for ImageNet): take the first element, otherwise take batch:
        if isinstance(batch, list):
            batch = batch[0]

        bs = batch.shape[0]
        if phi_ps is None:
            #sample phi_ps between -1 and 1
            phi_ps = torch.rand(bs, 1, device=self.device)*2 - 1

        #if phi is a scalar, cast to batch dimension. For training on a single phi.
        if isinstance(phi_ps, float) or isinstance(phi_ps, int):
            phi_ps = phi_ps * torch.ones(bs,1).to(self.device) 

        ts = torch.randint(0, self.diffusion_steps, (bs,1)).float().to(self.device)
        noise_imgs = []

        epsilons = get_colored_noise_2d(batch.shape, phi_ps, device= self.device) #B x C x H x W

        a_hat = self.alpha_bar_t[ts.squeeze(-1).int().cpu()].reshape(-1, 1, 1, 1).to(self.device)
        noise_imgs = torch.sqrt(a_hat) * batch + torch.sqrt(1 - a_hat) * epsilons

        e_hat = self.forward(noise_imgs, ts, phi_ps=phi_ps)
        loss = nn.functional.mse_loss(e_hat, epsilons)

        return loss

    def denoise_1step(self, x, t, phi_ps = None):
        """
        Corresponds to the inner loop of Algorithm 2 from (Ho et al., 2020).
        """
        #phi should be a tensor of the size of the batch_size, we want a different phi for each batch element
        if phi_ps is None:
            # If no phi_ps is given, assume it's white noise, i.e. phi_ps = 0
            phi_ps = torch.zeros(x.shape[0],1, device=self.device).float()
        
        #if phi is a scalar, cast to batch dimension
        if isinstance(phi_ps, int) or isinstance(phi_ps, float):
            phi_ps = phi_ps * torch.ones(x.shape[0],1, device=self.device).float() 
        
        else: 
            phi_ps = phi_ps.to(self.device).float()
        
        with torch.no_grad():
            if t > 1:
                z = get_colored_noise_2d(x.shape, phi_ps, device= self.device)
            else:
                z = 0
            e_hat = self.forward(x, t.view(1, 1).repeat(x.shape[0], 1), phi_ps=phi_ps)
            pre_scale = 1 / math.sqrt(self.alpha_t[t])
            e_scale = (self.beta_t[t]) / math.sqrt(1 - self.alpha_bar_t[t])
            post_sigma = math.sqrt(self.beta_t[t]) * z
            x = pre_scale * (x - e_scale * e_hat) + post_sigma
            return x
            
    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch, batch_idx)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.get_loss(batch, batch_idx)
        self.log("val/loss", loss)
        return

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    
    def blind_denoising(self, y, yt,
                         norm_phi_mode='compact',
                         num_chains_per_sample=1,
                         n_it_gibbs=30,
                         n_it_burnin=10,
                         sigma_min=0.04,
                         sigma_max=0.3,
                         return_chains=True):
        '''Gibbs-Diffusion: performs blind denoising with a Gibbs sampler alternating between the diffusion model step returning a sample from p(x|y,phi) and the HMC step that return estimates of parameters from p(phi|x,y).'''

        num_samples = y.shape[0]
        ps_model = iut.ColoredPS(norm_input_phi=norm_phi_mode)
        
        # Prior, likelihood, and posterior functions
        sample_phi_prior = lambda n: iut.sample_prior_phi(n, norm=norm_phi_mode, device=self.device) # Sample uniformly in [-1, 1]
        log_likelihood = lambda phi, x: iut.log_likelihood_eps_phi_sigma(phi[...,:1], phi[...,1:], x, ps_model)
        log_prior = lambda phi: iut.log_prior_phi_sigma(phi[...,:1], phi[...,1], sigma_min, sigma_max, norm=norm_phi_mode)
        log_posterior = lambda phi, x: log_likelihood(phi, x) + log_prior(phi) #  Log posterior (not normalized by the evidence).

        # Bounds and collision management
        phi_min_norm, phi_max_norm = iut.get_phi_bounds(device=self.device) #change to work in [-1,1]
        phi_min_norm, phi_max_norm = iut.normalize_phi(phi_min_norm, mode=norm_phi_mode), iut.normalize_phi(phi_max_norm, mode=norm_phi_mode) #change to work in [-1,1]
        sigma_min_tensor = torch.tensor([sigma_min]).to(self.device)
        sigma_max_tensor = torch.tensor([sigma_max]).to(self.device)
        phi_min_norm = torch.concatenate((phi_min_norm, sigma_min_tensor)) # Add sigma_min to the list of parameter bounds
        phi_max_norm = torch.concatenate((phi_max_norm, sigma_max_tensor)) # Add sigma_max to the list of parameter bounds

        def collision_manager(q, p, p_nxt):
            p_ret = p_nxt
            for i in range(2):
                crossed_min_boundary = q[..., i] < phi_min_norm[i]
                crossed_max_boundary = q[..., i] > phi_max_norm[i]
                # Reflecting boundary conditions
                p_ret[..., i][crossed_min_boundary] = -p[..., i][crossed_min_boundary]
                p_ret[..., i][crossed_max_boundary] = -p[..., i][crossed_max_boundary]
            return p_ret

        print("Normalized prior bounds are:", phi_min_norm, phi_max_norm)

        # Inference on the noise level \sigma and the parameters \varphi of the covariance of the noise

        # Repeat the data for each chain
        y_batch = y.repeat(num_chains_per_sample, 1, 1, 1)
        yt_batch = yt.repeat(num_chains_per_sample, 1, 1, 1)

        # Initalization
        phi_0 = sample_phi_prior(num_samples*num_chains_per_sample)
        sigma_0 = get_noise_level_estimate(y_batch, sigma_min, sigma_max).unsqueeze(-1) # sigma_0 is initalized with a rough estimate of the noise level
        phi_0 = torch.concatenate((phi_0, sigma_0), dim=-1) # Concatenate phi and sigma

        # Gibbs sampling
        phi_all, x_all = [], []
        phi_all.append(phi_0)
        phi_k = phi_0
        step_size, inv_mass_matrix = None, None
        for n in tqdm(range(n_it_gibbs + n_it_burnin)):

            # Diffusion step
            timesteps = self.get_closest_timestep(phi_k[:, 1])
            x_k = self.denoise_samples_batch_time(yt_batch, timesteps, phi_ps=iut.unnormalize_phi(phi_k[:, :1], mode=norm_phi_mode))
            eps_k = (y_batch - x_k)
            
            # HMC step
            log_prob = lambda phi: log_posterior(phi, eps_k)
            def log_prob_grad(phi):
                """ Compute the log posterior and its gradient."""
                phib = phi.clone()
                phib.requires_grad_(True)
                log_prob_val = log_posterior(phib, eps_k)
                grad_log_prob = torch.autograd.grad(log_prob_val, phib, grad_outputs=torch.ones_like(log_prob_val))[0]
                return log_prob_val.detach(), grad_log_prob
            
            if n == 0:
                hmc = HMC(log_prob, log_prob_and_grad=log_prob_grad)
                hmc.set_collision_fn(collision_manager)

                phi_k = hmc.sample(phi_k, nsamples=1, burnin=10, step_size=1e-6, nleap=(5, 15), epsadapt=300, verbose=False, ret_side_quantities=False)[:, 0, :].detach()
                step_size = hmc.step_size
                inv_mass_matrix = hmc.mass_matrix_inv
            else:
                hmc = HMC(log_prob, log_prob_and_grad=log_prob_grad)
                hmc.set_collision_fn(collision_manager)
                hmc.set_inv_mass_matrix(inv_mass_matrix, batch_dim=True)
                phi_k = hmc.sample(phi_k, nsamples=1, burnin=10, step_size=step_size, nleap=(5, 15), epsadapt=0, verbose=False)[:, 0, :].detach()

            # Save samples
            phi_all.append(phi_k)
            x_all.append(x_k.detach().cpu())

        phi_all = torch.stack(phi_all, dim=1)
        x_all = torch.stack(x_all, dim=1)
        if return_chains:
            return phi_all, x_all
        else:
            x_all
    
    def blind_denoising_pmean(self,y, yt,
                         norm_phi_mode='compact',
                         num_chains_per_sample=5,
                         n_it_gibbs=30,
                         n_it_burnin=10, 
                         avg_pmean = 10,
                         return_chains=True):
        '''Performs blind denoising with the posterior mean estimator.'''
        if return_chains:
            phi_all, x_all = self.blind_denoising(y, yt, norm_phi_mode=norm_phi_mode, num_chains_per_sample=num_chains_per_sample, n_it_gibbs=n_it_gibbs, n_it_burnin=n_it_burnin, return_chains=return_chains)
        else:
            x_all = self.blind_denoising(y, yt, norm_phi_mode=norm_phi_mode, num_chains_per_sample=num_chains_per_sample, n_it_gibbs=n_it_gibbs, n_it_burnin=n_it_burnin, return_chains=return_chains)
        x_denoised_pmean = x_all[:, -avg_pmean:].reshape(num_chains_per_sample, -1, 10, self.img_depth, 256, 256).mean(dim=(0, 2))
        if return_chains:
            return phi_all, x_denoised_pmean
        else:
            return x_denoised_pmean


    def get_closest_timestep(self, noise_level, ret_sigma=False):
        """
        Returns the closest timestep to the given noise level. If ret_sigma is True, also returns the noise level corresponding to the closest timestep.
        """
        

        alpha_bar_t = self.alpha_bar_t.to(noise_level.device)
        all_noise_levels = torch.sqrt((1-alpha_bar_t)/alpha_bar_t).reshape(-1, 1).repeat(1, noise_level.shape[0])
        closest_timestep = torch.argmin(torch.abs(all_noise_levels - noise_level), dim=0)
        if ret_sigma:
            return closest_timestep, all_noise_levels[closest_timestep, 0]
        else:
            return closest_timestep

    def denoise_samples_batch_time(self, noisy_batch, timesteps, batch_origin=None, return_sample = False, phi_ps = None):
        """
        Denoises a batch of images for a given number of timesteps (which can be different across the batch).
        """
        max_timesteps = torch.max(timesteps)
        mask = torch.ones(noisy_batch.shape[0], max_timesteps+1).to(self.device)
        for i in range(noisy_batch.shape[0]):
            mask[i, timesteps[i]+1:] = 0

        for t in range(max_timesteps, 0, -1):
            noisy_batch = self.denoise_1step(noisy_batch, torch.tensor(t).cuda(), phi_ps) * (mask[:, t]).reshape(-1,1,1,1) + noisy_batch * (1 - mask[:, t]).reshape(-1,1,1,1)
        if batch_origin is None:
            return noisy_batch
        else:
            if return_sample:
                return torch.mean(torch.norm(noisy_batch-batch_origin, dim = (-2,-1))), noisy_batch
            else:
                return torch.mean(torch.norm(noisy_batch-batch_origin, dim = (-2,-1))), None
    
    
def load_model(diffusion_steps=10000,
               in_size_image=256*256,
               n_channels=3,
               root_dir=None,
               device='cuda'):

    if root_dir is None:
        root_dir = "model_checkpoints/"
    model_dir = os.path.join(root_dir, f"GDiff_{diffusion_steps}steps/")
    ckpt_dir = glob.glob(model_dir + "*.ckpt")[-1]
    model = GDiff.load_from_checkpoint(ckpt_dir, 
                                        in_size=in_size_image, 
                                        diffusion_steps = diffusion_steps, 
                                        img_depth=n_channels) 
    model.to(device)
    return model




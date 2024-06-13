import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
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

def power_spectrum(
    x: torch.Tensor,
    bins: torch.Tensor = None,
    fourier_input: bool = False,
    sample_spacing: float = 1.0,
    return_counts: bool = False,
) -> tuple:
    """
    Compute the isotropic power spectrum of input tensor x.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of images.
    bins : torch.Tensor, optional
        Array of bin edges. If None, we use a default binning. The default is None.
    fourier_input : bool, optional
        If True, x is assumed to be the Fourier transform of the input data. The default is False.
    sample_spacing : float, optional
        Sample spacing. The default is 1.0.
    return_counts : bool, optional
        Return counts per bin. The default is False.

    Returns
    -------
    bins : torch.Tensor
        Array of bin edges.
    ps_mean : torch.Tensor
        Power spectrum (estimated as a mean over bins).
    ps_std : torch.Tensor
        Standard deviation of the power spectrum (estimated as a standard deviation over bins).
    counts : torch.Tensor, optional
        Counts per bin if return_counts=True.
    """
    spatial_dims = (-2, -1) # We are dealing with images
    spatial_shape = tuple(x.shape[dim] for dim in spatial_dims)
    ndim = len(spatial_dims)
    device = x.device

    # Compute array of isotropic wavenumbers
    wn_iso = torch.zeros(spatial_shape).to(device)
    for i in range(ndim):
        wn = (
            (2 * np.pi * torch.fft.fftfreq(spatial_shape[i], d=sample_spacing))
            .reshape((spatial_shape[i],) + (1,) * (ndim - 1))
            .to(device)
        )
        wn_iso += torch.moveaxis(wn, 0, i) ** 2
    wn_iso = torch.sqrt(wn_iso).flatten()

    if bins is None:
        bins = torch.linspace(
            0, wn_iso.max().item() + 1e-6, int(np.sqrt(min(spatial_shape)))
        ).to(device)  # Default binning
    indices = torch.bucketize(wn_iso, bins, right=True) - 1
    indices_mask = F.one_hot(indices, num_classes=len(bins))
    counts = torch.sum(indices_mask, dim=0)

    if not fourier_input:
        x = torch.fft.fftn(x, dim=spatial_dims)
    fx2 = torch.abs(x) ** 2
    fx2 = fx2.reshape(
        x.shape[: spatial_dims[0]] + (-1,)
    )  # Flatten spatial dimensions

    # Compute power spectrum
    ps_mean = torch.sum(fx2.unsqueeze(-1) * indices_mask, dim=-2) / (
        counts + 1e-7
    )
    ps_std = torch.sqrt(
        torch.sum(
            (fx2.unsqueeze(-1) - ps_mean.unsqueeze(-2)) ** 2
            * indices_mask,
            dim=-2,
        )
        / (counts + 1e-7)
    )

    # Discard the last bin (which has no upper limit)
    ps_mean = ps_mean[..., :-1]
    ps_std = ps_std[..., :-1]

    if return_counts:
        return bins, ps_mean, ps_std, counts
    else:
        return bins, ps_mean, ps_std

def plot_list_of_images(list_of_images, list_of_titles=None, figsize=None):
    fig, axs = plt.subplots(1, len(list_of_images), figsize=(5*len(list_of_images), 5) if figsize is None else figsize)
    for i, image in enumerate(list_of_images):
        if isinstance(image, torch.Tensor):
            assert image.ndim == 3
            x = image.permute(1, 2, 0).cpu().numpy()
        elif isinstance(image, np.ndarray):
            assert image.ndim == 3
            x = image.transpose(1, 2, 0)
        else:
            raise NotImplementedError
        axs[i].imshow(x)
        if list_of_titles is not None:
            axs[i].set_title(list_of_titles[i])
    fig.show()

def plot_power_spectrum(list_of_images, list_of_labels=None, figsize=(15, 4)):
    for image in list_of_images:
        assert image.ndim == 3
    tensor_of_images = torch.stack(list_of_images) if isinstance(list_of_images[0], torch.Tensor) \
        else torch.stack([torch.tensor(image) for image in list_of_images])
    bins = torch.linspace(0, np.pi, 100).to(tensor_of_images.device)
    bins_centers = (bins[1:] + bins[:-1]) / 2
    _, ps, _ = power_spectrum(tensor_of_images, bins=bins)

    fig, axs = plt.subplots(1, 3, figsize=figsize, sharex=True, sharey=True)
    for i in range(3):
        for j, image in enumerate(list_of_images):
            axs[i].plot(bins_centers.cpu().numpy(), ps[j, i].cpu().numpy(), label=list_of_labels[j] if list_of_labels is not None else None)
        axs[i].set_yscale('log')
        axs[i].set_xscale('log')
        axs[i].set_title(f'Channel {i}')
    axs[0].legend()
    fig.show()

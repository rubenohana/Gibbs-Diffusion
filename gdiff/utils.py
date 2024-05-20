import torch
import matplotlib.pyplot as plt
import numpy as np

    
def get_colored_noise_2d(shape, phi = 0, ret_psd=False, device = None):
    """
    Args:
        shape: (int tuple or torch.Size) shape of the image
        phi: (float or torch.Tensor of shape (B,1)) power spectrum exponent 
        ret_psd: (bool) if True, return the power spectrum
    Returns:
        noise: colored noise
        ps: power spectrum
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert len(shape) == 4 # (B, C, H, W)
    assert shape[2] == shape[3] # (B, C, H, W)
    if isinstance(phi, float) or isinstance(phi, int):
        phi = torch.tensor(phi).to(device).repeat(shape[0], 1)
    else:
        assert phi.shape == (shape[0], 1)
    N = shape[2]

    wn = torch.fft.fftfreq(N).reshape((N, 1)).to(device)
    S = torch.zeros((shape[0], shape[1], N, N)).to(device)
    for i in range(2): ## we are in 2D
        S += torch.moveaxis(wn, 0, i).pow(2)

    S.pow_(phi.reshape(-1, 1, 1, 1).to(device)/2)
    S[:, :, 0, 0] = 1.0
    S.div_(torch.mean(S, dim=(-1, -2), keepdim=True))  # Normalize S to keep std = 1

    X_white = torch.fft.fftn(torch.randn(shape).to(device), dim=(2,3))
    X_shaped = X_white * torch.sqrt(S)
    noises = torch.fft.ifftn(X_shaped, dim=(2,3)).real
    
    if ret_psd:
        return noises, S
    else:
        return noises
    

def _spectral_iso(data_sp, bins=None, sampling=1.0, return_counts=False):
    """
    Internal function.
    Check power_spectrum_iso for documentation details.
    Parameters
    ----------
    data_sp : TYPE
        DESCRIPTION.
    bins : TYPE, optional
        DESCRIPTION. The default is None.
    sampling : TYPE, optional
        DESCRIPTION. The default is 1.0.
    return_coutns : bool, optional
        Return counts per bin.
    Returns
    -------
    bins : TYPE
        DESCRIPTION.
    ps_mean : TYPE
        DESCRIPTION.
    ps_std : TYPE
        DESCRIPTION.
    counts : array, optional
        Return counts per bin return_counts=True.
    """
    N = data_sp.shape[0]
    ndim = data_sp.ndim
    # Build an array of isotropic wavenumbers making use of numpy broadcasting
    wn = (2 * np.pi * np.fft.fftfreq(N, d=sampling)).reshape((N,) + (1,) * (ndim - 1))
    wn_iso = np.zeros(data_sp.shape)
    for i in range(ndim):
        wn_iso += np.moveaxis(wn, 0, i) ** 2
    wn_iso = np.sqrt(wn_iso)
    # We do not need ND-arrays anymore
    wn_iso = wn_iso.ravel()
    data_sp = data_sp.ravel()
    # We compute associations between index and bins
    if bins is None:
        bins = np.sort(np.unique(wn_iso)) # Default binning
    index = np.digitize(wn_iso, bins) - 1
    # Stacking
    stacks = np.empty(len(bins), dtype=object)
    for i in range(len(bins)):
        stacks[i] = []
    for i in range(len(index)):
        if index[i] >= 0:
            stacks[index[i]].append(data_sp[i])
    counts = []
    # Computation for each bin of the mean power spectrum and standard deviations of the mean
    ps_mean = np.zeros(len(bins), dtype=data_sp.dtype) # Allow complex values (for cross-spectrum)
    ps_std = np.zeros(len(bins)) # If complex values, note that std first take the modulus
    for i in range(len(bins)):
        ps_mean[i] = np.mean(stacks[i])
        count = len(stacks[i])
        ps_std[i] = np.std(stacks[i]) / np.sqrt(count)
        counts.append(count)
    if return_counts:
        return bins, ps_mean, ps_std, np.array(counts)
    else:
        return bins, ps_mean, ps_std


def power_spectrum(data, data2=None, norm=None):
    """
    Compute the full power spectrum of input data.
    Parameters
    ----------
    data : array
        Input data.
    norm : str
        FFT normalization. Can be None or 'ortho'. The default is None.
    Returns
    -------
    None.
    """
    if data2 is None:
        result=np.absolute(np.fft.fftn(data, norm=norm))**2
    else:
        result=np.real(np.conjugate(np.fft.fftn(data, norm=norm))*np.fft.fftn(data2, norm=norm))
    return result


def power_spectrum_iso(data, data2=None, bins=None, sampling=1.0, norm=None, return_counts=False):
    """
    Compute the isotropic power spectrum of input data.
    bins parameter should be a list of bin edges defining:
    bins[0] <= bin 0 values < bins[1]
    bins[1] <= bin 1 values < bins[2]
                ...
    bins[N-2] <= bin N-2 values < bins[N-1]
    bins[N-1] <= bin N-1 values
    Note that the last bin has no superior limit.
    Parameters
    ----------
    data : array
        Input data.
    bins : array, optional
        Array of bins. If None, we use a default binning which corresponds to a full isotropic power spectrum.
        The default is None.
    sampling : float, optional
        Grid size. The default is 1.0.
    norm : TYPE, optional
        FFT normalization. Can be None or 'ortho'. The default is None.
    return_counts: bool, optional
        Return counts per bin. The default is None
    Raises
    ------
    Exception
        DESCRIPTION.
    Returns
    -------
    bins : TYPE
        DESCRIPTION.
    ps_mean : TYPE
        DESCRIPTION.
    ps_std : TYPE
        DESCRIPTION.
    counts : array, optional
        If return_counts=True, counts per bin.
    """
    # Check data shape
    for i in range(data.ndim):
        if data.shape[i] != data.shape[0]:
            raise Exception("Input data must be of shape (N, ..., N).")
    # Compute the full power spectrum of input data
    if data2 is None:
        data_ps = power_spectrum(data, norm=norm)
    else:
        data_ps = power_spectrum(data, data2=data2, norm=norm)
    return _spectral_iso(data_ps, bins=bins, sampling=sampling, return_counts=return_counts)

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
    bins = np.linspace(0, np.pi, 100)
    bins_centers = (bins[1:] + bins[:-1]) / 2
    fig, axs = plt.subplots(1, 3, figsize=figsize, sharex=True, sharey=True)
    for i in range(3):
        for j, image in enumerate(list_of_images):
            if isinstance(image, torch.Tensor):
                assert image.ndim == 3
                x = image[i].cpu().numpy()
            elif isinstance(image, np.ndarray):
                assert image.ndim == 3
                x = image[i]
            else:
                raise NotImplementedError
            _, ps, _ = power_spectrum_iso(x, bins=bins)
            axs[i].plot(bins_centers, ps[:-1], label=list_of_labels[j] if list_of_labels is not None else None)
        axs[i].set_yscale('log')
        axs[i].set_xscale('log')
        axs[i].set_title(f'Channel {i}')
    axs[0].legend()
    fig.show()

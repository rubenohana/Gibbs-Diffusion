import torch
import os
from torchvision.datasets import ImageFolder
from torchvision import transforms


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, name="cbsd68", transform=True):
        if transform:
            trans = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(256),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor()])
        else:
            trans = transforms.Compose([transforms.ToTensor()])
        
        if name in ["imagenet_train", "imagenet_val"]:
            path = os.path.join("/tmp/imagenet/", name.split("_")[1])
        elif name in ["CBSD68", "McMaster", "Kodak24"]:
            path = os.path.join("./data/", name)
        else:
            raise NotImplementedError
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} does not exist.")
        self.dataset = ImageFolder(path, transform=trans)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]

def get_colored_noise_2d(shape, phi=0, ret_psd=False, device=None):
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

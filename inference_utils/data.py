import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms

def ImageNet_train_dataset(path= "/tmp/imagenet/train/", transform = True):
    if transform:
        transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(256),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = ImageFolder(path, transform=transform)
    return train_dataset

def ImageNet_val_dataset(path= "/tmp/imagenet/val/", transform = True):
    if transform:
        transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(256),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.ToTensor()])
    val_dataset = ImageFolder(path, transform=transform)
    return val_dataset

def CBSD68_dataset(path = "./data/CBSD68/"):
    '''Returns images normalized in [0,1]'''
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(256),
                                    transforms.ToTensor()])
    dataset = ImageFolder(path, transform=transform)
    return dataset

def McMaster_dataset(path = "./data/McMaster/"):
    '''Returns images normalized in [0,1]'''
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(256),
                                    transforms.ToTensor()])
    dataset = ImageFolder(path, transform=transform)
    return dataset

def Kodak24_dataset(path = "./data/Kodak24/"):
    '''Returns images normalized in [0,1]'''
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(256),
                                    transforms.ToTensor()])
    dataset = ImageFolder(path, transform=transform)
    return dataset

class GDiff_dataset(torch.utils.data.Dataset):
    def __init__(self, dataset="cbsd68"):
        
        if dataset =="imagenet_train":
            self.dataset = ImageNet_train_dataset()
        elif dataset =="imagenet_val":
            self.dataset = ImageNet_val_dataset()
        elif dataset =="cbsd68":
            self.dataset = CBSD68_dataset()
        elif dataset =="mcmaster":
            self.dataset = McMaster_dataset()
        elif dataset =="kodak24":
            self.dataset = Kodak24_dataset()
        else:
            raise NotImplementedError
        self.dataset_len = len(self.dataset)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, item):
        return self.dataset[item]


def get_noise_level_estimate(y, sigma_min, sigma_max):
    """ Estimate the noise level of the image y. Image y is assumed to take values in [0, 1].
    Heuristic is calibrated for ImageNet and alpha in [-1, 1]. See calibration_notebooks/heuristic_sigma_estimation.ipynb. """
    assert y.ndim == 4 or y.ndim == 3 # (N, C, H, W) or (C, H, W)
    y_std = torch.std(y, dim=(-1, -2, -3))
    sigma_est = y_std*1.15 - 0.17 # Heuristic from heuristic_sigma_estimation.ipynb for Imagenet
    range_sigma = sigma_max - sigma_min
    sigma_est = torch.clamp(sigma_est, min=sigma_min + 0.05*range_sigma, max=sigma_max - 0.05*range_sigma)
    return sigma_est

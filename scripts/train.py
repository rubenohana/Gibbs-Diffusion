import torch
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import os, sys

sys.path.append('../')

from gdiff.model import GDiff
from gdiff.model import load_model as load_gdiff_model
from gdiff.data import ImageDataset


def main(hparams):
    # Training hyperparameters
    diffusion_steps = hparams.diffusion_steps
    dataset_choice = hparams.dataset_choice
    max_epoch = hparams.max_epoch
    batch_size = hparams.batch_size
    n_devices = hparams.n_devices
    optimizer = hparams.optimizer
    lr = hparams.learning_rate
    weight_decay = hparams.weight_decay
    accelerator = hparams.accelerator
    enable_ckpt = hparams.enable_ckpt

    #To be saved in wandb
    hyperparameters_config = dict(
        diffusion_steps = diffusion_steps,
        dataset_choice = dataset_choice,
        max_epoch = max_epoch,
        batch_size = batch_size,
        n_devices = n_devices,
        optimizer = optimizer,
        lr = lr,
        weight_decay = weight_decay
    )

    # Loading parameters
    load_model = hparams.load_model

    #directory where to store the checkpoints:
    default_root_dir=f"../model_checkpoints/{dataset_choice}_training/{diffusion_steps}steps_{optimizer}_{lr}lr_{batch_size}bs_{max_epoch}epochs/"

    #make directory if not existing:
    if not os.path.exists(default_root_dir):
        os.makedirs(default_root_dir, exist_ok=True)
    
    # Create datasets and data loaders
    if dataset_choice == 'ImageNet':

        train_dataset = ImageDataset("imagenet_train")
        val_dataset = ImageDataset("imagenet_val")

        n_channels, H, W = train_dataset[0][0].shape
        in_size_image = H * W
    else:
        #If you train on another dataset than imagenet, you will have to define yourself the train and validation sets in data.py
        train_dataset = ImageDataset(dataset_choice)
        val_dataset = ImageDataset(dataset_choice)

        n_channels, H, W = train_dataset[0][0].shape
        in_size_image = H * W

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=True)

    # Create model and trainer
    if load_model:
        model = load_gdiff_model(diffusion_steps=5000)
    else:
        model = GDiff(in_size=in_size_image, 
                      diffusion_steps=diffusion_steps, 
                      img_depth=n_channels, 
                      lr=lr, 
                      weight_decay=weight_decay)

    if hparams.wandb:
        wandb_logger = WandbLogger(project=hparams.wandb_project, 
                                   config=hyperparameters_config, 
                                   entity=hparams.wandb_entity, 
                                   group=hparams.wandb_group_name, 
                                   name=f"{dataset_choice}_steps{diffusion_steps}_lr{lr}_bs{batch_size}_opt{optimizer}_wd{weight_decay}")
    if enable_ckpt:
        checkpoint_callback = ModelCheckpoint(dirpath=default_root_dir, monitor=None)
    trainer = pl.Trainer(max_epochs=max_epoch, 
                        log_every_n_steps=10,
                        devices=n_devices, 
                        accelerator= accelerator, 
                        logger= wandb_logger if hparams.wandb else None,
                        callbacks=[checkpoint_callback] if enable_ckpt else None)

    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    parser = ArgumentParser()
    #Hardware
    parser.add_argument("--accelerator", default="gpu", type=str)
    parser.add_argument("--n_devices", default=1, type=int)
    #Hyperparameters
    parser.add_argument("--diffusion_steps", default=1000, type=int)
    parser.add_argument("--dataset_choice", default="ImageNet", type=str)
    parser.add_argument("--optimizer", default="AdamW", type=str)
    parser.add_argument("--learning_rate", default=1e-2, type=float)
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument("--max_epoch", default=10, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    #Logging Wandb
    parser.add_argument("--wandb", default=False, type=bool)
    parser.add_argument("--wandb_group_name", default="gdiff", type=str)
    parser.add_argument("--wandb_entity", default="####", type=str)
    parser.add_argument("--wandb_project", default="gdiff", type=str)
    #Model checkpointing
    parser.add_argument("--enable_ckpt", default=True, type=bool)
    parser.add_argument("--load_model", default=False, type=bool)

    args = parser.parse_args()
    main(args)

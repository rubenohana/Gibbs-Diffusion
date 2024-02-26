import torch
import lightning as pl
from gibbs_diffusion.model import GDiff, load_model
from gibbs_diffusion.model import load_model as load_gdiff_model
from torch.utils.data import DataLoader
from inference_utils.data import GDiff_dataset
import glob
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from argparse import ArgumentParser
import os

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
    alpha_ps = hparams.alpha_ps #if None, the model will be trained on many different colored noises
    wandb_group_name = hparams.wandb_group_name
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
        weight_decay = weight_decay,
        alpha_ps = alpha_ps
    )

    # Loading parameters
    load_model = hparams.load_model

    # Code for optionally loading model
    last_checkpoint = hparams.last_checkpoint

    #Use wandb or not
    wandb = hparams.wandb

    #directory where to store the checkpoints:
    default_root_dir=f"./model_checkpoints/{dataset_choice}_training/{diffusion_steps}steps_{optimizer}_{lr}lr_{batch_size}bs_{max_epoch}epochs/"

    #make directory if not existing:
    if not os.path.exists(default_root_dir):
        os.makedirs(default_root_dir, exist_ok=True)
    
    #default_root_dir=f"./ckpt_hyperparam_tuning/{dataset_choice}/{diffusion_steps}_steps_{optimizer}_{lr}_lr/lightning_logs/version_{pass_version}/checkpoints/"
        
    #Here default directory is ckpt_hyperparam_tuning
    if load_model:
        last_checkpoint = glob.glob(default_root_dir + "*.ckpt")[-1]
    
    # Create datasets and data loaders
    if dataset_choice == 'ImageNet':

        train_dataset = GDiff_dataset("imagenet_train")
        val_dataset = GDiff_dataset("imagenet_val")

        n_channels, H, W = train_dataset[0][0].shape
        in_size_image = H * W
    else:
        #Careful, if you train on another dataset than imagenet, you will have to define yourself the train and validation sets in data.py
        train_dataset = GDiff_dataset(dataset_choice)
        val_dataset = GDiff_dataset(dataset_choice)

        n_channels, H, W = train_dataset[0][0].shape
        in_size_image = H * W

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=True)

    # Create model and trainer
    if load_model:
        if dataset_choice == 'ImageNet':
            model = load_gdiff_model(diffusion_steps=10000,
                                in_size_image=256*256,
                                n_channels=3,
                                root_dir='model_checkpoints/',
                                device='gpu')
        else:
            print("model was only pretrained on Imagenet, no pretrained model to load on your dataset")
            raise NotImplementedError
    else:
        model = GDiff(in_size = in_size_image, 
                               diffusion_steps = diffusion_steps, 
                               img_depth = n_channels, 
                               lr = lr, 
                               weight_decay = weight_decay)

    if wandb:
        wandb_logger = WandbLogger(project="diffusion-models", 
                                   config = hyperparameters_config, 
                                   entity = 'rubenohana', 
                                   group= wandb_group_name, 
                                   name=f"{dataset_choice}_steps{diffusion_steps}_lr{lr}_bs{batch_size}_opt{optimizer}_wd{weight_decay}")
    if enable_ckpt:
        checkpoint_callback = ModelCheckpoint(dirpath=default_root_dir, 
                                          monitor=None)
    trainer = pl.Trainer(max_epochs=max_epoch, 
                        log_every_n_steps=10,
                        devices=n_devices, 
                        accelerator= accelerator, 
                        logger= wandb_logger if wandb else None,
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
    parser.add_argument("--alpha_ps", default=0, type=float or torch.Tensor)
    parser.add_argument("--max_epoch", default=10, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    #Logging Wandb
    parser.add_argument("--wandb", default=False, type = bool)
    parser.add_argument("--wandb_group_name", default="Hyperparameter_tuning", type=str)
    #Model checkpointing
    parser.add_argument("--enable_ckpt", default=True, type = bool)
    parser.add_argument("--load_model", default=False, type = bool)
    parser.add_argument("--last_checkpoint", default=None, type=str)

    args = parser.parse_args()
    main(args)
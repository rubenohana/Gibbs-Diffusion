# Listening to the Noise: Blind Denoising with Gibbs Diffusion (GDiff)

Code associated with the paper [Listening to the Noise: Blind Denoising with Gibbs Diffusion](https://arxiv.org/abs/2402.19455)

by [David Heurtel-Depeiges](https://david-heurtel-depeiges.github.io/), [Charles Margossian](https://charlesm93.github.io/), [Ruben Ohana](https://rubenohana.github.io/), [Bruno Régaldo-Saint Blancard](https://bregaldo.github.io/)

*Center for Computational Mathematics, Flatiron Institute, New York*

**Accepted to ICML 2024!**

This repository provides the code necessary to reproduce the application of GDiff on the blind denoising of natural images presented in our paper (Sect. 3.1.). We also release the corresponding trained model, as well as standard small-scale datasets for test purposes. Take a look at our [demo notebook](nbs/gdiff_demo.ipynb)!

The method demonstrated here is similar to the one used for the cosmology application in the paper (with some parts of the code being identical). Feel free to reach out to the authors for further information.

--------------------

GDiff is a Bayesian blind denoising method addressing posterior sampling of both signal and noise parameters. It relies on a Gibbs sampler that alternates sampling steps with a pretrained diffusion model (which defines the signal prior) and a Hamiltonian Monte Carlo sampler. Our paper presents applications in natural image denoising and cosmology (analysis of the cosmic microwave background).

Concretely, given a noisy observation $y = x + \varepsilon$, with $\varepsilon \sim \mathcal{N}(0, \Sigma_\phi)$, GDiff enables the sampling of the posterior distribution $p(x, \phi \mid y)$ for arbitrary diffusion-based signal prior $p(x)$.

In this repository, we showcase our method for the denoising of natural images in the case of colored noises with unknown amplitude $\sigma$ and spectral exponent $\varphi$.

  *Example of blind denoising on a noisy ImageNet sample (the signal prior is learned on ImageNet):*
<p align="center">
<img width="1000px"  src="figs/denoising_example.png">
</p>

*This prior remains relevant for the denoising of noisy BSD68 and Kodak24 images (*$\sigma = 0.2$ *and* $\varphi \in$ { $-0.4, 0, 0.4$ } *):*

<img src="figs/denoising_effect1.gif" width="270px"/> <img src="figs/denoising_effect3.gif" width="270px"/> <img src="figs/denoising_effect2.gif" width="270px"/>

## Installation

We recommend using an environment with Python >= 3.9 and PyTorch >= 2.0 (see [installation instructions](https://pytorch.org/)). GPU acceleration would require CUDA >= 11.6.

Dependencies can be installed by running in the relevant environment:
```
pip install -r requirements.txt
```

We release two diffusion models pretrained on ImageNet in a discrete time setting that can be downloaded by running:
```
gdown --folder --id 1E31OXJ9zZM3JzK9bsXsQFzFL16CPPCfN -O model_checkpoints
```
 The first model (used for the paper) was trained using 5,000 diffusion steps, while the second one was trained using 10,000 diffusion steps. Increasing the number of time steps led to refined results, at the expense of slower inferences.

You can alternatively download the models on this [Google Drive](https://drive.google.com/drive/folders/1E31OXJ9zZM3JzK9bsXsQFzFL16CPPCfN?usp=sharing). Make sure to put them in a folder called ```model_checkpoints```.

## Usage

### Demo Notebook

In the `nbs/` folder, we provide a [demo notebook](nbs/gdiff_demo.ipynb) showing how to use this code base to quickly perform blind denoising with GDiff.

Keep in mind that the computational cost of the inference highly depends on the chosen parameters and the model. Notably, it scales linearly with the number of Gibbs iterations and the total number of diffusion steps (intrinsic to the diffusion model here). It also depends on the initial noise level (the noisier the data, the longer the inference), the initialization strategy of the chains, and the dimensionality of the problem. The number of Gibbs iterations needed to converge to the target posterior distribution is typically determined empirically. In this example, this number was found to be quite low (i.e. about 50 iterations).

### Training Script

We train the diffusion model via the `scripts/train.py` script. For training on ImageNet, you would need to download ImageNet and update the `data_dir` paths. You can also train the model on your own dataset, which may require to adapt the ```gdiff/data.py``` script.

To train on a **single GPU**, you can simply run:
```python
python train.py
```

To train on **multiple GPUs** using DDP with Lightning, you can use (e.g., for a node with 8 GPUs):
```python
torchrun --standalone --nproc_per_node=8 train.py --n_devices 8
```

We highlight a few key arguments of the training script:
- `--diffusion_steps`: The number of diffusion steps of the diffusion model. The more the better, but also the longer the inference.
- `--dataset_choice`: The default is ImageNet, but you can also use `CBSD68`, `McMaster` or `Kodak24` (provided in this repository).
- `--wandb`: Set to True to use W&B.
- `--enable_ckpt`: Saves the model after `--max_epochs`.
- `--load_model`: Load a pre-trained model. Set to `True` for finetuning.

For referece, for the application of our paper, the training of a model on 100 epochs of ImageNet took about 40 hours on a node of 8 H100-80GB GPUs.

### Benchmarking Script

We also provide the script `scripts/denoise.py` allowing to benchmark GDiff, which can simply be run with:
```python
python denoise.py
```

Note that the comparison with DnCNN requires to clone the [KAIR](https://github.com/cszn/KAIR) repository and to download the corresponding model following [this page](https://github.com/cszn/KAIR/tree/master/model_zoo).

## Citation
```
@article{heurteldepeiges2024listening,
      title={Listening to the Noise: Blind Denoising with Gibbs Diffusion}, 
      author={David Heurtel-Depeiges and Charles C. Margossian and Ruben Ohana and Bruno {Régaldo-Saint Blancard}},
      year={2024},
      eprint={2402.19455},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```

"""
---
title: Utility functions for stable diffusion
summary: >
 Utility functions for stable diffusion
---

# Utility functions for [stable diffusion](index.html)
"""

import random
from pathlib import Path
from typing import Union, Mapping, Any

import PIL
import numpy as np
import torch
from PIL import Image

from labml import monit
from labml.logger import inspect
from stable_diffusion.latent_diffusion import LatentDiffusion
from stable_diffusion.model.autoencoder import Encoder, Decoder, Autoencoder
from stable_diffusion.model.clip_embedder import CLIPTextEmbedder
from stable_diffusion.model.unet import UNetModel

from stable_diffusion.utils.config import Config
from safetensors.torch import load_file


def set_seed(seed: int):
    """
    ### Set random seeds
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_encoder():
    encoder_config = Config.config['encoder']

    encoder = Encoder(
        **encoder_config
    )
    
    return encoder

def load_decoder():
    decoder_config = Config.config['decoder']

    decoder = Decoder(
        **decoder_config
    )
    
    return decoder

def load_autoencoder():

    with monit.section('Initialize autoencoder'):
        encoder = load_encoder()

        decoder = load_decoder()

        config = Config.config
        autoencoder_config = {
            **config['autoencoder'],
            'encoder': encoder,
            'decoder': decoder
        }

        autoencoder = Autoencoder(
            **autoencoder_config
        )
    
        return autoencoder

def load_unet_model():
    with monit.section('Initialize U-Net'):
        unet_config = Config.config['unet_model']

        unet_model = UNetModel(
            **unet_config
        )

    return unet_model

def load_clip_text_embedder(device = 'cuda:0'):
    with monit.section('Initialize CLIP Embedder'):
        clip_text_embedder = CLIPTextEmbedder(
            device=device,
        )

    return clip_text_embedder

def load_latent_diffusion_model(autoencoder, clip_text_embedder, unet_model):
    with monit.section('Initialize Latent Diffusion model'):
        latent_diffusion_config = {
            **Config.config['model'],
            'autoencoder': autoencoder,
            'clip_embedder': clip_text_embedder,
            'unet_model': unet_model
        }

        model = LatentDiffusion(
            **latent_diffusion_config
        )

        return model

def load_model(path: Union[str, Path] = '', device = 'cuda:0', config_path='') -> LatentDiffusion:
    """
    ### Load [`LatentDiffusion` model](latent_diffusion.html)
    """
    config = Config.load_config(config_path)
    Config.config = config

    autoencoder = load_autoencoder()
    clip_text_embedder = load_clip_text_embedder(device)
    unet_model = load_unet_model()

    # Initialize the Latent Diffusion model
    model = load_latent_diffusion_model(autoencoder, clip_text_embedder, unet_model)

    # Check if it is safetensor
    is_safetensor = str(path).endswith('.safetensors')

    # Load the checkpoint
    with monit.section(f"Loading model from {path}"):
        if is_safetensor:
            print("Loading safetensors")
            checkpoint: Mapping[str, Any] = load_file(str(path))
        else:
            checkpoint = torch.load(path, map_location="cpu")

    # Set model state
    with monit.section('Load state'):
        state_dict = checkpoint if is_safetensor else checkpoint["state_dict"]
        missing_keys, extra_keys = model.load_state_dict(checkpoint["state_dict"], strict=False)

    # Debugging output
    inspect(global_step=checkpoint.get('global_step', -1), missing_keys=missing_keys, extra_keys=extra_keys,
            _expand=True)

    #
    model.eval()
    return model


def load_img(path: str):
    """
    ### Load an image

    This loads an image from a file and returns a PyTorch tensor.

    :param path: is the path of the image
    """
    # Open Image
    image = Image.open(path).convert("RGB")
    # Get image size
    w, h = image.size
    # Resize to a multiple of 32
    w = w - w % 32
    h = h - h % 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    # Convert to numpy and map to `[-1, 1]` for `[0, 255]`
    image = np.array(image).astype(np.float32) * (2. / 255.0) - 1
    # Transpose to shape `[batch_size, channels, height, width]`
    image = image[None].transpose(0, 3, 1, 2)
    # Convert to torch
    return torch.from_numpy(image)


def save_images(images: torch.Tensor, dest_path: str, img_format: str = 'jpeg'):
    """
    ### Save a images

    :param images: is the tensor with images of shape `[batch_size, channels, height, width]`
    :param dest_path: is the folder to save images in
    :param img_format: is the image format
    """

    # Map images to `[0, 1]` space and clip
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
    # Transpose to `[batch_size, height, width, channels]` and convert to numpy
    images = images.cpu()
    images = images.permute(0, 2, 3, 1)
    images = images.float().numpy()

    # Save images
    for i, img in enumerate(images):
        img = Image.fromarray((255. * img).astype(np.uint8))
        img.save(dest_path, format=img_format)

def get_device(force_cpu: bool = False, cuda_fallback: str = 'cuda:0'):
    """
    ### Get device
    """
    if torch.cuda.is_available() and not force_cpu:
        device_name = torch.cuda.get_device_name(0)
        print("INFO: Using CUDA device: {}".format(device_name))
        return cuda_fallback

    print("WARNING: You are running this script without CUDA. Brace yourself for a slow ride.")
    return 'cpu'

def get_autocast(force_cpu: bool = False):
    """
    ### Get autocast
    """
    if torch.cuda.is_available() and not force_cpu:
        return torch.cuda.amp.autocast()

    return torch.cpu.amp.autocast()
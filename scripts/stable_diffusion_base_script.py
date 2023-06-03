import os, sys
sys.path.append(os.path.abspath(''))

import torch

from stable_diffusion.sampler.ddim import DDIMSampler
from stable_diffusion.sampler.ddpm import DDPMSampler
from stable_diffusion.utils.model import load_model, get_device
from stable_diffusion.latent_diffusion import LatentDiffusion
from stable_diffusion.sampler import DiffusionSampler

from stable_diffusion.utils.model import load_img

from typing import Union, Optional

from pathlib import Path

import torch
from typing import Tuple

def corrigir_entrada(imagem: torch.Tensor, mascara: torch.Tensor) -> torch.Tensor:
    num_canais_imagem = imagem.shape[0]
    num_canais_mascara = mascara.shape[0]

    if num_canais_imagem != 9 or num_canais_mascara != 1:
        nova_imagem = torch.cat([imagem] * (9 // num_canais_imagem), dim=0)
        nova_mascara = torch.cat([mascara] * (1 // num_canais_mascara), dim=0)

        if num_canais_imagem % 9 != 0:
            canais_restantes_imagem = num_canais_imagem % 9
            nova_imagem = torch.cat([nova_imagem, imagem[:canais_restantes_imagem]], dim=0)

        if num_canais_mascara != 1:
            canais_restantes_mascara = num_canais_mascara
            nova_mascara = torch.cat([nova_mascara, mascara[:canais_restantes_mascara]], dim=0)

        return torch.cat([nova_imagem, nova_mascara], dim=0)
    else:
        return torch.cat([imagem, mascara], dim=0)

class StableDiffusionBaseScript:
    model: LatentDiffusion
    sampler: DiffusionSampler

    def __init__(self, *, checkpoint_path: Union[str, Path],
                 ddim_steps: int = 50,
                 ddim_eta: float = 0.0,
                 force_cpu: bool = False,
                 sampler_name: str='ddim',
                 n_steps: int = 50,
                 cuda_device: str = 'cuda:0',
                 ):
        """
        :param checkpoint_path: is the path of the checkpoint
        :param sampler_name: is the name of the [sampler](../sampler/index.html)
        :param n_steps: is the number of sampling steps
        :param ddim_eta: is the [DDIM sampling](../sampler/ddim.html) $\eta$ constant
        """
        self.checkpoint_path = checkpoint_path
        self.ddim_steps = ddim_steps
        self.ddim_eta = ddim_eta
        self.force_cpu = force_cpu
        self.sampler_name = sampler_name
        self.n_steps = n_steps
        self.cuda_device = cuda_device
        self.device_id = get_device(force_cpu, cuda_device)
        self.device = torch.device(self.device_id)

        # Load [latent diffusion model](../latent_diffusion.html)
        # Get device or force CPU if requested

    def encode_image(self, orig_img: str, batch_size: int = 1, mask: Optional[torch.Tensor] = None):
        """
        Encode an image in the latent space
        """
        orig_image = load_img(orig_img).to(self.device)
        
        # Concatenate the mask tensor with the image tensor
        img_with_mask = orig_image
        if mask:
            img_with_mask = torch.cat([orig_image, mask], dim=1)

        # Encode the image in the latent space and make `batch_size` copies of it
        orig = self.model.autoencoder_encode(img_with_mask).repeat(batch_size, 1, 1, 1)

        return orig
    
    def prepare_mask(self, mask: Optional[torch.Tensor], orig: torch.Tensor):
        # If `mask` is not provided,
        # we set a sample mask to preserve the bottom half of the image
        if mask is None:
            mask = torch.zeros_like(orig, device=self.device)
            mask[:, :, mask.shape[2] // 2:, :] = 1.
        else:
            mask = mask.to(self.device)

        return mask
    
    def calc_strength_time_step(self, strength: float):
        # Get the number of steps to diffuse the original
        t_index = int(strength * self.ddim_steps)

        return t_index
    
    def get_text_conditioning(self, uncond_scale: float, prompts: list, batch_size: int = 1):
        # In unconditional scaling is not $1$ get the embeddings for empty prompts (no conditioning).
        if uncond_scale != 1.0:
            un_cond = self.model.get_text_conditioning(batch_size * [""])
        else:
            un_cond = None

        print(prompts)
        # Get the prompt embeddings
        cond = self.model.get_text_conditioning(prompts)

        return un_cond, cond
    
    def decode_image(self, x: torch.Tensor):
        return self.model.autoencoder_decode(x)

    def paint(self,
              orig: torch.Tensor,
              cond: torch.Tensor,
              t_index: int,
              uncond_scale: float = 1.0,
              un_cond: Optional[torch.Tensor] = None,
              mask: Optional[torch.Tensor] = None,
              orig_noise: Optional[torch.Tensor] = None):
        
        orig_2 = None
        # If we have a mask and noise, it's in-painting
        if mask is not None and orig_noise is not None:
            orig_2 = orig
        # Add noise to the original image
        x = self.sampler.q_sample(orig, t_index, noise=orig_noise)
        # Reconstruct from the noisy image
        x = self.sampler.paint(x, cond, t_index,
                                orig=orig_2,
                                mask=mask,
                                orig_noise=orig_noise,
                                uncond_scale=uncond_scale,
                                uncond_cond=un_cond)
        
        return x

    def initialize_script(self, config_path: str = ''):
        self.load_model(config_path)
        self.initialize_sampler()

    def load_model(self, config_path: str = ''):
        self.model = load_model(
            self.checkpoint_path,
            self.device_id,
            config_path
        )

        # Move the model to device
        self.model.to(self.device)

    def initialize_sampler(self):
        if self.sampler_name == 'ddim':
            self.sampler = DDIMSampler(self.model,
                                       n_steps=self.n_steps,
                                       ddim_eta=self.ddim_eta)
        elif self.sampler_name == 'ddpm':
            self.sampler = DDPMSampler(self.model)

    def unload_model(self):
        del self.model
        torch.cuda.empty_cache()


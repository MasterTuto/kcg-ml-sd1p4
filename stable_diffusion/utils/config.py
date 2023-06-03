from functools import lru_cache
from pathlib import Path
import yaml

class Config:
    default_config = {
        'encoder': {
            'z_channels': 4,
            'in_channels': 3,
            'channels': 128,
            'channel_multipliers': [1, 2, 4, 4],
            'n_resnet_blocks': 2
        },
        'decoder': {
            'out_channels':3,
            'z_channels':4,
            'channels':128,
            'channel_multipliers':[1, 2, 4, 4],
            'n_resnet_blocks':2
        },
        'autoencoder': {
            'emb_channels': 4,
            'z_channels': 4,
        },
        'unet_model': {
            'in_channels': 4,
            'out_channels': 4,
            'channels': 320,
            'attention_levels': [0, 1, 2],
            'n_res_blocks': 2,
            'channel_multipliers': [1, 2, 4, 4],
            'n_heads': 8,
            'tf_layers': 1,
            'd_cond': 768
        },
        'model': {
            'linear_start': 0.00085,
            'linear_end': 0.0120,
            'n_steps': 1000,
            'latent_scaling_factor': 0.18215
        }
    }

    @classmethod
    def load_config(cls, yaml_path: str):
        if not yaml_path:
            print("No config file provided, using default config")
            return cls.default_config
        
        if not Path(yaml_path).exists():
            print("Config file does not exist, exiting...")
            exit(1)

        with open(yaml_path) as f:
            cls.config = yaml.load(f, Loader=yaml.FullLoader)
        
        return cls.config
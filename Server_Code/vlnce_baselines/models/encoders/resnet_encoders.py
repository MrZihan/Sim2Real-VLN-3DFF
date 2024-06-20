import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import torchvision
#import clip
from .clip import CLIP
from PIL import Image
from torchvision import transforms




class CLIPEncoder(nn.Module):
    r"""
    Takes in observations and produces an embedding of the rgb component.

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
        device: torch.device
    """

    def __init__(
        self, device, patch_size=16
    ):
        super().__init__()
        #self.model, _ = clip.load("data/ViT-B-32.pt", device=device)
        self.model = CLIP(
            input_resolution=224, patch_size=patch_size, width=768, layers=12, heads=12
        )
        self.model.load_state_dict(torch.load('pretrained/ViT-B-'+str(patch_size)+'.pt', map_location = torch.device('cpu')),strict=False)

        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.eval()

        
        self.rgb_transform = torch.nn.Sequential(
            transforms.Resize((224,224), interpolation=Image.BICUBIC),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
            )


    def forward(self, observations, fine_grained_fts=False):
        r"""Sends RGB observation through the TorchVision ResNet50 pre-trained
        on ImageNet. Sends through fully connected layer, activates, and
        returns final embedding.
        """

        rgb_observations = observations["rgb"].permute(0, 3, 1, 2)
        rgb_observations = self.rgb_transform(rgb_observations)
        
        if fine_grained_fts:
            output = self.model(rgb_observations.contiguous())
        else:
            output = self.model.encode_image(rgb_observations.contiguous())
        return output.float() # to fp32
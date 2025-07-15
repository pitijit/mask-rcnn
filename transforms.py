from typing import Optional, Tuple, Union
import torch
from torch import nn
from torchvision.transforms import functional as F
import torchvision.transforms as T


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class PILToTensor(nn.Module):
    def forward(self, image, target=None):
        # Only convert if image is PIL Image, skip if tensor already
        if not torch.is_tensor(image):
            image = F.pil_to_tensor(image)
        return image, target


class ToDtype(nn.Module):
    def __init__(self, dtype: torch.dtype, scale: bool = False):
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def forward(self, image, target=None):
        if self.scale:
            image = F.convert_image_dtype(image, self.dtype)
        else:
            image = image.to(self.dtype)
        return image, target


class RandomHorizontalFlip(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, image, target=None):
        if torch.rand(1).item() < self.p:
            # Get original width before flip
            _, _, width = F.get_dimensions(image)

            image = F.hflip(image)

            if target is not None:
                if "boxes" in target:
                    boxes = target["boxes"]
                    # Flip boxes horizontally
                    boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                    target["boxes"] = boxes
                if "masks" in target:
                    target["masks"] = target["masks"].flip(-1)
        return image, target


def get_transform(train: bool):
    transforms = [
        PILToTensor(),
        ToDtype(torch.float, scale=True)
    ]
    if train:
        transforms.insert(1, RandomHorizontalFlip(p=0.5))
    return Compose(transforms)

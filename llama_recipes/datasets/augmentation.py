"""Image augmentation utilities for paired retinal images."""

import math
import random
from dataclasses import dataclass

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


@dataclass(frozen=True)
class PaperAugmentationConfig:
    rotation_degrees: tuple[float, float] = (-20.0, 20.0)
    max_shift_pixels: int = 10
    scale_range: tuple[float, float] = (0.8, 1.2)
    flip_probability: float = 0.5
    log_gamma_range: tuple[float, float] = (-0.3, 0.3)


def build_paper_augmentation():
    """Return the augmentation settings described for MoEA-Net training."""
    return PaperAugmentationConfig()


class PairedRetinalAugmenter:
    """Apply the same spatial transform to pre/post treatment images."""

    def __init__(self, probability=0.6, transform=None):
        if not 0 <= probability <= 1:
            raise ValueError("probability must be in [0, 1].")
        self.probability = probability
        self.transform = transform or build_paper_augmentation()

    def __call__(self, images):
        if images.ndim != 4 or images.shape[0] != 2:
            raise ValueError("Expected paired images with shape [2, C, H, W].")
        if random.random() >= self.probability:
            return images

        if callable(self.transform):
            return self.transform(images)

        angle = random.uniform(*self.transform.rotation_degrees)
        shift = (
            random.randint(-self.transform.max_shift_pixels, self.transform.max_shift_pixels),
            random.randint(-self.transform.max_shift_pixels, self.transform.max_shift_pixels),
        )
        scale = random.uniform(*self.transform.scale_range)
        gamma = math.exp(random.uniform(*self.transform.log_gamma_range))
        should_flip = random.random() < self.transform.flip_probability

        augmented = []
        for image in images:
            image = F.affine(
                image,
                angle=angle,
                translate=shift,
                scale=scale,
                shear=(0.0, 0.0),
                interpolation=InterpolationMode.BILINEAR,
            )
            if should_flip:
                image = F.hflip(image)
            image = F.adjust_gamma(image.clamp(0, 1), gamma=gamma)
            augmented.append(image)
        return torch.stack(augmented)

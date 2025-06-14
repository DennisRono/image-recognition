import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from typing import Dict, Any

class AdvancedAugmentations:
    def __init__(self, config):
        self.config = config
        
    def get_train_transforms(self, image_size: tuple) -> A.Compose:
        return A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.HorizontalFlip(p=self.config.horizontal_flip_prob),
            A.Rotate(limit=self.config.rotation_limit, p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=self.config.brightness_limit,
                contrast_limit=self.config.contrast_limit,
                p=0.4
            ),
            A.RandomSunFlare(flare_roi=(0,0,1,0.5), p=0.1),
            A.PixelDropout(dropout_prob=self.config.dropout_prob, p=0.1),
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=self.config.shadow_num_upper, p=0.1),
            A.CLAHE(clip_limit=3.0, tile_grid_size=(8,8), p=0.2),
            A.GaussianBlur(blur_limit=(3,7), p=0.1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def get_val_transforms(self, image_size: tuple) -> A.Compose:
        return A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

class FaceAugmentations:
    @staticmethod
    def get_face_transforms(image_size: tuple = (112, 112)) -> A.Compose:
        return A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomResizedCrop(image_size[0], scale=(0.9, 1.0), p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.3),
            A.GaussianBlur(blur_limit=(3,5), p=0.1),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2()
        ])

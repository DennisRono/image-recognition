from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class ModelConfig:
    model_name: str = "efficientnet_l3"
    num_classes: int = 1000
    image_size: Tuple[int, int] = (384, 384)
    batch_size: int = 32
    learning_rate: float = 3e-5
    weight_decay: float = 1e-4
    epochs: int = 150
    precision: str = "bf16-mixed"
    gradient_clip_val: float = 0.5
    
@dataclass
class DataConfig:
    train_path: str = "data/train"
    val_path: str = "data/val"
    test_path: str = "data/test"
    num_workers: int = 8
    pin_memory: bool = True
    
@dataclass
class AugmentationConfig:
    horizontal_flip_prob: float = 0.5
    rotation_limit: int = 15
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    dropout_prob: float = 0.01
    shadow_num_upper: int = 3

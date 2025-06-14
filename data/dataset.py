import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from typing import List, Tuple, Optional
import albumentations as A

class ImageDataset(Dataset):
    def __init__(self, data_path: str, transforms: Optional[A.Compose] = None, 
                 csv_file: Optional[str] = None):
        self.data_path = Path(data_path)
        self.transforms = transforms
        
        if csv_file:
            self.df = pd.read_csv(csv_file)
            self.image_paths = self.df['image_path'].tolist()
            self.labels = self.df['label'].tolist()
        else:
            self.image_paths = list(self.data_path.rglob("*.jpg")) + \
                              list(self.data_path.rglob("*.png"))
            self.labels = [self._get_label_from_path(p) for p in self.image_paths]
        
        self.label_to_idx = {label: idx for idx, label in enumerate(set(self.labels))}
        self.num_classes = len(self.label_to_idx)
    
    def _get_label_from_path(self, path: Path) -> str:
        return path.parent.name
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']
        
        label_idx = self.label_to_idx[label]
        return image, label_idx

class FaceDataset(Dataset):
    def __init__(self, data_path: str, transforms: Optional[A.Compose] = None):
        self.data_path = Path(data_path)
        self.transforms = transforms
        self.image_paths = []
        self.labels = []
        
        for person_dir in self.data_path.iterdir():
            if person_dir.is_dir():
                person_images = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))
                self.image_paths.extend(person_images)
                self.labels.extend([person_dir.name] * len(person_images))
        
        self.label_to_idx = {label: idx for idx, label in enumerate(set(self.labels))}
        self.num_classes = len(self.label_to_idx)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']
        
        label_idx = self.label_to_idx[label]
        return image, label_idx

import numpy as np
from pathlib import Path

from config import DataConfig

data_config = DataConfig()

# Download a CIFAR-10 Dataset
def setup_cifar10_dataset():
    import torchvision.datasets as datasets
    
    # Download CIFAR-10
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True)
    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True)
    
    # Convert to folder structure
    def save_cifar_as_folders(dataset, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
        for class_name in class_names:
            (path / class_name).mkdir(exist_ok=True)
        
        for idx, (img, label) in enumerate(dataset):
            class_name = class_names[label]
            img.save(path / class_name / f"{idx}.png")
    
    save_cifar_as_folders(train_dataset, 'data/train')
    save_cifar_as_folders(val_dataset, 'data/val')


# Add this function to main.py
def create_mock_dataset(data_path: str, num_classes: int = 10, samples_per_class: int = 100):
    from PIL import Image
    import os
    
    data_path = Path(data_path)
    data_path.mkdir(parents=True, exist_ok=True)
    
    for class_id in range(num_classes):
        class_dir = data_path / f"class_{class_id}"
        class_dir.mkdir(exist_ok=True)
        
        for sample_id in range(samples_per_class):
            # Create random image
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            img.save(class_dir / f"sample_{sample_id}.jpg")

# Add this before setup_data() in train_model function:
# if not Path(data_config.train_path).exists():
#     print("Creating mock dataset...")
#     create_mock_dataset(data_config.train_path)
#     create_mock_dataset(data_config.val_path, samples_per_class=20)


setup_cifar10_dataset()
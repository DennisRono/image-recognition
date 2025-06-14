import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from torch.utils.data import DataLoader
import argparse
from pathlib import Path

from config import ModelConfig, DataConfig, AugmentationConfig
from data.dataset import ImageDataset, FaceDataset
from data.transforms import AdvancedAugmentations, FaceAugmentations
from models.architectures import create_model
from training.trainer import ImageClassificationModule, FaceRecognitionModule
from training.callbacks import get_callbacks

def setup_data(config: DataConfig, aug_config: AugmentationConfig, task: str = 'classification'):
    if task == 'face_recognition':
        transforms_class = FaceAugmentations()
        dataset_class = FaceDataset
        train_transforms = transforms_class.get_face_transforms()
        val_transforms = transforms_class.get_face_transforms()
    else:
        transforms_class = AdvancedAugmentations(aug_config)
        dataset_class = ImageDataset
        train_transforms = transforms_class.get_train_transforms((384, 384))
        val_transforms = transforms_class.get_val_transforms((384, 384))
    
    train_dataset = dataset_class(config.train_path, transforms=train_transforms)
    val_dataset = dataset_class(config.val_path, transforms=val_transforms)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    return train_loader, val_loader, train_dataset.num_classes

def train_model(args):
    # Setup configs
    model_config = ModelConfig()
    data_config = DataConfig()
    aug_config = AugmentationConfig()
    
    # Setup data
    train_loader, val_loader, num_classes = setup_data(
        data_config, aug_config, args.task
    )
    
    # Create model
    model = create_model(
        args.model_name, 
        num_classes=num_classes,
        pretrained=True
    )
    
    # Setup training module
    config_dict = {
        'num_classes': num_classes,
        'learning_rate': model_config.learning_rate,
        'weight_decay': model_config.weight_decay,
        'epochs': model_config.epochs,
        'use_focal_loss': args.use_focal_loss
    }
    
    if args.task == 'face_recognition':
        pl_module = FaceRecognitionModule(model, config_dict)
    else:
        pl_module = ImageClassificationModule(model, config_dict)
    
    # Setup logger
    if args.use_wandb:
        logger = WandbLogger(project="sota-image-recognition", name=args.experiment_name)
    else:
        logger = TensorBoardLogger("logs", name=args.experiment_name)
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=model_config.epochs,
        precision=model_config.precision,
        gradient_clip_val=model_config.gradient_clip_val,
        callbacks=get_callbacks(model_config),
        logger=logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        deterministic=True
    )
    
    # Train
    trainer.fit(pl_module, train_loader, val_loader)
    
    # Save model
    output_dir = Path("models") / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as TorchScript
    model.eval()
    scripted_model = torch.jit.script(model)
    scripted_model.save(output_dir / "model.pt")
    
    print(f"Model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['classification', 'face_recognition'], default='classification')
    parser.add_argument('--model_name', default='efficientnet_b3')
    parser.add_argument('--experiment_name', default='sota_experiment')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--use_focal_loss', action='store_true')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    pl.seed_everything(42)
    
    train_model(args)

if __name__ == "__main__":
    main()

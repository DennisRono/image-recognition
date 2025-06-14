import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
import torchmetrics
from typing import Dict, Any, Optional
import wandb

class ImageClassificationModule(pl.LightningModule):
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        super().__init__()
        self.model = model
        self.config = config
        self.save_hyperparameters(ignore=['model'])
        
        # Metrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=config['num_classes'])
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=config['num_classes'])
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=config['num_classes'])
        
        # Loss function
        if config.get('use_focal_loss', False):
            from models.losses import FocalLoss
            self.criterion = FocalLoss(alpha=0.75, gamma=3.0)
        else:
            self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        self.train_acc(outputs, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        self.val_acc(outputs, labels)
        self.val_f1(outputs, labels)
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_epoch=True, prog_bar=True)
        self.log('val_f1', self.val_f1, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self) -> Dict[str, Any]:
        # Layer-specific learning rates
        backbone_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if 'classifier' in name or 'head' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
        
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': self.config['learning_rate']},
            {'params': head_params, 'lr': self.config['learning_rate'] * 3}
        ], weight_decay=self.config['weight_decay'])
        
        scheduler = OneCycleLR(
            optimizer,
            max_lr=[self.config['learning_rate'], self.config['learning_rate'] * 3],
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }

class FaceRecognitionModule(pl.LightningModule):
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        super().__init__()
        self.model = model
        self.config = config
        self.save_hyperparameters(ignore=['model'])
        
        from models.losses import ArcFaceLoss
        self.criterion = ArcFaceLoss(margin=0.5, scale=64.0)
        
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=config['num_classes'])
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=config['num_classes'])
    
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None):
        return self.model(x, labels)
    
    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        images, labels = batch
        outputs, embeddings = self(images, labels)
        loss = self.criterion(outputs, labels)
        
        self.train_acc(outputs, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        images, labels = batch
        outputs, embeddings = self(images, labels)
        loss = self.criterion(outputs, labels)
        
        self.val_acc(outputs, labels)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = optim.AdamW(self.parameters(), lr=self.config['learning_rate'], 
                              weight_decay=self.config['weight_decay'])
        
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config['epochs'])
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }

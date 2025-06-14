import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import torch
import numpy as np

class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, **kwargs):
        super().__init__(
            monitor='val_acc',
            mode='max',
            save_top_k=3,
            save_last=True,
            filename='{epoch}-{val_acc:.4f}',
            **kwargs
        )

class GradientClippingCallback(pl.Callback):
    def __init__(self, gradient_clip_val: float = 0.5):
        self.gradient_clip_val = gradient_clip_val
    
    def on_before_optimizer_step(self, trainer, pl_module, optimizer, optimizer_idx):
        torch.nn.utils.clip_grad_norm_(pl_module.parameters(), self.gradient_clip_val)

class WarmupCallback(pl.Callback):
    def __init__(self, warmup_epochs: int = 5):
        self.warmup_epochs = warmup_epochs
    
    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch < self.warmup_epochs:
            lr_scale = min(1.0, float(trainer.current_epoch + 1) / self.warmup_epochs)
            for pg in trainer.optimizers[0].param_groups:
                pg['lr'] = pg['initial_lr'] * lr_scale

def get_callbacks(config):
    callbacks = [
        CustomModelCheckpoint(),
        EarlyStopping(monitor='val_loss', patience=10, mode='min'),
        LearningRateMonitor(logging_interval='step'),
        GradientClippingCallback(config.gradient_clip_val),
        WarmupCallback(warmup_epochs=5)
    ]
    return callbacks

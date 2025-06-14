import torch
import torch.nn as nn
import timm
from transformers import AutoModel, AutoConfig
from typing import Optional, Dict, Any

class EfficientNetL3(nn.Module):
    def __init__(self, num_classes: int = 1000, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_l3', pretrained=pretrained, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.backbone.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)
    
class EfficientNetB3(nn.Module):
    def __init__(self, num_classes: int = 1000, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b3', pretrained=pretrained, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.backbone.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

class ViTHybrid384(nn.Module):
    def __init__(self, num_classes: int = 1000, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model('vit_large_patch16_384', pretrained=pretrained, num_classes=0)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.backbone.num_features),
            nn.Dropout(0.1),
            nn.Linear(self.backbone.num_features, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

class InternVL2(nn.Module):
    def __init__(self, num_classes: int = 1000, model_name: str = "OpenGVLab/InternVL2-2B"):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=x)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(pooled_output)

class ArcFaceModel(nn.Module):
    def __init__(self, backbone: str = "resnet50", num_classes: int = 1000, 
                 embedding_size: int = 512, margin: float = 0.5, scale: float = 64.0):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        self.embedding = nn.Linear(self.backbone.num_features, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        features = self.backbone(x)
        embeddings = self.bn(self.embedding(features))
        
        if labels is None:
            return embeddings
        
        # ArcFace loss computation
        cosine = torch.mm(torch.nn.functional.normalize(embeddings), 
                         torch.nn.functional.normalize(self.weight).t())
        
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * torch.cos(self.margin) - sine * torch.sin(self.margin)
        
        one_hot = torch.zeros(cosine.size()).to(x.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        
        return output, embeddings

def create_model(model_name: str, num_classes: int, **kwargs) -> nn.Module:
    models = {
        "efficientnet_l3": EfficientNetL3,
        "efficientnet_b3": EfficientNetB3,
        "vit_hybrid_384": ViTHybrid384,
        "internvl2": InternVL2,
        "arcface": ArcFaceModel
    }
    
    if model_name not in models:
        raise ValueError(f"Model {model_name} not supported")
    
    return models[model_name](num_classes=num_classes, **kwargs)

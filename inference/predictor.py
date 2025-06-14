import torch
import torch.nn as nn
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
import onnxruntime as ort
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ModelPredictor:
    def __init__(self, model_path: str, device: str = 'cuda', use_onnx: bool = False):
        self.device = device
        self.use_onnx = use_onnx
        
        if use_onnx:
            self.session = ort.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
        else:
            self.model = torch.jit.load(model_path, map_location=device)
            self.model.eval()
        
        self.transforms = A.Compose([
            A.Resize(384, 384),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        augmented = self.transforms(image=image)
        tensor = augmented['image'].unsqueeze(0)
        
        if not self.use_onnx:
            tensor = tensor.to(self.device)
        
        return tensor
    
    def predict(self, image: np.ndarray) -> Tuple[int, float]:
        tensor = self.preprocess(image)
        
        with torch.no_grad():
            if self.use_onnx:
                outputs = self.session.run(None, {self.input_name: tensor.numpy()})
                logits = torch.from_numpy(outputs[0])
            else:
                logits = self.model(tensor)
            
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            return predicted.item(), confidence.item()
    
    def predict_batch(self, images: List[np.ndarray]) -> List[Tuple[int, float]]:
        results = []
        for image in images:
            pred, conf = self.predict(image)
            results.append((pred, conf))
        return results

class FaceRecognitionPredictor:
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = device
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()
        
        self.transforms = A.Compose([
            A.Resize(112, 112),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2()
        ])
    
    def extract_embedding(self, face_image: np.ndarray) -> np.ndarray:
        tensor = self.preprocess(face_image)
        
        with torch.no_grad():
            embedding = self.model(tensor)
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            
        return embedding.cpu().numpy()
    
    def compare_faces(self, face1: np.ndarray, face2: np.ndarray, 
                     threshold: float = 0.6) -> Tuple[bool, float]:
        emb1 = self.extract_embedding(face1)
        emb2 = self.extract_embedding(face2)
        
        similarity = np.dot(emb1, emb2.T)[0][0]
        is_same = similarity > threshold
        
        return is_same, similarity
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        augmented = self.transforms(image=image)
        tensor = augmented['image'].unsqueeze(0).to(self.device)
        return tensor

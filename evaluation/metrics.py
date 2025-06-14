import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

class ModelEvaluator:
    def __init__(self):
        self.predictions = []
        self.targets = []
        self.probabilities = []
    
    def add_batch(self, preds: torch.Tensor, targets: torch.Tensor, probs: torch.Tensor):
        self.predictions.extend(preds.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        self.probabilities.extend(probs.cpu().numpy())
    
    def compute_metrics(self) -> Dict[str, float]:
        accuracy = accuracy_score(self.targets, self.predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.targets, self.predictions, average='weighted'
        )
        
        # For multiclass, compute AUC using one-vs-rest
        try:
            auc = roc_auc_score(self.targets, self.probabilities, multi_class='ovr')
        except:
            auc = 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    def plot_roc_curve(self, save_path: str = None):
        if len(np.unique(self.targets)) == 2:  # Binary classification
            fpr, tpr, _ = roc_curve(self.targets, self.probabilities[:, 1])
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(self.targets, self.probabilities[:, 1]):.3f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            
            if save_path:
                plt.savefig(save_path)
            plt.show()
    
    def reset(self):
        self.predictions = []
        self.targets = []
        self.probabilities = []

class FaceRecognitionEvaluator:
    def __init__(self):
        self.similarities = []
        self.labels = []  # 1 for same person, 0 for different
    
    def add_pair(self, similarity: float, is_same: bool):
        self.similarities.append(similarity)
        self.labels.append(int(is_same))
    
    def compute_metrics(self, threshold: float = 0.6) -> Dict[str, float]:
        predictions = [1 if sim > threshold else 0 for sim in self.similarities]
        
        accuracy = accuracy_score(self.labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.labels, predictions, average='binary'
        )
        
        # Compute EER (Equal Error Rate)
        fpr, tpr, thresholds = roc_curve(self.labels, self.similarities)
        fnr = 1 - tpr
        eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'eer': eer,
            'eer_threshold': eer_threshold,
            'auc': roc_auc_score(self.labels, self.similarities)
        }

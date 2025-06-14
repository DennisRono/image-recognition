import torch
import time
import numpy as np
from pathlib import Path
import sys
sys.path.append('..')

from inference.predictor import ModelPredictor
from evaluation.metrics import ModelEvaluator

def benchmark_model():
    # Load model
    predictor = ModelPredictor('models/sota_experiment/model.pt')
    
    # Generate random test data
    batch_sizes = [1, 8, 16, 32]
    image_size = (384, 384, 3)
    
    results = {}
    
    for batch_size in batch_sizes:
        images = [np.random.randint(0, 255, image_size, dtype=np.uint8) for _ in range(batch_size)]
        
        # Warmup
        for _ in range(10):
            predictor.predict_batch(images[:1])
        
        # Benchmark
        start_time = time.time()
        iterations = 100
        
        for _ in range(iterations):
            predictions = predictor.predict_batch(images)
        
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_batch = total_time / iterations
        fps = batch_size / avg_time_per_batch
        
        results[batch_size] = {
            'avg_time_per_batch': avg_time_per_batch,
            'fps': fps
        }
        
        print(f"Batch size {batch_size}: {avg_time_per_batch:.4f}s per batch, {fps:.2f} FPS")
    
    return results

if __name__ == "__main__":
    benchmark_model()

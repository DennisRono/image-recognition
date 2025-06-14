import torch
import sys
sys.path.append('..')

from models.architectures import create_model
from utils.optimization import ModelOptimizer, TensorRTOptimizer

def export_model():
    # Load trained model
    model = create_model('efficientnet_l3', num_classes=1000)
    checkpoint = torch.load('models/sota_experiment/model.pt')
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Export to ONNX
    optimizer = ModelOptimizer()
    optimizer.export_to_onnx(
        model, 
        input_shape=(3, 384, 384),
        output_path='models/model.onnx'
    )
    
    # Optimize ONNX
    optimizer.optimize_onnx('models/model.onnx', 'models/model_optimized.onnx')
    
    # Convert to TensorRT (if available)
    try:
        trt_optimizer = TensorRTOptimizer()
        trt_optimizer.convert_to_tensorrt(
            'models/model_optimized.onnx',
            'models/model.trt',
            max_batch_size=32,
            fp16=True
        )
        print("TensorRT engine created successfully")
    except ImportError:
        print("TensorRT not available, skipping conversion")
    
    print("Model export completed")

if __name__ == "__main__":
    export_model()

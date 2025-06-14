import torch
import torch.nn as nn
from torch.nn.utils import prune
import onnx
import onnxruntime as ort
from pathlib import Path

class ModelOptimizer:
    @staticmethod
    def quantize_model(model: nn.Module, calibration_loader) -> nn.Module:
        model.eval()
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)
        
        # Calibration
        with torch.no_grad():
            for batch in calibration_loader:
                images, _ = batch
                model(images)
        
        torch.quantization.convert(model, inplace=True)
        return model
    
    @staticmethod
    def prune_model(model: nn.Module, amount: float = 0.3) -> nn.Module:
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prune.l1_unstructured(module, name='weight', amount=amount)
                prune.remove(module, 'weight')
        return model
    
    @staticmethod
    def export_to_onnx(model: nn.Module, input_shape: tuple, output_path: str):
        dummy_input = torch.randn(1, *input_shape)
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
    
    @staticmethod
    def optimize_onnx(model_path: str, output_path: str):
        import onnxoptimizer
        model = onnx.load(model_path)
        optimized_model = onnxoptimizer.optimize(model)
        onnx.save(optimized_model, output_path)

class TensorRTOptimizer:
    @staticmethod
    def convert_to_tensorrt(onnx_path: str, engine_path: str, 
                          max_batch_size: int = 32, fp16: bool = True):
        import tensorrt as trt
        
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        
        with open(onnx_path, 'rb') as model:
            parser.parse(model.read())
        
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB
        
        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        
        profile = builder.create_optimization_profile()
        profile.set_shape('input', (1, 3, 384, 384), (max_batch_size//2, 3, 384, 384), (max_batch_size, 3, 384, 384))
        config.add_optimization_profile(profile)
        
        engine = builder.build_engine(network, config)
        
        with open(engine_path, 'wb') as f:
            f.write(engine.serialize())

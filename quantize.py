import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
model_fp32 = "/home/harrison/Desktop/Stuff/rwkv-onnx/RWKV_24_2048_32_15.onnx"
model_quant = "/home/harrison/Desktop/Stuff/rwkv-onnx/RWKV_24_2048_32_15_quant.onnx"
quantized_model = quantize_dynamic(model_fp32, model_quant, per_channel=True, reduce_range=True)

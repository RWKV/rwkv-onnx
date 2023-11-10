# for use if the converter crashes before quantization
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic("RWKV_24_2048_32_15.onnx","RWKV_24_2048_32_15_quant.onnx", per_channel=True, reduce_range=True, use_external_data_format=False, weight_type=QuantType.QUInt8)
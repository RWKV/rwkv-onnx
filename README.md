# rwkv-onnx
A converter and basic tester for rwkv V5 onnx

## supports
* fp16, fp32 
* onnx opsversions 15/17 
* file sizes > 2GB
* cuda, cpu, tensorRT, mps

## Quantization
* Quantize using 'python3 ./quant.py'

## Note: converter requires protobuf 3.20.2
Install with
`pip install protobuf==3.20.2`

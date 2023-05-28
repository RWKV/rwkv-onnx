# rwkv-onnx
A converter and basic tester for rwkv onnx

## supports
* fp16
* fp32
* onnx opsversions 15/17 
* file sizes > 2GB

## Note: fp16+ops15 may not work with cuda erp, but may work with tensorRT

## Note: converter requires protobuf 3.20.0
Install with
`pip install protobuf==3.20.0`

# for use if the converter crashes before quantization


import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import customoptools.helper as helper
import os
files = [f for f in os.listdir('.') if os.path.isfile(f)]
files = [f for f in files if f.endswith(".onnx") or f.endswith(".ort")]
import inquirer
model = inquirer.list_input("Select model", choices=files)

choices = inquirer.checkbox(
        "Select quantization options (use space bar to select checkboxes)", choices=["per_channel(slow, more acurate)", "reduce_range(sometimes more accurate)", "use_external_data_format(use if model larger than 2B)"], default=["use_external_data_format(use if model larger than 2B)"])

reduce_range = False
per_channel = False
use_external_data_format = False
if "per_channel(slow, more acurate)" in choices:
    per_channel = True
if "reduce_range(sometimes more accurate)" in choices:
    reduce_range = True
if "use_external_data_format(use if model larger than 2B)" in choices:
    use_external_data_format = True

# choose quantization type (default is QUInt8) // choose 1
qtype = inquirer.list_input("Select quantization type", choices=[e for e in QuantType], default=QuantType.QUInt8.value )
qtype = QuantType(qtype)
endoption = f'_{qtype}-{"pc" if per_channel else "npc"}-{"rr" if reduce_range else "norr"}-{"ext" if use_external_data_format else "noext"}.onnx'

# load custom op
helper.ScrubCustomModel(model)
quantize_dynamic(model,model.replace(".onnx",endoption), per_channel=per_channel, reduce_range=reduce_range, use_external_data_format=use_external_data_format, weight_type=qtype,
                 op_types_to_quantize=[
                     # matmul only
                        "MatMul",
                 ],
                 extra_options={
                     # gpu only
                     
                 }
                 )
helper.FixCustomModel(model.replace(".onnx",endoption))
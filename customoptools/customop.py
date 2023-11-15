from onnxruntime_extensions import (
            onnx_op, PyOp, make_onnx_model,
            get_library_path as _get_library_path)

# register custom op for domain recursal.rwkv

import wkv5 as customOP

@onnx_op(op_type="wkv5",
                 inputs=[
                     PyOp.dt_float,
                     PyOp.dt_float, 
                     PyOp.dt_float,
                     PyOp.dt_float,
                     PyOp.dt_float,
                     PyOp.dt_float],
                    outputs=[PyOp.dt_float, PyOp.dt_float]
                     )
def wkv5(k, v, r, td, tf, state):
    
    ro, newstate = customOP.wkv5(k, v, r, td, tf, state)
    return ro, newstate
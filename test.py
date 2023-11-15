import onnxruntime as rt
# register custom op for domain recursal.rwkv
from customoptools.customop import _get_library_path





def initONNXFile(path, STREAMS, useAllAvailableProviders=False):
    

    # session execution provider options
    sess_options = rt.SessionOptions()
    sess_options.register_custom_ops_library(_get_library_path())
    # sess_options.enable_profiling = True

    print(rt.get_available_providers())
    if(not useAllAvailableProviders):
        import inquirer
    providers = inquirer.checkbox(
        "Select execution providers(use space bar to select checkboxes)", choices=rt.get_available_providers()) if not useAllAvailableProviders else rt.get_available_providers()
    print(providers)
    # sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess_options.intra_op_num_threads = 6
    sess_options.inter_op_num_threads = 6
    sess_options.execution_mode = rt.ExecutionMode.ORT_PARALLEL
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess_options.add_session_config_entry("session.intra_op.allow_spinning", "1")

    sess = rt.InferenceSession(
        path, sess_options, providers=providers)

    ins = {

    }

    embed = sess.get_inputs()[1].shape[-1]
    layers = (sess.get_inputs().__len__()-1)//3
    typenum = sess.get_inputs()[1].type
    heads = sess.get_inputs()[layers*2+1].shape[1]
    print("HEADS: ", heads)
    print("LAYERS: ", layers)
    print("EMBED: ", embed)
    print("TYPE: ", typenum)
    
    import numpy as np

    if typenum == "tensor(float)":
        typenum = np.float32
    elif typenum == "tensor(float16)":
        typenum = np.float16

    class InterOp():

        RnnOnly = True

        def forward(selff, xi, statei, statei2):
            # print(statei[0][23])
            # create inputs
            inputs = ins
            # get input names
            input_names = sess.get_inputs()
            input_names = [x.name for x in input_names]
            # get output names
            output_names = sess.get_outputs()
            output_names = [x.name for x in output_names]
            # print(output_names)

            # create input dict
            inputs[input_names[0]] = np.array(xi, dtype=np.int32)
            for i in range(len(input_names)-1):
                # print(input_names[i+1])
                if "wkv" in input_names[i+1]:
                    inputs[input_names[i+1]] = statei2[i-statei.__len__()]
                else:
                    # print(i, statei.__len__())
                    inputs[input_names[i+1]] = statei[i]

            outputs = sess.run(output_names, inputs)
            # print(outputs[1][23])

            return outputs[0], outputs[1:statei.__len__()+1], outputs[statei.__len__()+1:]
        
    model = InterOp()

    # emptyState = []

    emptyState = np.zeros((layers*2,STREAMS,embed), dtype=typenum)
    emptyState2 = np.zeros((layers,STREAMS,heads,embed//heads,embed//heads), dtype=typenum)
    # emptyState = np.array(([[[0.01]*embed]*STREAMS, STREAMS*[[0.01]*embed]])*layers, typenum)
    # emptyState2 = np.array(([[[[[0.01]*64]*64]*32]*STREAMS])*layers, typenum)
    print (emptyState.shape)
    print (emptyState2.shape)

    return model, emptyState, emptyState2

def npsample(ozut, temp: float = 1.0, top_p_usual: float = 0.8) -> int:
    import numpy as np
    from scipy.special import softmax

    try:
        ozut = ozut.numpy()
    except:
        try:
            ozut = ozut.cpu().numpy()
        except:
            ozut = np.array(ozut)
    # out[self.UNKNOWN_CHAR] = -float('Inf')
    # out[self.UNKNOWN_CHAR] = -float('Inf')
    # turn to float if is half and cpu
    probs = softmax(ozut, axis=-1)

    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = float(sorted_probs[np.argmax(
        cumulative_probs > top_p_usual)])
    probs[probs < cutoff] = 0
    if temp != 1.0:
        probs = pow(probs, 1.0 / temp)
    probs = probs / np.sum(probs, axis=0)
    mout = np.random.choice(a=len(probs), p=probs)
    return mout
# Example usage:

import inquirer
# get all .onnx files in current directory
import os
files = [f for f in os.listdir('.') if os.path.isfile(f)]
files = [f for f in files if f.endswith(".onnx") or f.endswith(".ort")]

from tokenizer import world as tokenizer
STREAMS = 1
model, state, state2 = initONNXFile(inquirer.list_input("Select model", choices=files), STREAMS) 

prompt = STREAMS * [tokenizer.encode("### Instruction:\nPlease write a short story of a man defeating a two headed dragon\n### Result\n")]

print(prompt.__len__())
# 3b is 2.5 tokens pers econd with 32 streams = 64 + 32 = 96 tokens per second
import tqdm
for tokennum in tqdm.tqdm(range(prompt[0].__len__()-1)):
    logits, state, state2 = model.forward([prompt[i][tokennum] for i in range(STREAMS)],state, state2)

print("Loaded prompt.")
import numpy as np
for i in range(1000):
    logits, state, state2 = model.forward([prompt[i][-1] for i in range(STREAMS)],state, state2)
    prompt = [prompt[i]+[np.argmax(logits[i])] for i in range(STREAMS)]
  
    print(tokenizer.decode(prompt[0][-1:]), end="", flush=True)
print(tokenizer.decode(prompt))

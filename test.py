
def initONNXFile(path, useAllAvailableProviders=False):
    import onnxruntime as rt

    # session execution provider options
    sess_options = rt.SessionOptions()

    print(rt.get_available_providers())
    if(not useAllAvailableProviders):
        import inquirer
    providers = inquirer.checkbox(
        "Select execution providers(use space bar to select checkboxes)", choices=rt.get_available_providers()) if not useAllAvailableProviders else rt.get_available_providers()
    print(providers)

    sess = rt.InferenceSession(
        path, sess_options, providers=providers)

    ins = {

    }

    embed = int(path.split("_")[2].split(".")[0])
    layers = int(path.split("_")[1])
    typenum = sess.get_inputs()[1].type
    print(typenum)
    import numpy as np

    if typenum == "tensor(float)":
        typenum = np.float32
    # elif typenum == "tensor(float16)":
    #     typenum = np.float16

    class InterOp():

        RnnOnly = True

        def forward(selff, xi, statei):
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
            inputs[input_names[0]] = np.array([xi], dtype=np.int32)
            for i in range(len(input_names)-1):
                inputs[input_names[i+1]] = statei[i]

            outputs = sess.run(output_names, inputs)
            # print(outputs[1][23])

            return outputs[0], outputs[1:]
        
    model = InterOp()

    # emptyState = []
    emptyState = np.array((([[0.00]*embed, [0.00]*embed, [0.00]*embed, [
            0.00]*embed]+[[0]*embed]))*layers, typenum)

    return model, emptyState

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

import inquirer
# get all .onnx files in current directory
import os
files = [f for f in os.listdir('.') if os.path.isfile(f)]
files = [f for f in files if f.endswith(".onnx")]
model, state = initONNXFile(inquirer.list_input("Select model", choices=files)) 

from transformers import GPT2Tokenizer
tokenizer:GPT2Tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

prompt = tokenizer.encode("User: What is the purpose of a 3.5 inch finglslop with an attached turboencorboratator? Bot:")

for token in prompt[:-1]:
    logits, state = model.forward(token,state)

print("Loaded prompt.")

for i in range(100):
    logits, state = model.forward(prompt[-1],state)
    prompt = prompt+[npsample(logits)]
    print(tokenizer.decode(prompt[-1]),end="", flush=True)
print(tokenizer.decode(prompt))

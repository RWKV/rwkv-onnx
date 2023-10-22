
import opslist

def RnnRWKV(ops:opslist.RWKVOnnxOps, *args):
    class myRWKV(ops.module):

        @ ops.initfunc
        def __init__(self, w):
            super(myRWKV, self).__init__()
            print("Legacy RWKV")

            self.ops = ops
            self.postprocess0 = ops.initTensor((w["ln_out.weight"]))
            self.postprocess1 = ops.initTensor((w["ln_out.bias"]))
            self.postprocess2 = ops.initTensor((w["head.weight"]))
            self.emb = ops.initTensor(w["emb.weight"])
            self.emb1 = ops.initTensor(w["blocks.0.ln0.weight"])
            self.emb2 = ops.initTensor(w["blocks.0.ln0.bias"])
            self.ln1w = (ops.stack(
                [w[f"blocks.{x}.ln1.weight"] for x in range(ops.n_layers)]))
            self.ln1b = (ops.stack(
                [w[f"blocks.{x}.ln1.bias"] for x in range(ops.n_layers)]))
            self.ln2w = (ops.stack(
                [w[f"blocks.{x}.ln2.weight"] for x in range(ops.n_layers)]))
            self.ln2b = (ops.stack(
                [w[f"blocks.{x}.ln2.bias"] for x in range(ops.n_layers)]))
            self.lnxw = (ops.stack(
                [w[f"blocks.{x}.att.ln_x.weight"].reshape(32,-1) for x in range(ops.n_layers)]))
            self.lnxb = (ops.stack(
                [w[f"blocks.{x}.att.ln_x.bias"].reshape(32,-1)  for x in range(ops.n_layers)]))
            self.time_decay = (ops.stack([
                w[f"blocks.{x}.att.time_decay"].double().exp().neg().exp().reshape(32,64,1).repeat(1,1,64) for x in range(ops.n_layers)], True))
            self.time_first = (ops.stack([
                w[f"blocks.{x}.att.time_faaaa"].reshape(32,64,1).repeat(1,1,64)  for x in range(ops.n_layers)],True))
            self.kktk = (ops.stack(
                [w[f"blocks.{x}.att.time_mix_k"] for x in range(ops.n_layers)]))
            self.vvtv = (ops.stack(
                [w[f"blocks.{x}.att.time_mix_v"] for x in range(ops.n_layers)]))
            self.rrtr = (ops.stack(
                [w[f"blocks.{x}.att.time_mix_r"] for x in range(ops.n_layers)]))
            self.ggtg = (ops.stack(
                [w[f"blocks.{x}.att.time_mix_g"] for x in range(ops.n_layers)]))
            self.key = (ops.stack(
                [w[f"blocks.{x}.att.key.weight"] for x in range(ops.n_layers)], exname="_key"))
            self.value = (ops.stack(
                [w[f"blocks.{x}.att.value.weight"] for x in range(ops.n_layers)], exname="_value"))
            self.receptance = (ops.stack([
                w[f"blocks.{x}.att.receptance.weight"] for x in range(ops.n_layers)], exname="_receptance"))
            self.gate = (ops.stack([
                w[f"blocks.{x}.att.gate.weight"] for x in range(ops.n_layers)], exname="_gate"))
            self.outputvv = (ops.stack([
                w[f"blocks.{x}.att.output.weight"] for x in range(ops.n_layers)], exname="_outputvv"))
            self.time_mix_k_ffn = (ops.stack([
                w[f"blocks.{x}.ffn.time_mix_k"] for x in range(ops.n_layers)]))
            self.time_mix_r_ffn = (ops.stack([
                w[f"blocks.{x}.ffn.time_mix_r"] for x in range(ops.n_layers)]))
            self.key_ffn = (ops.stack(
                [w[f"blocks.{x}.ffn.key.weight"] for x in range(ops.n_layers)], exname="_key_ffn"))
            self.receptance_ffn = (ops.stack([
                w[f"blocks.{x}.ffn.receptance.weight"] for x in range(ops.n_layers)], exname="_receptance_ffn"))
            self.value_ffn = (ops.stack([
                w[f"blocks.{x}.ffn.value.weight"] for x in range(ops.n_layers)], exname="_value_ffn"))
            del w
        # def torchwise(self, B, T, C, H, s, r, k, v, w, u):
 
        # at = k@v
        # att = at*u

        # for t in range(T):
   
            
        #     premat = (att[:,t] + s)
        #     # print(premat.shape, rt.shape)        
        #     rt = r[:,:,t:t+1,:].float()
            
        #     out[:,t] = ((rt @ premat)).reshape(out[:,t].shape)
            
        #     s = at[:,t] + w * s

          

        # out = out.reshape(B, T, C)  
        # return out, ss

        def wkv5(self, k,v, r, xx, state):

            
            td = self.time_decay[xx]
            tf = self.time_first[xx]
            kreshaped = ops.reshape(k, self.ops.kshape)
            vreshaped = ops.reshape(v, self.ops.vshape)
            rreshaped = ops.reshape(r, self.ops.rshape)

            kv = ops.matvec(kreshaped, vreshaped)
            kkv = ops.multiply(kv, tf)
            premat = ops.add(kkv, state)
            wkv = ops.matvec(rreshaped, premat)
            state = ops.multiply(state, td)
            state = ops.add(state, kv)

            return wkv, state
        

        @ops.layerdef
        def doLayer(self, x, statea, stateb, statec, xx):

            xy = ops.layernorm(x, self.ln1w[xx], self.ln1b[xx])

            k = ops.matvec(
                self.key[xx], ops.lerp(statea, xy, self.kktk[xx]), True)

            v = ops.matvec(self.value[xx], ops.lerp(
                statea, xy, self.vvtv[xx]), True)
            rr = ops.matvec(
                self.receptance[xx], ops.lerp(statea, xy, self.rrtr[xx]), True)
            
            g = ops.matvec(
                self.gate[xx], ops.lerp(statea, xy, self.ggtg[xx]))
            
            gg = ops.silu(g)

            

            wkv, state = self.wkv5(k,v, rr, xx,statec)
            wkv = self.ops.convertToFloat16(wkv)
            wkv8 = ops.divide(wkv, ops.eight)
        #             x = self.ln_x(x / self.head_size_divisor).view(B, T, C)
        # x = self.output(x * g)
            lnx = ops.groupnorm(wkv8, self.lnxw[xx], self.lnxb[xx])

            lnxo = ops.reshape(lnx, self.ops.normshape)
            
            mvvo = ops.matvec(
                self.outputvv[xx], ops.multiply(gg, lnxo))
            
            mvv = ops.add(mvvo, x)

            ddd = ops.layernorm(mvv, self.ln2w[xx], self.ln2b[xx])

            km = ops.relu(ops.matvec(self.key_ffn[xx], ops.lerp(
                stateb, ddd, self.time_mix_k_ffn[xx])))

            rt = ops.logistical((ops.matvec(self.receptance_ffn[xx], ops.lerp(
                stateb, ddd, self.time_mix_r_ffn[xx]))))

            x = ops.add(mvv, ops.multiply(
                ops.matvec(self.value_ffn[xx], ops.multiply(km, km)), rt))

            return x, xy, ddd, state

        @ ops.mainfunc
        def forward(self, x, state = None, statec = None):

            if (state is None):
                state = ops.emptyState
                statec = ops.emptyWkvState

            x = ops.layernorm(
                ops.getIndex(self.emb, x),
                self.emb1, self.emb2)

            statea = state[0::2]
            stateb = state[1::2]
            statec = statec
            
            # statee = state[4::5] if ops.useSafeWKV else [None]*ops.n_layers

            ot = []
            ot2 = []


            for i in range(ops.n_layers):
                x, aaa, bbb, ccc = self.doLayer(
                    
                    x, ops.convertToFloat16(statea[i]), ops.convertToFloat16(stateb[i]),ops.convertToFloat32(statec[i]), i)
                ot = ot + ([ops.convertToFloat32(aaa),ops.convertToFloat32(bbb)])   
                ot2 = ot2 + [ops.convertToFloat32(ccc)]
            x = ops.matvec(self.postprocess2, ops.layernorm(x, self.postprocess0,
                                                            self.postprocess1))

            return ops.convertToFloat32(x), ot, ot2


    ops.postProcessModule(myRWKV(*args))
    


import torch

def convert_model(path, dtype):
    #delete all .onnx and .bin files
    import os
    for file in os.listdir("."):
        if file.endswith(".onnx") or file.endswith(".bin"):
            os.remove(file)
    w = torch.load(path, map_location="cpu")
    dims = len(w["blocks.0.att.key.weight"])
    layers = len(
        list(filter(lambda x: "blocks" in x and "ln1.bias" in x, w.keys())))


    ops = opslist.RWKVOnnxOps(layers,dims,dtype=dtype, opsVersion=version.get(), externalData=use_external_data.get(), splitExternalData=splitExternalData.get(), fp32inout=fp32inout.get())

    RnnRWKV(ops,w)


import tkinter as tk
from tkinter import filedialog


# Create the main window
root = tk.Tk()
root.title("File Converter")

# Define the functions
def choose_input_file():
    input_file = filedialog.askopenfilename()
    input_path.set(input_file)

import numpy as np
def convert():
    path = input_path.get()
    dtype = np.float16 if use_fp16.get() else np.float32
    convert_model(path, dtype)

# Define the variables
input_path = tk.StringVar()
use_fp16 = tk.BooleanVar(value=True)
use_external_data = tk.BooleanVar(value=True)
splitExternalData = tk.BooleanVar(value=False)
fp32inout = tk.BooleanVar(value=False)
# version, number either 15/17
version = tk.IntVar(value=15)

# Create the widgets
input_label = tk.Label(root, text="Input Path:")
input_entry = tk.Entry(root, textvariable=input_path)
input_button = tk.Button(root, text="Browse...", command=choose_input_file)



check_button = tk.Checkbutton(root, text="Use fp16", variable=use_fp16)
check_button3 = tk.Checkbutton(root, text="External Data", variable=use_external_data)
check_button4 = tk.Checkbutton(root, text="Split External Data", variable=splitExternalData)
check_button5 = tk.Checkbutton(root, text="Float32 inputs/outputs", variable=fp32inout)
input_select = tk.OptionMenu(root, version, 15, 17, 18)


convert_button = tk.Button(root, text="Convert", command=convert)

# Add the widgets to the window
input_label.grid(row=0, column=0)
input_entry.grid(row=0, column=1)
input_button.grid(row=0, column=2)

check_button.grid(row=2, column=0)
check_button3.grid(row=2, column=2)
check_button4.grid(row=2, column=3)
check_button5.grid(row=2, column=4)
input_select.grid(row=3, column=0)

convert_button.grid(row=3, column=1)


# Start the main event loop
root.mainloop()

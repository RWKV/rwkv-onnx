
def RnnRWKV(ops, *args):
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
            self.time_decay = (ops.stack([
                w[f"blocks.{x}.att.time_decay"].double().exp().neg() for x in range(ops.n_layers)]))
            self.time_first = (ops.stack([
                w[f"blocks.{x}.att.time_first"] for x in range(ops.n_layers)]))
            self.kktk = (ops.stack(
                [w[f"blocks.{x}.att.time_mix_k"] for x in range(ops.n_layers)]))
            self.vvtv = (ops.stack(
                [w[f"blocks.{x}.att.time_mix_v"] for x in range(ops.n_layers)]))
            self.rrtr = (ops.stack(
                [w[f"blocks.{x}.att.time_mix_r"] for x in range(ops.n_layers)]))
            self.key = (ops.stack(
                [w[f"blocks.{x}.att.key.weight"] for x in range(ops.n_layers)]))
            self.value = (ops.stack(
                [w[f"blocks.{x}.att.value.weight"] for x in range(ops.n_layers)]))
            self.receptance = (ops.stack([
                w[f"blocks.{x}.att.receptance.weight"] for x in range(ops.n_layers)]))
            self.outputvv = (ops.stack([
                w[f"blocks.{x}.att.output.weight"] for x in range(ops.n_layers)]))
            self.time_mix_k_ffn = (ops.stack([
                w[f"blocks.{x}.ffn.time_mix_k"] for x in range(ops.n_layers)]))
            self.time_mix_r_ffn = (ops.stack([
                w[f"blocks.{x}.ffn.time_mix_r"] for x in range(ops.n_layers)]))
            self.key_ffn = (ops.stack(
                [w[f"blocks.{x}.ffn.key.weight"] for x in range(ops.n_layers)]))
            self.receptance_ffn = (ops.stack([
                w[f"blocks.{x}.ffn.receptance.weight"] for x in range(ops.n_layers)]))
            self.value_ffn = (ops.stack([
                w[f"blocks.{x}.ffn.value.weight"] for x in range(ops.n_layers)]))

        @ops.layerdef
        def doLayer(self, x, statea, stateb, statec, stated, statee, xx):

            xy = ops.layernorm(x, self.ln1w[xx], self.ln1b[xx])

            k = ops.matvec(
                self.key[xx], ops.lerp(statea, xy, self.kktk[xx]))

            v = ops.matvec(self.value[xx], ops.lerp(
                statea, xy, self.vvtv[xx]))
            rr = ops.matvec(
                self.receptance[xx], ops.lerp(statea, xy, self.rrtr[xx]))
            r = ops.logistical((rr))

            ww = ops.add(k, self.time_first[xx])
            p = ops.maximum(statee, ww)

            e1 = ops.exp(ops.subtract(statee, p))
            e2 = ops.exp(ops.subtract(ww, p))
            a = ops.add(ops.multiply(e1, stateb), ops.multiply(e2, v))
            b = ops.add(ops.multiply(e1, statec), e2)
            ww = ops.add(statee, self.time_decay[xx])

            p = ops.maximum(ww, k)

            e1 = ops.exp(ops.subtract(ww, p))
            e2 = ops.exp(ops.subtract(k, p))
            outb = ops.add(ops.multiply(e1, stateb), ops.multiply(e2, v))
            outc = ops.add(ops.multiply(e1, statec), e2)
            eee = p
            wkv = ops.divide(a, b)

            mvv = ops.add(x, ops.matvec(
                self.outputvv[xx], ops.multiply(r, wkv)))

            ddd = ops.layernorm(mvv, self.ln2w[xx], self.ln2b[xx])

            km = ops.relu(ops.matvec(self.key_ffn[xx], ops.lerp(
                stated, ddd, self.time_mix_k_ffn[xx])))

            rt = ops.logistical((ops.matvec(self.receptance_ffn[xx], ops.lerp(
                stated, ddd, self.time_mix_r_ffn[xx]))))

            x = ops.add(mvv, ops.multiply(
                ops.matvec(self.value_ffn[xx], ops.multiply(km, km)), rt))

            return x, xy, outb, outc, ddd, eee

        @ ops.mainfunc
        def forward(self, x, state = None):

            if (state is None):
                state = ops.emptyState

            x = ops.layernorm(
                ops.getIndex(self.emb, x),
                self.emb1, self.emb2)

            statea = state[0::5]
            stateb = state[1::5]
            statec = state[2::5]
            stated = state[3::5]
            statee = state[4::5]

            ot = []

            for i in range(ops.n_layers):
                x, aaa, bbb, ccc, ddd, eee = self.doLayer(
                    x, statea[i], stateb[i], statec[i], stated[i], statee[i], i)
                ot = ot + [aaa, bbb, ccc, ddd, eee]

            x = ops.matvec(self.postprocess2, ops.layernorm(x, self.postprocess0,
                                                            self.postprocess1))

            return x, ot


    ops.postProcessModule(myRWKV(*args))
    

import opslist

import torch

def convert_model(path, dtype):
    w = torch.load(path, map_location="cpu")
    dims = len(w["blocks.0.att.key.weight"])
    layers = len(
        list(filter(lambda x: "blocks" in x and "ln1.bias" in x, w.keys())))


    ops = opslist.RWKVOnnxOps(layers,dims,dtype=dtype)

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

# Create the widgets
input_label = tk.Label(root, text="Input Path:")
input_entry = tk.Entry(root, textvariable=input_path)
input_button = tk.Button(root, text="Browse...", command=choose_input_file)


check_button = tk.Checkbutton(root, text="Use fp16", variable=use_fp16)

convert_button = tk.Button(root, text="Convert", command=convert)

# Add the widgets to the window
input_label.grid(row=0, column=0)
input_entry.grid(row=0, column=1)
input_button.grid(row=0, column=2)

check_button.grid(row=2, column=0)

convert_button.grid(row=3, column=1)

# Start the main event loop
root.mainloop()

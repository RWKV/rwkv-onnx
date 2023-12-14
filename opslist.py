import numpy as np


class RWKVOnnxOps():

    def __init__(self, layers, embed, opsVersion = 15, externalData = True, splitExternalData = True,fp32inout=True, quantized = False, *args, dtype=None, heads=32, useCustomOperator=True, **kwargs):
        import onnx
        self.n_layers = layers
        self.n_embed = embed

        print("embed ", embed)
        
        dtype = onnx.TensorProto.FLOAT if dtype == np.float32 else onnx.TensorProto.FLOAT16 if dtype == np.float16 else onnx.TensorProto.BFLOAT16 if dtype == np.bfloat16 else onnx.TensorProto.FLOAT
        nptype = np.float32 if dtype == onnx.TensorProto.FLOAT else np.float16 if dtype == onnx.TensorProto.FLOAT16 else np.float16 if dtype == onnx.TensorProto.BFLOAT16 else np.float32

        self.nm = 0
        exportname = f"RWKV_{layers}_{embed}_{'32' if dtype == onnx.TensorProto.FLOAT else '16'}_{opsVersion}{'_custom' if useCustomOperator else ''}.onnx"
        externalname = f"RWKV_{layers}_{embed}_{'32' if dtype == onnx.TensorProto.FLOAT else '16'}_{opsVersion}{'_custom' if useCustomOperator else ''}"

        # remove old files
        import os
        if os.path.exists(exportname):
            os.remove(exportname)
        if os.path.exists(externalname):
            os.remove(externalname)

        self.TensorList = []
        self.NodeList = []


        def initTensor(x, isfp32 = False, exname = ""):

            npdtype = np.float32 if (isfp32 and fp32inout) else nptype
            ddtype = onnx.TensorProto.FLOAT if (isfp32 and fp32inout) else dtype
            name = f"PreTrainedTensor_{self.nm}"
            self.nm += 1
            if isinstance(x, list):
                xx = np.array(x).astype(npdtype)
            else:
                xx = x.float().cpu().numpy()
                # convert to float32
                xx = xx.astype(npdtype)
            rrx = onnx.helper.make_tensor(
                name,
                ddtype,
                xx.shape,
                xx.tobytes(),
                raw=True

            )



            if externalData:
                if not splitExternalData:
                    exname = ""
                onnx.external_data_helper.set_external_data(
                    rrx,
                    location=externalname+exname+".bin",

                )

            self.TensorList.append(rrx)
            return name
        
        def initIntTensor(x, exname = ""):
            name = f"PreTrainedTensor_{self.nm}"
            self.nm += 1
            if isinstance(x, list):
                xx = np.array(x).astype(np.int64)
            else:
                xx = x.squeeze().int().cpu().numpy()
                # convert to float32
                xx = xx.astype(np.int64)
            rrx = onnx.helper.make_tensor(
                name,
                onnx.TensorProto.INT64,
                xx.shape,
                xx.tobytes(),
                raw=True

            )



            # if externalData:
            #     if not splitExternalData:
            #         exname = ""
            #     onnx.external_data_helper.set_external_data(
            #         rrx,
            #         location=externalname+exname+".bin",

            #     )

            self.TensorList.append(rrx)
            return name

        self.initTensor = initTensor
        self.initIntTensor = initIntTensor

        def sqrt(x):
            name = f"sqrt_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Sqrt',
                inputs=[x],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name
        
        def convertToFloat16(x):
            if x == None:
                return None
            name = f"convertToFloat16_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Cast',
                inputs=[x],
                outputs=[name],
                to=onnx.TensorProto.FLOAT16
            )
            self.NodeList.append(node)

            return name
        
        def convertToFloat32(x):
            if x == None :
                return None
            name = f"convertToFloat32_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Cast',
                inputs=[x],
                outputs=[name],
                to=onnx.TensorProto.FLOAT
            )
            self.NodeList.append(node)

            return name
        
        self.convertToFloat16 = convertToFloat16 if (dtype == onnx.TensorProto.FLOAT16 and fp32inout) else lambda x: x
        self.convertToFloat32 = convertToFloat32 if (dtype == onnx.TensorProto.FLOAT16 and fp32inout) else lambda x: x

        self.sqrt = sqrt

        def mean(x, dim=None):
            if dim == None:
                dim = self.zeroInt
            name = f"mean_{self.nm}_out"
            self.nm += 1
            if opsVersion >= 18:
                
                node = onnx.helper.make_node(
                    'ReduceMean',
                    inputs=[x,dim],
                    outputs=[name]
                )

            else:
                node = onnx.helper.make_node(
                    'ReduceMean',
                    inputs=[x],
                    outputs=[name],
                    axes=dim,
                    keepdims=1


                )
            self.NodeList.append(node)

            return name

        self.mean = mean

        def meanvarnorm(x, dim=None):
            if dim == None:
                dim = self.zeroInt

            if dtype == onnx.TensorProto.FLOAT16:
                x = convertToFloat32(x)
                
            name = f"meanvarnorm_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'MeanVarianceNormalization',
                inputs=[x],
                outputs=[name],
                axes=dim,
                
            )
            self.NodeList.append(node)
            if dtype == onnx.TensorProto.FLOAT16:
                name = convertToFloat16(name)
            return name
        
        self.meanvarnorm = meanvarnorm

        def relu(x):
            name = f"relu_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Relu',
                inputs=[x],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        self.relu = relu

        def exp(x):
            name = f"exp_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Exp',
                inputs=[x],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        self.exp = exp

        def stack(x, fp32 = False, exname = ""):
            return [initTensor(r, fp32, exname) for r in x]

        self.stack = stack

        def matvec(x, y, outputfp32 = False):
            name = f"matvec_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'MatMul',
                inputs=[x, y],
                outputs=[name]
            )
            self.NodeList.append(node)
            if outputfp32:
                return self.convertToFloat32(name)
            return name
        
        

        self.matvec = matvec

        def prod(x):
            name = f"prod_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'ReduceProd',
                inputs=[x],
                outputs=[name],
                axes=[1],
                keepdims=0


            )
            self.NodeList.append(node)

            return name

        self.prod = prod

        def mul(x, y, opname=""):
            name = f"mul_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Mul',
                name=opname,
                inputs=[x, y],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        self.multiply = mul



        def add(x, y, newName = None):

            name = f"add_{self.nm}_out" if newName == None else newName
            self.nm += 1
            node = onnx.helper.make_node(
                'Add',
                inputs=[x, y],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        self.add = add

        def sub(x, y):
            name = f"sub_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Sub',
                inputs=[x, y],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        self.subtract = sub

        self.one = initTensor([1.0]*embed)
        self.margins = initTensor([0.00001]*embed, True)
        self.margins16 = initTensor([0.00001]*embed)
        self.margins32 = initTensor([0.00001]*(embed//heads))
        self.margins3232 = initTensor([0.00001]*(embed//heads),True)

        def lerpx(x, y, z):
            return self.add(x, self.multiply(self.subtract(y, x), z))

        self.lerp = lerpx

        def minimum(x, y):
            name = f"minimum_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Min',
                inputs=[x, y],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name
        self.minimum = minimum
        # module def
        self.module = object

        def log(x):
            name = f"log_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Log',
                inputs=[x],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        self.log = log

        # pytorch function defs
        self.initfunc = lambda x: x
        self.layerdef = lambda x: x
        self.mainfunc = lambda x: x

        def divide(x, y):
            name = f"divide_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Div',
                inputs=[x, y],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        self.divide = divide

        def layernorm17(x, w, b, newName = None):
            name = f"layernorm_{self.nm}_out" if newName == None else newName
            self.nm += 1
            node = onnx.helper.make_node(
                'LayerNormalization',
                inputs=[x, w, b],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name 
        # ort 15 does not support layernorm

        def layernorm(x, w, b, newName = None):
            # xee2 = self.subtract(x,self.mean(x))
            # x2 = self.add(self.sqrt(self.add(self.mean(self.multiply(xee2,xee2)), self.margins16)), self.margins16)
            # xo = self.divide(xee2, x2)

            xo = self.meanvarnorm(x)
            
            return self.add(self.multiply(w, xo), b, newName)


        self.layernorm = layernorm if opsVersion < 17 else layernorm17

        def groupnorm(x, w, b):
            # xee2 = self.subtract(x,self.mean(x,self.oneInt))
            # x2 = self.add(self.sqrt(self.add(self.mean(self.multiply(xee2,xee2),self.oneInt), self.margins32)), self.margins32)
            # lnx = self.divide(xee2, x2)
            lnx = self.meanvarnorm(x, self.oneInt)
            xo = self.add(self.multiply(w, lnx), b)
            xo = self.reshape(xo, self.normalshape)
            return xo
        
        def groupnorm18(x, w, b):

            x = self.reshape(x, self.normshape)
            
            name = f"groupnorm_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'GroupNormalization',
                inputs=[x, w, b],
                outputs=[name],
                num_groups=heads
            )
            self.NodeList.append(node)
            name = self.reshape(name, self.normalshape)
            return name
            

        
        self.groupnorm = groupnorm if opsVersion != 18 else groupnorm18

        

        def getIndex(x, y):
            name = f"getIndex_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Gather',
                inputs=[x, y],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name
        
        def wkv5(k, v, r, td, tf, state):
            name = f"getIndex_{self.nm}_out"
            stateout = state+"out"
            self.nm += 1
            node = onnx.helper.make_node(
                'wkv5',
                inputs=[k, v, r, td, tf, state],
                outputs=[name, stateout],
                **dict(
                domain="ai.onnx.contrib"
                ) if useCustomOperator else dict(
                )
            )
            self.NodeList.append(node)

            return name, stateout
        
        def wkv5compat(k, v, r, td, tf, state):
            kreshaped = self.reshape(k, self.kshape)
            vreshaped = self.reshape(v, self.vshape)
            rreshaped = self.reshape(r, self.rshape)

            kv = self.matvec(kreshaped, vreshaped)
            kkv = self.multiply(kv, tf)
            premat = self.add(kkv, state)
            wkv = self.matvec(rreshaped, premat)

            state2 = self.multiply(state, td)
            state3 = self.add(state2, kv, state+"out")

            return wkv, state3
        
        self.wkv5op = wkv5 if useCustomOperator else wkv5compat

        self.stackEmbed = False

        def neg(x):
            name = f"neg_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Neg',
                inputs=[x],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        self.neg = neg

        def logistic(x):
            name = f"logistic_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Sigmoid',
                inputs=[x],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name
        self.logistical = logistic

        def silu(x):
            return self.multiply(x, logistic(x))
        
        self.silu = silu

        def reshape(x, y):
            name = f"reshape_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Reshape',
                inputs=[x, y],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name
        
        self.reshape = reshape

        self.kshape = initIntTensor([-1,heads, embed//heads, 1])
        self.vshape = initIntTensor([-1,heads, 1, embed//heads])
        self.rshape = initIntTensor([-1, heads, 1, embed//heads])
        self.postwkvop = initIntTensor([-1, heads, embed//heads, embed//heads])
        self.prematshape = initIntTensor([-1, embed//heads, embed//heads])
        self.normshape = initIntTensor([-1, heads, embed//heads])
        self.normalshape = initIntTensor([-1, embed])
        self.pregnshape = initIntTensor([-1, heads, embed//heads])
        self.zeroInt = initIntTensor([1]) if opsVersion == 18 else [1]
        self.oneInt = initIntTensor([3]) if opsVersion == 18 else [3]
        self.eight = initTensor([[8.0]])
        self.premshape = initIntTensor([-1, heads,  embed//heads])

        def maximum(x, y):
            name = f"maximum_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Max',
                inputs=[x, y],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        self.maximum = maximum

        self.getIndex = getIndex

        # convert to float32
        self.emptyState = np.array((([[[0.00]*embed]*1, [[0.00]*embed]*1]))*layers)
        self.emptyState = np.array(self.emptyState)

        # emptwkv state is n_layers,32,64,64
        hs = embed//heads
        self.emptyWkvState = np.array(([1*[[[[0.0]*hs]*hs]*heads]]*layers))

        if dtype == onnx.TensorProto.FLOAT16 and not fp32inout:
            self.emptyState = self.emptyState.astype(np.float16)
            self.emptyWkvState = self.emptyWkvState.astype(np.float16)

        # self.zero = initTensor([0.0]*embed)

        def ppm(x):
            import onnx
            inputtensor = onnx.helper.make_tensor_value_info("input0",
                                                             onnx.TensorProto.INT32,
                                                             [-1]), "input0"

            emptyState = list(map(lambda x: (onnx.helper.make_tensor_value_info("state"+str(x),
                                                                                onnx.TensorProto.FLOAT if fp32inout else dtype,
                                                                                [-1,embed]), "state"+str(x)), range((2)*layers)))
            emptystate2 = list(map(lambda x: (onnx.helper.make_tensor_value_info("statewkv"+str(x),
                                                                                    onnx.TensorProto.FLOAT if fp32inout else dtype,
                                                                                    [-1,heads, hs, hs]), "statewkv"+str(x)), range(layers)))
            outs = x.forward(
                inputtensor[1], list(map(lambda x: x[1], emptyState)), list(map(lambda x: x[1], emptystate2)))
            print(self.TensorList.__len__())
            print(self.NodeList.__len__())
            print(outs)
            logits = onnx.helper.make_tensor_value_info(outs[0],
                                                        onnx.TensorProto.FLOAT if fp32inout else dtype,
                                                        [-1,65536])
            state = list(map(lambda x: onnx.helper.make_tensor_value_info(x,
                                                                          onnx.TensorProto.FLOAT if fp32inout else dtype,
                                                                          [-1, embed]), outs[1]))
            state2 = list(map(lambda x: onnx.helper.make_tensor_value_info(x,
                                                                            onnx.TensorProto.FLOAT if fp32inout else dtype,
                                                                            [-1, heads, hs, hs]), outs[2]))

            # Create the graph (GraphProto)
            graph_def = onnx.helper.make_graph(
                nodes=self.NodeList,  # The list of nodes in the graph.
                name="RWKV",
                # Graph input

                inputs=[inputtensor[0], * \
                        list(map(lambda x:x[0], emptyState)), * \
                        list(map(lambda x:x[0], emptystate2))],

                outputs=[logits, *state, *state2],  # Graph output

                initializer=self.TensorList,  # initializer



                # did not work, needs to be external

            )

            modelDef = onnx.helper.make_model(
                graph_def, producer_name="rwkvstic",
                **dict(
                opset_imports=[onnx.helper.make_opsetid("ai.onnx.contrib", opsVersion)])
                if useCustomOperator else 
                dict(
                )
            )


            

            modelDef.opset_import[0].version = opsVersion

            print("Nearly save")

            onnx.save(modelDef, exportname)
            del modelDef
            
            if(not useCustomOperator):
                onnx.checker.check_model(exportname)
                onnx.shape_inference.infer_shapes_path(exportname, check_type=True, strict_mode=True, data_prop=True)

                
            else:
                print("skipping model checking")

            
            
            # run model
            print("Model saved to: ", exportname, " and is ready to be run")
            print("Data type: ", dtype)
            print("Embedding size: ", embed)
            print("Number of layers: ", layers)
            print("external data: ", externalname)
            exit()
        self.postProcessModule = ppm

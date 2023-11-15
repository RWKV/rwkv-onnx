
def FixCustomModel(filepath):
    import onnx
    # load model even with external data
    file = filepath
    model = onnx.load(file, load_external_data=True)
    # set imports to custom op "ai.onnx.contrib"
    nodes = model.graph.node

    change = False
    # get type of node
    for i in range(len(nodes)):
        if "wkv5" in nodes[i].op_type:
            if(model.opset_import[0].domain != "ai.onnx.contrib"):
                model.opset_import[0].domain = "ai.onnx.contrib"
            
            change = True
            # set import to custom op 
            model.graph.node[i].domain = "ai.onnx.contrib"

    if change:
        onnx.save_model(model, filepath, save_as_external_data=True, location=file.replace(".onnx", ".bin"))

def ScrubCustomModel(filepath):
    import onnx
    # load model even with external data
    file = filepath
    model = onnx.load(file, load_external_data=True)

    # set domain to default
    model.opset_import[0].domain = "ai.onnx"
    
    nodes = model.graph.node
    change = False
    # get type of node
    for i in range(len(nodes)):
        if "wkv5" in nodes[i].op_type:
            # set import to custom op 
            change = True
            model.graph.node[i].domain = ""

    if change:
        onnx.save_model(model, filepath, save_as_external_data=True, location=file.replace(".onnx", ".bin"))
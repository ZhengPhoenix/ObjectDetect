import tensorflow as tf
import os
import struct

def read_compiled_weights(mlmodelc_path):
    """Read a compiled model.espresso.weights file.
 
    Args:
        mlmodelc_path (str): location of mlmodelc folder.
 
    Returns: dict[int, list[float]] of section to list of weights.
    """
    layer_bytes = []
    layer_data = {}
    filename = os.path.join(mlmodelc_path, 'model.espresso.weights')
    with open(filename, 'rb') as f:
        # First byte of the file is an integer with how many
        # sections there are.  This lets us iterate through each section
        # and get the map for how to read the rest of the file.
        num_layers = struct.unpack('<i', f.read(4))[0]
        print('layers num:', num_layers)
        f.read(4)  # padding bytes
 
        # The next section defines the number of bytes each layer contains.
        # It has a format of
        # | Layer Number | <padding> | Bytes in layer | <padding> |
        while len(layer_bytes) < num_layers:
            layer_num, _, num_bytes, _ = struct.unpack('<iiii', f.read(16))
            layer_bytes.append((layer_num, num_bytes))
 
        # Read actual layer weights.  Weights are floats as far as I can tell.
        for layer_num, num_bytes in layer_bytes:
#             print(num_bytes, type(num_bytes))
            data = struct.unpack('f' * int((num_bytes / 4)), f.read(num_bytes))
            layer_data[layer_num] = data
 
        return layer_data



def read_pb_weigths(pb_path):
    with tf.gfile.FastGFile(pb_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        constant_values = {}

    constant_values = {}

    with tf.Session() as sess:
    constant_ops = [op for op in sess.graph.get_operations() if op.type == "Const"]
    #   constant_ops = [op for op in sess.graph.get_operations()]
    for constant_op in constant_ops:
        constant_values[constant_op.name] = sess.run(constant_op.outputs[0])

    return constant_values
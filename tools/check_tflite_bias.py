import tensorflow as tf
import numpy as np

interp = tf.lite.Interpreter('data/model_int8.tflite')
interp.allocate_tensors()

for t in interp.get_tensor_details():
    if 'pseudo_qconst4' in t['name']:  # Conv1 bias
        bias = interp.get_tensor(t['index'])
        print(f"TFLite Conv1 bias:")
        print(f"  Shape: {bias.shape}")
        print(f"  Values: {bias}")
        print(f"  Dtype: {bias.dtype}")
        
        # Get quant params
        qp = t.get('quantization_parameters', {})
        print(f"  Scale: {qp.get('scales', [])}")
        print(f"  Zero point: {qp.get('zero_points', [])}")
        break

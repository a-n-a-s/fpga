import numpy as np
import tensorflow as tf

itp = tf.lite.Interpreter(model_path='data/model_int8.tflite')
itp.allocate_tensors()
in_d = itp.get_input_details()[0]
in_scale, in_zero = in_d['quantization']
print(f'Input scale: {in_scale}, zero_point: {in_zero}')

x = np.load('data/X_test.npy', allow_pickle=False)
print(f'X_test already normalized: min={x.min()}, max={x.max()}')

# Use first window - already normalized
w = x[0].astype(np.float32)
print(f'Window 0 (normalized): {w}')

# Quantize: (x / scale + zero_point)
q = np.round(w / in_scale + in_zero).astype(np.int32)
q = np.clip(q, -128, 127).astype(np.int8)
print(f'Quantized: {q}')

# Get TFLite prediction
inp_det = itp.get_input_details()[0]
out_det = itp.get_output_details()[0]
itp.set_tensor(inp_det['index'], q.reshape(1, 12, 1).astype(np.int8))
itp.invoke()
out_q = itp.get_tensor(out_det['index']).astype(np.int16)
tflite_pred = int(np.argmax(out_q, axis=1)[0])
print(f'TFLite prediction: {tflite_pred}')
print(f'TFLite logits (int8): {itp.get_tensor(out_det["index"])}')

import numpy as np
import tensorflow as tf

itp = tf.lite.Interpreter(model_path='data/model_int8.tflite')
itp.allocate_tensors()
in_d = itp.get_input_details()[0]
in_scale, in_zero = in_d['quantization']
print(f'Input scale: {in_scale}, zero_point: {in_zero}')

x = np.load('data/X_test.npy', allow_pickle=False)
w_raw = x[0].astype(np.float32)
print(f'Raw window: {w_raw}')

x_norm = w_raw / 400.0
q = np.round(x_norm / in_scale + in_zero).astype(np.int32)
q = np.clip(q, -128, 127).astype(np.int8)
print(f'Quantized (decimal): {q}')
print(f'Quantized (hex): {[format(int(x) & 0xFF, "02X") for x in q]}')

with open('test_decimal.mem', 'w') as f:
    f.write('\n'.join(str(int(x)) for x in q) + '\n')

with open('test_hex.mem', 'w') as f:
    f.write('\n'.join(format(int(x) & 0xFF, '02X') for x in q) + '\n')

print('Written test_decimal.mem and test_hex.mem')

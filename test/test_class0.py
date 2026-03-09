import numpy as np
import tensorflow as tf

itp = tf.lite.Interpreter(model_path='data/model_int8.tflite')
itp.allocate_tensors()
in_d = itp.get_input_details()[0]
in_scale, in_zero = in_d['quantization']

x = np.load('data/X_test.npy', allow_pickle=False)
y = np.load('data/y_test.npy', allow_pickle=False)

# Find a Class 0 example
for i in range(100):
    w_raw = x[i].astype(np.float32)
    x_norm = w_raw / 400.0
    q = np.round(x_norm / in_scale + in_zero).astype(np.int32)
    q = np.clip(q, -128, 127).astype(np.int8)
    
    # Get TFLite prediction
    inp_det = itp.get_input_details()[0]
    out_det = itp.get_output_details()[0]
    itp.set_tensor(inp_det['index'], q.reshape(1, 12, 1).astype(np.int8))
    itp.invoke()
    out_q = itp.get_tensor(out_det['index']).astype(np.int16)
    tflite_pred = int(np.argmax(out_q, axis=1)[0])
    
    print(f'Window {i}: TFLite={tflite_pred}, Label={y[i]}, quant={q}')
    if tflite_pred == 0:
        with open('test_class0.mem', 'w') as f:
            f.write('\n'.join(format(int(x) & 0xFF, '02X') for x in q) + '\n')
        print(f'Wrote test_class0.mem for window {i}')
        break

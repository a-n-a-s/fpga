import tensorflow as tf
import numpy as np

interp = tf.lite.Interpreter('1_1data/model_int8.tflite')
interp.allocate_tensors()

X = np.load('1_1data/X_test.npy')[:200]
y = np.load('1_1data/y_test.npy')[:200]

input_scale, input_zero = interp.get_input_details()[0]['quantization']
output_scale, output_zero = interp.get_output_details()[0]['quantization']

ca = 0  # Argmax correct
ct = 0  # Threshold 0.7 correct

for i in range(200):
    x = X[i].reshape(1, 12, 1)
    x_int8 = (x / input_scale + input_zero).astype(np.int8)
    interp.set_tensor(interp.get_input_details()[0]['index'], x_int8)
    interp.invoke()
    out = interp.get_tensor(interp.get_output_details()[0]['index'])[0]
    out_f = (out.astype(np.float32) - output_zero) * output_scale
    probs = tf.nn.softmax(out_f).numpy()
    
    if np.argmax(out_f) == y[i]:
        ca += 1
    if (1 if probs[1] > 0.7 else 0) == y[i]:
        ct += 1

print(f'Argmax accuracy: {ca}/200 = {100*ca/200:.1f}%')
print(f'Threshold 0.7 accuracy: {ct}/200 = {100*ct/200:.1f}%')

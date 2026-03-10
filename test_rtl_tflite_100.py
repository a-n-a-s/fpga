import numpy as np
import tensorflow as tf
import subprocess

# Load data
X = np.load('1_1data/X_test.npy')[:100]
y = np.load('1_1data/y_test.npy')[:100]

# TFLite setup
interp = tf.lite.Interpreter('1_1data/model_int8.tflite')
interp.allocate_tensors()
input_scale, input_zero = interp.get_input_details()[0]['quantization']

tflite_correct = 0
rtl_correct = 0
agreement = 0

for i in range(100):
    # TFLite
    x = X[i].reshape(1, 12, 1)
    x_int8 = (x / input_scale + input_zero).astype(np.int8)
    interp.set_tensor(interp.get_input_details()[0]['index'], x_int8)
    interp.invoke()
    out = interp.get_tensor(interp.get_output_details()[0]['index'])[0]
    tflite_pred = np.argmax(out)
    if tflite_pred == y[i]:
        tflite_correct += 1
    
    # RTL
    input_hex = [(int(v) + 256) % 256 for v in x_int8.flatten()]
    with open('input_data.mem', 'w') as f:
        for v in input_hex:
            f.write(f'{v:02X}\n')
    result = subprocess.run(['vvp', 'scripts/simv'], capture_output=True, text=True)
    # Find the LAST DEBUG_ARGMAX line (final result)
    rtl_pred = None
    for line in result.stdout.split('\n'):
        if 'DEBUG_ARGMAX' in line and 'class=' in line:
            rtl_pred = int(line.split('class=')[1].strip())
    
    if rtl_pred is not None:
        if rtl_pred == y[i]:
            rtl_correct += 1
        if rtl_pred == tflite_pred:
            agreement += 1
    else:
        print(f"Warning: No DEBUG_ARGMAX found for sample {i}")

print(f'TFLite accuracy: {tflite_correct}/100 = {tflite_correct}%')
print(f'RTL accuracy:    {rtl_correct}/100 = {rtl_correct}%')
print(f'Agreement:       {agreement}/100 = {agreement}%')

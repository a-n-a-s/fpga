#!/usr/bin/env python3
"""Debug Class 1 sample."""

import numpy as np
import tensorflow as tf
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / '1_1data'

def main():
    print("="*60)
    print("Class 1 Sample Debug")
    print("="*60)
    
    X_test = np.load(DATA / 'X_test.npy')
    y_test = np.load(DATA / 'y_test.npy')
    
    # Find a Class 1 sample
    for i in range(len(y_test)):
        if y_test[i] == 1:
            print(f"\nFound Class 1 sample at index {i}")
            break
    
    input_data = X_test[i].astype(np.float32)
    true_label = int(y_test[i])
    
    print(f"Input: {input_data}")
    
    # TFLite
    model_path = DATA / 'model_int8.tflite'
    interpreter = tf.lite.Interpreter(str(model_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale, input_zero = input_details[0]['quantization']
    output_scale, output_zero = output_details[0]['quantization']
    
    x_int8 = (input_data / input_scale + input_zero).astype(np.int8)
    x_int8 = x_int8.reshape(1, 12, 1)
    interpreter.set_tensor(input_details[0]['index'], x_int8)
    interpreter.invoke()
    
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    output_float = (output.astype(np.float32) - output_zero) * output_scale
    
    tflite_class = int(np.argmax(output_float))
    print(f"\nTFLite: class={tflite_class}, logits={output_float}")
    
    # RTL
    input_hex = [(int(v) + 256) % 256 for v in x_int8.flatten()]
    
    with open(ROOT / 'input_data.mem', 'w') as f:
        for v in input_hex:
            f.write(f"{v:02X}\n")
    
    result = subprocess.run(
        ['vvp', str(ROOT / 'scripts' / 'simv')],
        capture_output=True,
        text=True,
        cwd=ROOT
    )
    
    rtl_class = None
    rtl_logits = None
    for line in result.stdout.split('\n'):
        if 'DEBUG_ARGMAX' in line:
            logit0 = int(line.split('logit0=')[1].split(',')[0])
            logit1 = int(line.split('logit1=')[1].split(',')[0])
            rtl_class = int(line.split('class=')[1].strip())
            rtl_logits = [logit0, logit1]
            print(f"RTL:    class={rtl_class}, logits={rtl_logits}")
    
    print(f"\nTrue label: {true_label}")
    print(f"Match: {'YES' if rtl_class == tflite_class else 'NO'}")

if __name__ == "__main__":
    main()

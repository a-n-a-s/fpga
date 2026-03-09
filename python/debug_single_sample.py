#!/usr/bin/env python3
"""
Simple debug: Run one sample through RTL and TFLite and compare.
"""

import numpy as np
import tensorflow as tf
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
LATEST_DATA = ROOT / 'latest_data'

def main():
    print("="*60)
    print("Single Sample RTL vs TFLite Debug")
    print("="*60)
    
    # Load test data
    X_test = np.load(LATEST_DATA / 'X_test.npy')
    y_test = np.load(LATEST_DATA / 'y_test.npy')
    
    # Test first sample
    i = 0
    input_data = X_test[i].astype(np.float32)
    true_label = int(y_test[i])
    
    print(f"\nSample {i}: True label = {true_label}")
    print(f"Input: {input_data}")
    
    # === TFLite ===
    model_path = LATEST_DATA / 'model_int8.tflite'
    interpreter = tf.lite.Interpreter(str(model_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale, input_zero = input_details[0]['quantization']
    output_scale, output_zero = output_details[0]['quantization']
    
    # Quantize input
    x_int8 = (input_data / input_scale + input_zero).astype(np.int8)
    print(f"\nTFLite:")
    print(f"  Input quantized: {x_int8}")
    print(f"  Input scale={input_scale}, zero={input_zero}")
    
    x_int8 = x_int8.reshape(1, 12, 1)
    interpreter.set_tensor(input_details[0]['index'], x_int8)
    interpreter.invoke()
    
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    output_float = (output.astype(np.float32) - output_zero) * output_scale
    
    tflite_class = int(np.argmax(output_float))
    print(f"  Output (quantized): {output}")
    print(f"  Output (dequantized): {output_float}")
    print(f"  Prediction: class={tflite_class}")
    
    # === RTL ===
    input_hex = [(int(v) + 256) % 256 for v in x_int8.flatten()]
    print(f"\nRTL:")
    print(f"  Input hex: {input_hex}")
    
    # Write input file
    with open(ROOT / 'input_data.mem', 'w') as f:
        for v in input_hex:
            f.write(f"{v:02X}\n")
    
    # Run simulation
    result = subprocess.run(
        ['vvp', str(ROOT / 'scripts' / 'simv')],
        capture_output=True,
        text=True,
        cwd=ROOT
    )
    
    # Parse output
    rtl_class = None
    rtl_logits = None
    for line in result.stdout.split('\n'):
        if 'DEBUG_ARGMAX' in line:
            logit0 = int(line.split('logit0=')[1].split(',')[0])
            logit1 = int(line.split('logit1=')[1].split(',')[0])
            rtl_class = int(line.split('class=')[1].strip())
            rtl_logits = [logit0, logit1]
            print(f"  {line.strip()}")
    
    print(f"\n=== COMPARISON ===")
    print(f"TFLite: class={tflite_class}, logits={output_float}")
    print(f"RTL:    class={rtl_class}, logits={rtl_logits}")
    print(f"Match: {'YES' if rtl_class == tflite_class else 'NO'}")

if __name__ == "__main__":
    main()

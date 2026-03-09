#!/usr/bin/env python3
"""
Debug: Compare RTL and TFLite intermediate outputs sample by sample.
"""

import numpy as np
import tensorflow as tf
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
LATEST_DATA = ROOT / 'latest_data'

def run_rtl_debug(input_hex):
    """Run RTL and get intermediate dump."""
    # Write input
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
            try:
                logit0 = int(line.split('logit0=')[1].split(',')[0])
                logit1 = int(line.split('logit1=')[1].split(',')[0])
                rtl_class = int(line.split('class=')[1].strip())
                rtl_logits = [logit0, logit1]
            except:
                pass
    
    # Read intermediates dump
    intermediates_path = ROOT / 'rtl_intermediates.dump'
    intermediates = {}
    if intermediates_path.exists():
        with open(intermediates_path, 'r') as f:
            for line in f:
                if '=' in line:
                    parts = line.strip().split('=')
                    if len(parts) == 2:
                        name = parts[0].strip()
                        vals = [int(v) for v in parts[1].strip().split()]
                        intermediates[name] = vals
    
    return rtl_class, rtl_logits, intermediates

def run_tflite_debug(input_data):
    """Run TFLite and get intermediate outputs."""
    model_path = LATEST_DATA / 'model_int8.tflite'
    interpreter = tf.lite.Interpreter(str(model_path))
    interpreter.allocate_tensors()
    
    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Quantize input
    input_scale, input_zero = input_details[0]['quantization']
    x_int8 = (input_data / input_scale + input_zero).astype(np.int8)
    x_int8 = x_int8.reshape(1, 12, 1)
    
    interpreter.set_tensor(input_details[0]['index'], x_int8)
    interpreter.invoke()
    
    # Get output
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    output_scale, output_zero = output_details[0]['quantization']
    output_float = (output.astype(np.float32) - output_zero) * output_scale
    
    # Get intermediate tensors
    tensor_details = interpreter.get_tensor_details()
    intermediates = {}
    
    for t in tensor_details:
        try:
            data = interpreter.get_tensor(t['index'])
            intermediates[t['name']] = data.flatten()
        except:
            pass
    
    tflite_class = int(np.argmax(output_float))
    return tflite_class, output_float.tolist(), intermediates

def main():
    print("="*60)
    print("RTL vs TFLite Layer-by-Layer Debug")
    print("="*60)
    
    # Load test data
    X_test = np.load(LATEST_DATA / 'X_test.npy')
    y_test = np.load(LATEST_DATA / 'y_test.npy')
    
    # Test first 5 samples
    for i in range(5):
        print(f"\n{'='*60}")
        print(f"SAMPLE {i}: True label = {y_test[i]}")
        print(f"{'='*60}")
        
        input_data = X_test[i]
        input_hex = [(int(v) + 256) % 256 for v in input_data.astype(np.int8)]
        
        # Run TFLite
        tflite_class, tflite_logits, tflite_int = run_tflite_debug(input_data)
        print(f"\nTFLite: class={tflite_class}, logits={tflite_logits}")
        
        # Run RTL
        rtl_class, rtl_logits, rtl_int = run_rtl_debug(input_hex)
        print(f"RTL:    class={rtl_class}, logits={rtl_logits}")
        
        # Compare intermediates
        print("\n--- Intermediate Comparison ---")
        
        # Conv1 output
        if 'sequential_1/re_lu_1/Relu;sequential_1/conv1d_1/BiasAdd;sequential_1/conv1d_1/convolution/Squeeze;' in tflite_int:
            tflite_conv1 = tflite_int['sequential_1/re_lu_1/Relu;sequential_1/conv1d_1/BiasAdd;sequential_1/conv1d_1/convolution/Squeeze;']
            if 'conv1_out' in rtl_int:
                rtl_conv1 = rtl_int['conv1_out']
                print(f"Conv1 output: TFLite={tflite_conv1[:8]}... RTL={rtl_conv1[:8]}...")
                if len(tflite_conv1) == len(rtl_conv1):
                    match = np.array_equal(tflite_conv1, rtl_conv1)
                    print(f"  Match: {match}")
                    if not match:
                        diff = np.abs(tflite_conv1.astype(np.int32) - rtl_conv1.astype(np.int32))
                        print(f"  Max diff: {diff.max()}, Mean diff: {diff.mean():.2f}")
        
        # Pool output
        if 'sequential_1/max_pooling1d_1/MaxPool1d/Squeeze' in tflite_int:
            tflite_pool = tflite_int['sequential_1/max_pooling1d_1/MaxPool1d/Squeeze']
            if 'pool_out' in rtl_int:
                rtl_pool = rtl_int['pool_out']
                print(f"Pool output: TFLite={tflite_pool[:8]}... RTL={rtl_pool[:8]}...")
        
        # Conv2 output
        if 'sequential_1/re_lu_1_2/Relu;sequential_1/conv1d_1_2/BiasAdd;sequential_1/conv1d_1_2/convolution/Squeeze;' in tflite_int:
            tflite_conv2 = tflite_int['sequential_1/re_lu_1_2/Relu;sequential_1/conv1d_1_2/BiasAdd;sequential_1/conv1d_1_2/convolution/Squeeze;']
            if 'conv2_out' in rtl_int:
                rtl_conv2 = rtl_int['conv2_out']
                print(f"Conv2 output: TFLite={tflite_conv2[:8]}... RTL={rtl_conv2[:8]}...")
        
        print(f"\nMatch: {'✓' if rtl_class == tflite_class else '✗'}")

if __name__ == "__main__":
    main()

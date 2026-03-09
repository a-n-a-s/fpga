#!/usr/bin/env python3
"""
Debug Conv1 output element-by-element to find exact divergence.
"""

import numpy as np
import tensorflow as tf
import json

# Load TFLite model
interp = tf.lite.Interpreter(
    'data/model_int8.tflite',
    experimental_preserve_all_tensors=True
)
interp.allocate_tensors()

# Load input
input_data = np.load('test_input.npy').reshape(1, 12, 1)
print(f"Input: {input_data.flatten()}")

# Run inference
interp.set_tensor(interp.get_input_details()[0]['index'], input_data)
interp.invoke()

# Get Conv1 output
for t in interp.get_tensor_details():
    if 're_lu_1/Relu' in t['name'] and '1_2' not in t['name']:
        conv1_out = interp.get_tensor(t['index'])
        print(f"\nTFLite Conv1 output shape: {conv1_out.shape}")
        print(f"TFLite Conv1 output (first filter, all positions):")
        print(conv1_out[0, :, 0])
        break

# Load RTL output
print("\n" + "="*60)
print("RTL Conv1 output (from rtl_intermediates.dump)")
print("="*60)

rtl_conv1 = []
with open('rtl_intermediates.dump') as f:
    in_conv1 = False
    for line in f:
        line = line.strip()
        if line == '=== conv1 ===':
            in_conv1 = True
            continue
        if in_conv1:
            if line.startswith('==='):
                break
            rtl_conv1.append(int(line))

rtl_conv1 = np.array(rtl_conv1, dtype=np.int32).reshape(12, 8)
print(f"RTL Conv1 output shape: {rtl_conv1.shape}")
print(f"RTL Conv1 output (first filter, all positions):")
print(rtl_conv1[:, 0])

# Compare
print("\n" + "="*60)
print("COMPARISON (filter 0)")
print("="*60)
tflite_f0 = conv1_out[0, :, 0]
rtl_f0 = rtl_conv1[:, 0]

for i in range(12):
    match = "✓" if tflite_f0[i] == rtl_f0[i] else "✗"
    print(f"  [{i:2d}] TFLite={tflite_f0[i]:4d}  RTL={rtl_f0[i]:4d}  diff={tflite_f0[i]-rtl_f0[i]:5d}  {match}")

# Check if it's all -128 (zero-point floor)
print(f"\nRTL min: {rtl_conv1.min()}, max: {rtl_conv1.max()}")
print(f"RTL values: {np.unique(rtl_conv1)}")

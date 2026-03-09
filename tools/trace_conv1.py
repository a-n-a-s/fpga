#!/usr/bin/env python3
"""
Trace exact Conv1 computation to find divergence point.
"""

import numpy as np
import json

# Load input
input_data = np.load('test_input.npy')
print("Input:", input_data)

# De-zero-point (ACT_ZP = -128)
ACT_ZP = -128
input_dezp = input_data - ACT_ZP
print("Input de-zero-pointed:", input_dezp)

# Load conv1 weights
conv1_w = []
with open('data/conv1_weights_hex.mem') as f:
    for line in f:
        line = line.strip()
        if line:
            val = int(line, 16)
            if val > 127:
                val -= 256
            conv1_w.append(val)
conv1_w = np.array(conv1_w, dtype=np.int8).reshape(8, 3)
print("\nConv1 weights (8 filters x 3 kernel):")
print(conv1_w)

# Load conv1 bias
conv1_b = []
with open('data/conv1_bias_hex.mem') as f:
    for line in f:
        line = line.strip()
        if line:
            val = int(line, 16)
            if val > 2147483647:  # Check if > INT32 max
                val -= 4294967296
            conv1_b.append(val)
conv1_b = np.array(conv1_b, dtype=np.int32)
print("\nConv1 bias (8 filters):")
print(conv1_b)

# Manual convolution for filter 0, position 0
# With padding='same', position 0 uses kernel centered at input position 0
# This means: kernel[0] * input[-1] + kernel[1] * input[0] + kernel[2] * input[1]
# But input[-1] is out of bounds, so it's 0 (zero padding)

print("\n=== Manual Conv1 computation for filter 0 ===")

# For position 0 with padding='same':
# in_idx = pos + kernel_idx - 1
# kernel_idx=0: in_idx = 0 + 0 - 1 = -1 (out of bounds, skip)
# kernel_idx=1: in_idx = 0 + 1 - 1 = 0
# kernel_idx=2: in_idx = 0 + 2 - 1 = 1

acc = 0
for k in range(3):
    in_idx = 0 + k - 1  # position + kernel_idx - 1
    if in_idx >= 0 and in_idx < 12:
        contrib = input_dezp[in_idx] * conv1_w[0, k]
        acc += contrib
        print(f"  k={k}: in_idx={in_idx}, input_dezp={input_dezp[in_idx]}, weight={conv1_w[0,k]}, contrib={contrib}")
    else:
        print(f"  k={k}: in_idx={in_idx} (out of bounds, skip)")

print(f"\nAccumulator before bias: {acc}")
print(f"Bias: {conv1_b[0]}")
print(f"Accumulator after bias: {acc + conv1_b[0]}")

# Now apply requantization
# Load multiplier
with open('data/exact_multipliers.json' if False else 'data/quant_params.json') as f:
    try:
        mult_data = json.load(f)
        conv1_mult = mult_data.get('conv1_multipliers', [3307])[0]
    except:
        conv1_mult = 3307  # Default from cnn_top.v

print(f"\nRequantization:")
print(f"  Multiplier: {conv1_mult}")
print(f"  REQUANT_SHIFT: 20")

# Requant
REQUANT_SHIFT = 20
prod = acc * conv1_mult
print(f"  acc * mult = {acc} * {conv1_mult} = {prod}")

# Round and shift
if prod >= 0:
    scaled = (prod + (1 << (REQUANT_SHIFT - 1))) >> REQUANT_SHIFT
else:
    scaled = (prod - (1 << (REQUANT_SHIFT - 1))) >> REQUANT_SHIFT
print(f"  After shift: {scaled}")

# Add zero point
ACT_ZP_OUT = -128
scaled = scaled + ACT_ZP_OUT
print(f"  After zero-point: {scaled}")

# Saturate
if scaled > 127:
    result = 127
elif scaled < -128:
    result = -128
else:
    result = scaled
print(f"  After saturate: {result}")

# Compare with TFLite
print(f"\n=== TFLite output for filter 0, position 0 ===")
print("Expected: -128 (from earlier debug)")
print(f"Computed: {result}")
print(f"Match: {result == -128}")

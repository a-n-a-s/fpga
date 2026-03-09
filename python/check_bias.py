import json
import numpy as np

with open('data/quant_params.json') as f:
    qp = json.load(f)

# Check GAP computation - TFLite uses MEAN, RTL uses division
print('=== GAP Computation ===')
print(f'GAP divides by: {6} (CONV2_OUT_LEN)')
print(f'GAP output scale: {qp["activations"]["global_average_pooling1d_1/Mean"]["scale"][0]}')
print(f'GAP zero point: {qp["activations"]["global_average_pooling1d_1/Mean"]["zero_point"][0]}')

# Check what the correct GAP requant should be
gap_scale = qp['activations']['global_average_pooling1d_1/Mean']['scale'][0]
conv2_scale = qp['activations']['conv1d_1_2/BiasAdd']['scale'][0]

# GAP: mean = sum / 6, then requantize
# In RTL: acc accumulates (input - zp), then divides by 6
# But the scale change is: gap_scale / conv2_scale
print()
print(f'conv2 output scale: {conv2_scale}')
print(f'GAP output scale: {gap_scale}')
print(f'Ratio (should be ~1.0 if same scale): {gap_scale / conv2_scale:.4f}')

# Check bias values
print()
print('=== CONV1 Biases (raw int32) ===')
with open('data/conv1_bias_hex.mem') as f:
    lines = f.read().strip().split('\n')
    for i, line in enumerate(lines[:8]):
        val = int(line, 16)
        if val >= 2**31:
            val -= 2**32
        print(f'Bias {i}: {val}')

print()
print('=== CONV2 Biases (raw int32) ===')
with open('data/conv2_bias_hex.mem') as f:
    lines = f.read().strip().split('\n')
    for i, line in enumerate(lines[:16]):
        val = int(line, 16)
        if val >= 2**31:
            val -= 2**32
        print(f'Bias {i}: {val}')

print()
print('=== Dense Biases (raw int32) ===')
with open('data/dense_bias_hex.mem') as f:
    lines = f.read().strip().split('\n')
    for i, line in enumerate(lines[:2]):
        val = int(line, 16)
        if val >= 2**31:
            val -= 2**32
        print(f'Bias {i}: {val}')

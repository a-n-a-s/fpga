#!/usr/bin/env python3
"""Extract exact activation scales from TFLite model."""

import tensorflow as tf
import json

interp = tf.lite.Interpreter('data/model_int8.tflite')
interp.allocate_tensors()
details = interp.get_tensor_details()

print("All tensor quantization parameters:")
print("="*80)

activations = {}

for t in details:
    qp = t.get('quantization_parameters', {})
    scales = qp.get('scales', [])
    zps = qp.get('zero_points', [])
    
    if len(scales) > 0:
        name = t['name'][:60]
        print(f"{name:60s} scale={scales[0]:.10f} zp={zps[0] if zps else 'N/A'}")
        
        # Store key activations
        if 're_lu_1/Relu' in name and '1_2' not in name:
            activations['conv1_out'] = {'scale': scales[0], 'zp': zps[0] if zps else 0}
        elif 'max_pooling1d_1' in name and 'Squeeze' in name:
            activations['pool_out'] = {'scale': scales[0], 'zp': zps[0] if zps else 0}
        elif 're_lu_1_2/Relu' in name:
            activations['conv2_out'] = {'scale': scales[0], 'zp': zps[0] if zps else 0}
        elif 'global_average_pooling1d_1' in name:
            activations['gap_out'] = {'scale': scales[0], 'zp': zps[0] if zps else 0}
        elif 'StatefulPartitionedCall' in name:
            activations['output'] = {'scale': scales[0], 'zp': zps[0] if zps else 0}

print("\n" + "="*80)
print("KEY ACTIVATION SCALES (for multiplier calculation)")
print("="*80)

for name, params in activations.items():
    print(f"{name:15s} scale={params['scale']:.10f} zp={params['zp']}")

# Save to file
with open('data/activation_scales.json', 'w') as f:
    json.dump(activations, f, indent=2)

print(f"\nSaved to data/activation_scales.json")

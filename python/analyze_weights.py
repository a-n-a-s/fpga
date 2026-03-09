#!/usr/bin/env python3
"""
Analyze weight export from Colab and compare with RTL expectations.
"""

import numpy as np
import tensorflow as tf
from pathlib import Path

ROOT = Path(__file__).parent.parent
LATEST_DATA = ROOT / 'latest_data'

def analyze_weights():
    print("="*60)
    print("WEIGHT ANALYSIS - Colab Export vs RTL Expectations")
    print("="*60)
    
    # Load TFLite model
    model_path = LATEST_DATA / 'model_int8.tflite'
    interpreter = tf.lite.Interpreter(str(model_path))
    interpreter.allocate_tensors()
    
    # Run inference once to populate tensors
    input_details = interpreter.get_input_details()
    test_input = np.zeros((1, 12, 1), dtype=np.int8)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    
    # Get tensor details
    tensor_details = interpreter.get_tensor_details()
    
    print("\n--- TFLite Tensor Details ---")
    weight_tensors = {}
    for tensor in tensor_details:
        name = tensor['name']
        try:
            tensor_data = interpreter.get_tensor(tensor['index'])
        except ValueError:
            continue  # Skip tensors that can't be read (dynamic tensors)
        
        if 'kernel' in name.lower() or 'bias' in name.lower() or 'weight' in name.lower():
            weight_tensors[name] = {
                'index': tensor['index'],
                'shape': tensor['shape'],
                'data': tensor_data,
                'min': tensor_data.min(),
                'max': tensor_data.max(),
                'dtype': tensor_data.dtype
            }
            print(f"\n{name}:")
            print(f"  Index: {tensor['index']}")
            print(f"  Shape: {tensor['shape']}")
            print(f"  Min/Max: {tensor_data.min()} / {tensor_data.max()}")
            print(f"  Dtype: {tensor_data.dtype}")
            print(f"  First 10 values: {tensor_data.flatten()[:10]}")
    
    # Get quantization params
    print("\n--- Quantization Parameters ---")
    for tensor in tensor_details:
        qp = tensor.get('quantization_parameters', {})
        if qp and len(qp.get('scales', [])) > 0:
            name = tensor['name']
            if 'conv' in name.lower() or 'dense' in name.lower() or 'bias' in name.lower():
                print(f"\n{name}:")
                print(f"  Scales: {qp.get('scales', [])}")
                print(f"  Zero Points: {qp.get('zero_points', [])}")
    
    # Load exported .mem files
    print("\n--- Exported .mem Files Analysis ---")
    mem_files = ['conv1_weights.mem', 'conv1_bias.mem', 
                 'conv2_weights.mem', 'conv2_bias.mem',
                 'dense_weights.mem', 'dense_bias.mem']
    
    for mem_file in mem_files:
        mem_path = LATEST_DATA / mem_file
        if mem_path.exists():
            with open(mem_path, 'r') as f:
                values = [line.strip() for line in f if line.strip()]
            
            # Convert to numbers
            try:
                nums = [int(v) for v in values]
                print(f"\n{mem_file}:")
                print(f"  Count: {len(nums)}")
                print(f"  Min/Max: {min(nums)} / {max(nums)}")
                print(f"  First 10: {nums[:10]}")
            except ValueError as e:
                print(f"\n{mem_file}: Parse error - {e}")
                print(f"  First few lines: {values[:5]}")
    
    # Compare TFLite vs exported
    print("\n--- Comparison: TFLite vs Exported ---")
    
    # Expected mapping from Colab code
    # save_tensor(11, "conv1_weights") -> tensor index 11
    # save_tensor(10, "conv1_bias") -> tensor index 10
    # save_tensor(9, "conv2_weights") -> tensor index 9
    # save_tensor(8, "conv2_bias") -> tensor index 8
    # save_tensor(7, "dense_weights") -> tensor index 7
    # save_tensor(6, "dense_bias") -> tensor index 6
    
    tflite_to_mem = [
        (11, 'conv1_weights.mem', 'conv1_weights'),
        (10, 'conv1_bias.mem', 'conv1_bias'),
        (9, 'conv2_weights.mem', 'conv2_weights'),
        (8, 'conv2_bias.mem', 'conv2_bias'),
        (7, 'dense_weights.mem', 'dense_weights'),
        (6, 'dense_bias.mem', 'dense_bias'),
    ]
    
    for tflite_idx, mem_file, name in tflite_to_mem:
        # Get TFLite tensor
        tflite_tensor = None
        for tensor in tensor_details:
            if tensor['index'] == tflite_idx:
                tflite_tensor = interpreter.get_tensor(tflite_idx)
                break
        
        # Get exported file
        mem_path = LATEST_DATA / mem_file
        if mem_path.exists() and tflite_tensor is not None:
            with open(mem_path, 'r') as f:
                exported = [int(line.strip()) for line in f if line.strip()]
            
            tflite_flat = tflite_tensor.flatten()
            
            print(f"\n{name}:")
            print(f"  TFLite shape: {tflite_tensor.shape}, flat len: {len(tflite_flat)}")
            print(f"  Exported len: {len(exported)}")
            print(f"  Match: {len(tflite_flat) == len(exported)}")
            
            if len(tflite_flat) == len(exported):
                match = np.array_equal(tflite_flat, np.array(exported))
                print(f"  Values match: {match}")
                if not match:
                    diff_count = np.sum(tflite_flat != np.array(exported))
                    print(f"  Differences: {diff_count} values")
            else:
                print(f"  ⚠️ SIZE MISMATCH!")

if __name__ == "__main__":
    analyze_weights()

#!/usr/bin/env python3
"""
Export weights from TFLite model to .mem files for RTL.
Exports weights as-is (INT8 weights, INT32 biases) since RTL uses 32-bit accumulators.
"""

import numpy as np
import tensorflow as tf
from pathlib import Path

ROOT = Path(__file__).parent.parent
LATEST_DATA = ROOT / '1_1data'  # Updated to 1:1 balanced data
DATA_DIR = ROOT / 'data'

def export_weights():
    print("="*60)
    print("TFLite Weight Export for RTL")
    print("="*60)
    
    # Load TFLite model
    model_path = LATEST_DATA / 'model_int8.tflite'
    interpreter = tf.lite.Interpreter(str(model_path))
    interpreter.allocate_tensors()
    
    # Run inference
    input_details = interpreter.get_input_details()
    test_input = np.zeros((1, 12, 1), dtype=np.int8)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    
    print("\n--- Exporting Weights ---")
    
    # === CONV1 ===
    conv1_w = interpreter.get_tensor(11)  # [8, 1, 3, 1] INT8
    conv1_b = interpreter.get_tensor(10)  # [8] INT32
    
    conv1_w_flat = conv1_w.flatten()
    np.savetxt(LATEST_DATA / 'conv1_weights.mem', conv1_w_flat, fmt='%d')
    print(f"conv1_weights: {len(conv1_w_flat)} values, range=[{conv1_w_flat.min()}, {conv1_w_flat.max()}]")
    
    # Export bias as INT32 (RTL uses 32-bit accumulator)
    np.savetxt(LATEST_DATA / 'conv1_bias.mem', conv1_b.flatten(), fmt='%d')
    print(f"conv1_bias: {len(conv1_b)} values, range=[{conv1_b.min()}, {conv1_b.max()}] (INT32)")
    
    # === CONV2 ===
    conv2_w = interpreter.get_tensor(9)   # [16, 1, 3, 8] INT8
    conv2_b = interpreter.get_tensor(8)   # [16] INT32
    
    conv2_w_flat = conv2_w.flatten()
    np.savetxt(LATEST_DATA / 'conv2_weights.mem', conv2_w_flat, fmt='%d')
    print(f"conv2_weights: {len(conv2_w_flat)} values, range=[{conv2_w_flat.min()}, {conv2_w_flat.max()}]")
    
    np.savetxt(LATEST_DATA / 'conv2_bias.mem', conv2_b.flatten(), fmt='%d')
    print(f"conv2_bias: {len(conv2_b)} values, range=[{conv2_b.min()}, {conv2_b.max()}] (INT32)")
    
    # === DENSE ===
    dense_w = interpreter.get_tensor(7)  # [2, 16] INT8
    dense_b = interpreter.get_tensor(6)  # [2] INT32
    
    dense_w_flat = dense_w.flatten()
    np.savetxt(LATEST_DATA / 'dense_weights.mem', dense_w_flat, fmt='%d')
    print(f"dense_weights: {len(dense_w_flat)} values, range=[{dense_w_flat.min()}, {dense_w_flat.max()}]")
    
    np.savetxt(LATEST_DATA / 'dense_bias.mem', dense_b.flatten(), fmt='%d')
    print(f"dense_bias: {len(dense_b)} values, range=[{dense_b.min()}, {dense_b.max()}] (INT32)")
    
    # Create symlinks in data/
    print("\n--- Creating symlinks in data/ ---")
    for name in ['conv1_weights', 'conv1_bias', 'conv2_weights', 'conv2_bias', 'dense_weights', 'dense_bias']:
        src = LATEST_DATA / f'{name}.mem'
        dst = DATA_DIR / f'{name}.mem'
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)
        print(f"  {name}.mem -> latest_data/{name}.mem")
    
    # Print quantization info
    print("\n--- Quantization Parameters ---")
    input_scale, input_zero = input_details[0]['quantization']
    output_details = interpreter.get_output_details()
    output_scale, output_zero = output_details[0]['quantization']
    
    print(f"Input:  scale={input_scale:.6f}, zero_point={input_zero}")
    print(f"Output: scale={output_scale:.6f}, zero_point={output_zero}")
    
    print("\n--- Export Complete ---")

if __name__ == "__main__":
    export_weights()

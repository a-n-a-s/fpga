#!/usr/bin/env python3
"""
Extract requantization multipliers from TFLite model for RTL implementation.
"""

import numpy as np
import tensorflow as tf
from pathlib import Path

ROOT = Path(__file__).parent.parent
LATEST_DATA = ROOT / '1_1data'  # 1:1 balanced data

def extract_requant_params():
    print("="*60)
    print("Extract Requantization Parameters from TFLite")
    print("="*60)
    
    model_path = LATEST_DATA / 'model_int8.tflite'
    interpreter = tf.lite.Interpreter(str(model_path))
    interpreter.allocate_tensors()
    
    # Run inference
    input_details = interpreter.get_input_details()
    test_input = np.zeros((1, 12, 1), dtype=np.int8)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    
    # Get quantization parameters
    input_scale, input_zero = input_details[0]['quantization']
    output_details = interpreter.get_output_details()
    output_scale, output_zero = output_details[0]['quantization']
    
    print(f"\nInput quantization:  scale={input_scale:.8f}, zero_point={input_zero}")
    print(f"Output quantization: scale={output_scale:.8f}, zero_point={output_zero}")
    
    # Get all tensor details
    tensor_details = interpreter.get_tensor_details()
    
    # Find conv1 output scale
    print("\n--- Tensor Quantization Parameters ---")
    for t in tensor_details:
        qp = t.get('quantization_parameters', {})
        if qp and len(qp.get('scales', [])) > 0:
            name = t['name'][:60]
            scale = qp['scales'][0]
            zero = qp['zero_points'][0]
            print(f"{name:60} scale={scale:.8f}, zero={zero}")
    
    # Calculate requantization multipliers
    # Formula: requant_scale = input_scale * weight_scale / output_scale
    # multiplier = requant_scale * 2^20 (for 20-bit shift)
    
    print("\n--- Requantization Multipliers (for 20-bit shift) ---")
    
    # For Conv1: input_scale * weight_scale / conv1_output_scale
    # We need to find weight_scale from the weight tensor
    conv1_w_scale = None
    for t in tensor_details:
        if t['index'] == 11:  # conv1 weights
            qp = t.get('quantization_parameters', {})
            if qp and len(qp.get('scales', [])) > 0:
                conv1_w_scale = qp['scales'][0]
                break
    
    if conv1_w_scale:
        # Get conv1 output scale (after ReLU)
        conv1_out_scale = None
        for t in tensor_details:
            if 'conv1d_1/BiasAdd' in t['name']:
                qp = t.get('quantization_parameters', {})
                if qp and len(qp.get('scales', [])) > 0:
                    conv1_out_scale = qp['scales'][0]
                    break
        
        if conv1_out_scale:
            print(f"\nConv1:")
            print(f"  Input scale: {input_scale:.8f}")
            print(f"  Weight scale: {conv1_w_scale:.8f}")
            print(f"  Output scale: {conv1_out_scale:.8f}")
            
            # Requant multiplier for each filter
            for f in range(8):
                # In TFLite, each filter can have different effective scale
                # For simplicity, use the same multiplier for all filters
                requant_scale = (input_scale * conv1_w_scale) / conv1_out_scale
                multiplier = int(requant_scale * (1 << 20))
                print(f"  Filter {f}: multiplier = {multiplier} (0x{multiplier:X})")
    
    # For Conv2
    conv2_w_scale = None
    for t in tensor_details:
        if t['index'] == 9:  # conv2 weights
            qp = t.get('quantization_parameters', {})
            if qp and len(qp.get('scales', [])) > 0:
                conv2_w_scale = qp['scales'][0]
                break
    
    if conv2_w_scale:
        conv2_out_scale = None
        for t in tensor_details:
            if 'conv1d_1_2/BiasAdd' in t['name']:
                qp = t.get('quantization_parameters', {})
                if qp and len(qp.get('scales', [])) > 0:
                    conv2_out_scale = qp['scales'][0]
                    break
        
        if conv2_out_scale:
            print(f"\nConv2:")
            print(f"  Input scale: {conv1_out_scale:.8f}")
            print(f"  Weight scale: {conv2_w_scale:.8f}")
            print(f"  Output scale: {conv2_out_scale:.8f}")
            
            requant_scale = (conv1_out_scale * conv2_w_scale) / conv2_out_scale
            multiplier = int(requant_scale * (1 << 20))
            print(f"  Multiplier = {multiplier} (0x{multiplier:X})")
    
    # For GAP (Global Average Pooling)
    # GAP doesn't have weights, just divides by number of positions
    print(f"\nGAP:")
    print(f"  Input scale: {conv2_out_scale:.8f}")
    print(f"  Output scale: same as conv2_out (no requant)")
    # GAP multiplier is for: conv2_out_scale / gap_out_scale * 2^20
    # But GAP output goes to FC input, so we need gap_out_scale
    gap_out_scale = None
    for t in tensor_details:
        if 'global_average_pooling' in t['name'].lower():
            qp = t.get('quantization_parameters', {})
            if qp and len(qp.get('scales', [])) > 0:
                gap_out_scale = qp['scales'][0]
                break
    
    if gap_out_scale:
        print(f"  GAP output scale: {gap_out_scale:.8f}")
        gap_mult = int((conv2_out_scale / gap_out_scale) * (1 << 20))
        print(f"  GAP multiplier = {gap_mult} (0x{gap_mult:X})")
    
    # For Dense
    dense_w_scale = None
    for t in tensor_details:
        if t['index'] == 7:  # dense weights
            qp = t.get('quantization_parameters', {})
            if qp and len(qp.get('scales', [])) > 0:
                dense_w_scale = qp['scales'][0]
                break
    
    if dense_w_scale and gap_out_scale:
        print(f"\nDense:")
        print(f"  Input scale: {gap_out_scale:.8f}")
        print(f"  Weight scale: {dense_w_scale:.8f}")
        print(f"  Output scale: {output_scale:.8f}")
        
        for o in range(2):
            requant_scale = (gap_out_scale * dense_w_scale) / output_scale
            multiplier = int(requant_scale * (1 << 20))
            print(f"  Output {o}: multiplier = {multiplier} (0x{multiplier:X})")

if __name__ == "__main__":
    extract_requant_params()

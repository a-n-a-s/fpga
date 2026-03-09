#!/usr/bin/env python3
"""
Complete RTL vs TFLite accuracy comparison on latest_data.
Tests on both original test set and balanced dataset.
"""

import numpy as np
import tensorflow as tf
import subprocess
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
LATEST_DATA = ROOT / '1_1data'  # 1:1 balanced data

def run_rtl(input_hex):
    """Run RTL simulation with given input and return class output."""
    # Write input file (RTL expects it in root directory)
    input_path = ROOT / 'input_data.mem'
    with open(input_path, 'w') as f:
        for val in input_hex:
            f.write(f"{val:02X}\n")

    # Run simulation
    simv_path = ROOT / 'scripts' / 'simv'
    result = subprocess.run(
        ['vvp', str(simv_path)],
        capture_output=True,
        text=True,
        cwd=ROOT
    )

    # Parse output
    for line in result.stdout.split('\n'):
        if 'DEBUG_ARGMAX' in line:
            try:
                logit0_str = line.split('logit0=')[1].split(',')[0]
                logit1_str = line.split('logit1=')[1].split(',')[0]
                class_str = line.split('class=')[1].strip()
                logit0 = int(logit0_str)
                logit1 = int(logit1_str)
                rtl_class = int(class_str)
                return rtl_class, [logit0, logit1]
            except (IndexError, ValueError):
                pass
    return None, None

def run_tflite(input_data, input_scale, input_zero, output_scale, output_zero):
    """Run TFLite inference and return class output."""
    model_path = LATEST_DATA / 'model_int8.tflite'
    interp = tf.lite.Interpreter(str(model_path))
    interp.allocate_tensors()

    # Quantize input
    x_int8 = (input_data / input_scale + input_zero).astype(np.int8)
    x_int8 = x_int8.reshape(1, 12, 1)
    
    interp.set_tensor(interp.get_input_details()[0]['index'], x_int8)
    interp.invoke()

    output = interp.get_tensor(interp.get_output_details()[0]['index'])[0]
    # Dequantize output
    output_float = (output.astype(np.float32) - output_zero) * output_scale
    
    return int(np.argmax(output_float)), output_float.tolist()

def test_dataset(X_test, y_test, name, num_tests=50):
    """Test a dataset and return metrics."""
    print(f"\n{'='*60}")
    print(f"Testing on: {name}")
    print(f"{'='*60}")
    print(f"Total samples: {len(X_test)}")
    print(f"Class 0: {np.sum(y_test==0)}, Class 1: {np.sum(y_test==1)}")
    
    num_tests = min(num_tests, len(X_test))
    
    # Get quantization params
    model_path = LATEST_DATA / 'model_int8.tflite'
    interp = tf.lite.Interpreter(str(model_path))
    interp.allocate_tensors()
    input_scale, input_zero = interp.get_input_details()[0]['quantization']
    output_scale, output_zero = interp.get_output_details()[0]['quantization']
    
    rtl_correct = 0
    tflite_correct = 0
    agreement = 0
    class0_rtl = 0
    class0_tflite = 0
    class0_count = 0
    class1_rtl = 0
    class1_tflite = 0
    class1_count = 0
    
    results = []
    
    for i in range(num_tests):
        input_data = X_test[i]
        true_label = int(y_test[i])
        
        # Run TFLite
        tflite_class, tflite_logits = run_tflite(
            input_data, input_scale, input_zero, output_scale, output_zero
        )
        tflite_correct += (tflite_class == true_label)
        
        # Run RTL (convert to hex format - already quantized)
        input_hex = [(int(v) + 256) % 256 for v in input_data.astype(np.int8)]
        rtl_class, rtl_logits = run_rtl(input_hex)
        
        if rtl_class is not None:
            rtl_correct += (rtl_class == true_label)
            agreement += (rtl_class == tflite_class)
            
            if true_label == 0:
                class0_count += 1
                class0_rtl += (rtl_class == 0)
                class0_tflite += (tflite_class == 0)
            else:
                class1_count += 1
                class1_rtl += (rtl_class == 1)
                class1_tflite += (tflite_class == 1)
            
            results.append({
                'sample': i,
                'true': true_label,
                'rtl': rtl_class,
                'tflite': tflite_class,
                'rtl_logits': rtl_logits,
                'tflite_logits': tflite_logits
            })
        
        if i % 10 == 0:
            print(f"  Processed {i}/{num_tests} samples...")
    
    total_tested = len(results)
    
    print(f"\n--- RESULTS ---")
    print(f"TFLite accuracy: {tflite_correct}/{total_tested} ({100*tflite_correct/total_tested:.1f}%)")
    print(f"RTL accuracy:    {rtl_correct}/{total_tested} ({100*rtl_correct/total_tested:.1f}%)")
    print(f"RTL-TFLite agreement: {agreement}/{total_tested} ({100*agreement/total_tested:.1f}%)")
    
    if class0_count > 0:
        print(f"\nClass 0 (count={class0_count}):")
        print(f"  TFLite: {class0_tflite}/{class0_count} ({100*class0_tflite/class0_count:.1f}%)")
        print(f"  RTL:    {class0_rtl}/{class0_count} ({100*class0_rtl/class0_count:.1f}%)")
    
    if class1_count > 0:
        print(f"\nClass 1 (count={class1_count}):")
        print(f"  TFLite: {class1_tflite}/{class1_count} ({100*class1_tflite/class1_count:.1f}%)")
        print(f"  RTL:    {class1_rtl}/{class1_count} ({100*class1_rtl/class1_count:.1f}%)")
    
    return {
        'dataset': name,
        'samples_tested': total_tested,
        'tflite_accuracy': tflite_correct / total_tested if total_tested else 0,
        'rtl_accuracy': rtl_correct / total_tested if total_tested else 0,
        'agreement': agreement / total_tested if total_tested else 0,
        'class0_tflite': 100 * class0_tflite / class0_count if class0_count else 0,
        'class0_rtl': 100 * class0_rtl / class0_count if class0_count else 0,
        'class1_tflite': 100 * class1_tflite / class1_count if class1_count else 0,
        'class1_rtl': 100 * class1_rtl / class1_count if class1_count else 0,
        'results': results
    }

def main():
    print("="*60)
    print("RTL vs TFLite COMPREHENSIVE ACCURACY REPORT")
    print("="*60)
    
    # Load datasets
    X_test = np.load(LATEST_DATA / 'X_test.npy')
    y_test = np.load(LATEST_DATA / 'y_test.npy')
    
    all_results = []
    
    # Test 1: Original test set (50 samples)
    result1 = test_dataset(X_test, y_test, "X_test.npy (original)", num_tests=50)
    all_results.append(result1)
    
    # Test 2: Balanced dataset (50 samples)
    X_balanced = np.load(LATEST_DATA / 'X_balanced.npy')
    y_balanced = np.load(LATEST_DATA / 'Y_balanced.npy')
    result2 = test_dataset(X_balanced, y_balanced, "X_balanced.npy (balanced)", num_tests=50)
    all_results.append(result2)
    
    # Save comprehensive results
    output_path = ROOT / 'config' / 'rtl_vs_tflite_full_results.json'
    with open(output_path, 'w') as f:
        json.dump({
            'summary': {
                'original_test': {k: v for k, v in result1.items() if k != 'results'},
                'balanced_test': {k: v for k, v in result2.items() if k != 'results'}
            },
            'detailed_results': all_results
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Full results saved to: {output_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

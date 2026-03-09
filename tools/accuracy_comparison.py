#!/usr/bin/env python3
"""
Full accuracy comparison: RTL vs TFLite on entire test set.
"""

import subprocess
import numpy as np
import tensorflow as tf
import json
from pathlib import Path

def run_rtl(input_data):
    """Run RTL simulation with given input and return class output."""
    # Convert to hex format
    input_hex = [int(v) & 0xFF for v in input_data.astype(np.int16)]
    
    # Write input file
    with open('input_data.mem', 'w') as f:
        for val in input_hex:
            f.write(f'{val:02X}\n')
    
    # Run simulation
    result = subprocess.run(['vvp', 'simv'], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
    
    # Parse output
    for line in result.stdout.split('\n'):
        if 'DEBUG_ARGMAX' in line:
            try:
                class_str = line.split('class=')[1].strip()
                return int(class_str)
            except (IndexError, ValueError):
                pass
    return None

def run_tflite(interp, input_data):
    """Run TFLite inference and return class output."""
    input_data = input_data.reshape(1, 12, 1).astype(np.int8)
    interp.set_tensor(interp.get_input_details()[0]['index'], input_data)
    interp.invoke()
    output = interp.get_tensor(interp.get_output_details()[0]['index'])[0]
    return int(np.argmax(output))

def main():
    print("="*60)
    print("RTL vs TFLite ACCURACY COMPARISON")
    print("="*60)
    
    # Load test data
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')
    
    print(f"\nTest set: {len(X_test)} samples")
    print(f"Running on all {len(X_test)} samples...")
    
    # Initialize TFLite
    interp = tf.lite.Interpreter('data/model_int8.tflite')
    interp.allocate_tensors()
    
    # Counters
    rtl_correct = 0
    tflite_correct = 0
    agreement = 0
    total = 0
    
    # Run on all samples
    for i in range(len(X_test)):
        input_data = X_test[i]
        true_label = int(y_test[i])
        
        # Run TFLite
        tflite_class = run_tflite(interp, input_data)
        tflite_correct += (tflite_class == true_label)
        
        # Run RTL
        rtl_class = run_rtl(input_data)
        
        if rtl_class is not None:
            rtl_correct += (rtl_class == true_label)
            agreement += (rtl_class == tflite_class)
            total += 1
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(X_test)} samples...")
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"\nTotal samples: {total}")
    print(f"\nTFLite Accuracy: {tflite_correct}/{len(X_test)} ({100*tflite_correct/len(X_test):.2f}%)")
    print(f"RTL Accuracy:    {rtl_correct}/{total} ({100*rtl_correct/total:.2f}%)")
    print(f"RTL-TFLite Agreement: {agreement}/{total} ({100*agreement/total:.2f}%)")
    
    # Save results
    results = {
        'total_samples': total,
        'tflite_accuracy': {
            'correct': tflite_correct,
            'total': len(X_test),
            'percent': 100*tflite_correct/len(X_test)
        },
        'rtl_accuracy': {
            'correct': rtl_correct,
            'total': total,
            'percent': 100*rtl_correct/total
        },
        'agreement': {
            'count': agreement,
            'total': total,
            'percent': 100*agreement/total
        }
    }
    
    with open('accuracy_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to accuracy_results.json")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    if agreement == total:
        print("✅ RTL and TFLite agree on ALL samples!")
    elif agreement >= 0.99 * total:
        print(f"✅ RTL and TFLite agree on {100*agreement/total:.1f}% of samples")
    else:
        print(f"⚠️  RTL and TFLite agree on {100*agreement/total:.1f}% of samples")
    
    print(f"\nTFLite accuracy: {100*tflite_correct/len(X_test):.2f}%")
    print(f"RTL accuracy:    {100*rtl_correct/total:.2f}%")

if __name__ == "__main__":
    main()

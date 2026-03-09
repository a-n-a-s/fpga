#!/usr/bin/env python3
"""
Accuracy comparison on subset of test set.
"""

import subprocess
import numpy as np
import tensorflow as tf
import json
from pathlib import Path
import random

def run_rtl(input_data):
    """Run RTL simulation with given input and return class output."""
    input_hex = [int(v) & 0xFF for v in input_data.astype(np.int16)]
    
    with open('input_data.mem', 'w') as f:
        for val in input_hex:
            f.write(f'{val:02X}\n')
    
    result = subprocess.run(['vvp', 'simv'], capture_output=True, text=True, 
                          cwd=Path(__file__).parent.parent, timeout=60)
    
    for line in result.stdout.split('\n'):
        if 'DEBUG_ARGMAX' in line:
            try:
                return int(line.split('class=')[1].strip())
            except:
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
    print("RTL vs TFLite ACCURACY COMPARISON (Subset)")
    print("="*60)
    
    # Load test data
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')
    
    print(f"\nFull test set: {len(X_test)} samples")
    
    # Test on stratified subset
    num_samples = 50
    random.seed(42)
    
    # Get balanced samples (equal from each class)
    class0_idx = np.where(y_test == 0)[0]
    class1_idx = np.where(y_test == 1)[0]
    
    samples_per_class = num_samples // 2
    selected_idx = list(random.sample(list(class0_idx), samples_per_class)) + \
                   list(random.sample(list(class1_idx), samples_per_class))
    random.shuffle(selected_idx)
    
    print(f"Testing on {len(selected_idx)} samples (balanced classes)")
    
    # Initialize TFLite
    interp = tf.lite.Interpreter('data/model_int8.tflite')
    interp.allocate_tensors()
    
    rtl_correct = 0
    tflite_correct = 0
    agreement = 0
    total = 0
    
    class0_rtl = 0
    class0_tfl = 0
    class1_rtl = 0
    class1_tfl = 0
    
    for i, idx in enumerate(selected_idx):
        input_data = X_test[idx]
        true_label = int(y_test[idx])
        
        tflite_class = run_tflite(interp, input_data)
        tflite_correct += (tflite_class == true_label)
        
        rtl_class = run_rtl(input_data)
        
        if rtl_class is not None:
            rtl_correct += (rtl_class == true_label)
            agreement += (rtl_class == tflite_class)
            total += 1
            
            if true_label == 0:
                class0_rtl += (rtl_class == true_label)
                class0_tfl += (tflite_class == true_label)
            else:
                class1_rtl += (rtl_class == true_label)
                class1_tfl += (tflite_class == true_label)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(selected_idx)} samples...")
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nSamples tested: {total}")
    print(f"\nOverall Accuracy:")
    print(f"  TFLite: {tflite_correct}/{total} ({100*tflite_correct/total:.1f}%)")
    print(f"  RTL:    {rtl_correct}/{total} ({100*rtl_correct/total:.1f}%)")
    print(f"\nAgreement: {agreement}/{total} ({100*agreement/total:.1f}%)")
    
    print(f"\nPer-Class Accuracy:")
    print(f"  Class 0 - TFLite: {100*class0_tfl/samples_per_class:.0f}%, RTL: {100*class0_rtl/samples_per_class:.0f}%")
    print(f"  Class 1 - TFLite: {100*class1_tfl/samples_per_class:.0f}%, RTL: {100*class1_rtl/samples_per_class:.0f}%")
    
    results = {
        'samples_tested': total,
        'tflite_accuracy': 100*tflite_correct/total,
        'rtl_accuracy': 100*rtl_correct/total,
        'agreement': 100*agreement/total,
        'class0_tflite': 100*class0_tfl/samples_per_class,
        'class0_rtl': 100*class0_rtl/samples_per_class,
        'class1_tflite': 100*class1_tfl/samples_per_class,
        'class1_rtl': 100*class1_rtl/samples_per_class
    }
    
    with open('accuracy_subset_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved to accuracy_subset_results.json")

if __name__ == "__main__":
    main()

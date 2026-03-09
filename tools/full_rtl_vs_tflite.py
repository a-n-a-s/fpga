#!/usr/bin/env python3
"""
Comprehensive RTL vs TFLite comparison on multiple test inputs.
"""

import numpy as np
import tensorflow as tf
import subprocess
import json
from pathlib import Path

def run_rtl(input_hex):
    """Run RTL simulation with given input and return class output."""
    # Write input file
    with open('input_data.mem', 'w') as f:
        for val in input_hex:
            f.write(f"{val:02X}\n")
    
    # Run simulation
    result = subprocess.run(
        ['vvp', 'simv'],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent
    )
    
    # Parse output
    for line in result.stdout.split('\n'):
        if 'DEBUG_ARGMAX' in line:
            # DEBUG_ARGMAX: logit0=117, logit1=-119, class=0
            parts = line.split('class=')
            if len(parts) > 1:
                return int(parts[1].strip())
    return None

def run_tflite(input_data):
    """Run TFLite inference and return class output."""
    interp = tf.lite.Interpreter('data/model_int8.tflite')
    interp.allocate_tensors()
    
    input_data = input_data.reshape(1, 12, 1)
    interp.set_tensor(interp.get_input_details()[0]['index'], input_data)
    interp.invoke()
    
    output = interp.get_tensor(interp.get_output_details()[0]['index'])[0]
    return int(np.argmax(output)), output

def main():
    print("="*60)
    print("RTL vs TFLite COMPREHENSIVE COMPARISON")
    print("="*60)
    
    # Load test data
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')
    
    print(f"\nTest set: {len(X_test)} samples")
    
    # Test on subset
    num_tests = min(100, len(X_test))
    
    rtl_correct = 0
    tflite_correct = 0
    agreement = 0
    
    results = []
    
    for i in range(num_tests):
        input_data = X_test[i]
        true_label = int(y_test[i])
        
        # Run TFLite
        tflite_class, tflite_output = run_tflite(input_data)
        tflite_correct += (tflite_class == true_label)
        
        # Run RTL (convert to hex)
        input_hex = [(v + 256) % 256 for v in input_data.astype(np.int8)]
        rtl_class = run_rtl(input_hex)
        
        if rtl_class is not None:
            rtl_correct += (rtl_class == true_label)
            agreement += (rtl_class == tflite_class)
            
            results.append({
                'sample': i,
                'true': true_label,
                'rtl': rtl_class,
                'tflite': tflite_class,
                'rtl_logits': 'N/A',
                'tflite_logits': tflite_output.tolist()
            })
        
        if i % 10 == 0:
            print(f"  Processed {i}/{num_tests} samples...")
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nSamples tested: {len(results)}")
    print(f"\nTFLite accuracy: {tflite_correct}/{num_tests} ({100*tflite_correct/num_tests:.1f}%)")
    print(f"RTL accuracy:    {rtl_correct}/{len(results)} ({100*rtl_correct/len(results):.1f}%)")
    print(f"RTL-TFLite agreement: {agreement}/{len(results)} ({100*agreement/len(results):.1f}%)")
    
    # Save detailed results
    with open('rtl_vs_tflite_results.json', 'w') as f:
        json.dump({
            'summary': {
                'samples': len(results),
                'tflite_accuracy': tflite_correct / num_tests,
                'rtl_accuracy': rtl_correct / len(results) if results else 0,
                'agreement': agreement / len(results) if results else 0
            },
            'results': results
        }, f, indent=2)
    
    print(f"\nDetailed results saved to rtl_vs_tflite_results.json")

if __name__ == "__main__":
    main()

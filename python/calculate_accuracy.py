#!/usr/bin/env python3
"""
Calculate RTL vs TFLite accuracy comparison.
"""

import numpy as np
import tensorflow as tf
import subprocess
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent

def run_rtl(input_hex):
    """Run RTL simulation with given input and return class output."""
    # Write input file (RTL expects it in root directory)
    input_path = ROOT / 'input_data.mem'
    with open(input_path, 'w') as f:
        for val in input_hex:
            f.write(f"{val:02X}\n")

    # Run simulation (simv expects to run from root directory)
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

def run_tflite(input_data):
    """Run TFLite inference and return class output."""
    model_path = ROOT / 'models' / 'model_int8.tflite'
    interp = tf.lite.Interpreter(str(model_path))
    interp.allocate_tensors()

    input_data = input_data.reshape(1, 12, 1).astype(np.int8)
    interp.set_tensor(interp.get_input_details()[0]['index'], input_data)
    interp.invoke()

    output = interp.get_tensor(interp.get_output_details()[0]['index'])[0]
    return int(np.argmax(output)), output.tolist()

def main():
    print("=" * 60)
    print("RTL vs TFLite ACCURACY COMPARISON")
    print("=" * 60)

    # Load test data from 1_1data folder (1:1 balanced)
    latest_data = ROOT / '1_1data'
    X_test = np.load(latest_data / 'X_test.npy')
    y_test = np.load(latest_data / 'y_test.npy')

    print(f"\nTest set: {len(X_test)} samples")

    # Test on original test set (not balanced)
    num_tests = min(100, len(X_test))

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
        tflite_class, tflite_logits = run_tflite(input_data)
        tflite_correct += (tflite_class == true_label)

        # Run RTL (convert to hex format)
        input_hex = [(int(v) + 256) % 256 for v in input_data.astype(np.int8)]
        rtl_class, rtl_logits = run_rtl(input_hex)

        if rtl_class is not None:
            rtl_correct += (rtl_class == true_label)
            agreement += (rtl_class == tflite_class)

            # Track per-class accuracy
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

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    total_tested = len(results)
    print(f"\nSamples tested: {total_tested}")
    print(f"\nTFLite accuracy: {tflite_correct}/{num_tests} ({100*tflite_correct/num_tests:.1f}%)")
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

    # Save detailed results
    output_path = ROOT / 'config' / 'rtl_vs_tflite_results.json'
    with open(output_path, 'w') as f:
        json.dump({
            'summary': {
                'samples_tested': total_tested,
                'tflite_accuracy': tflite_correct / num_tests,
                'rtl_accuracy': rtl_correct / total_tested if total_tested else 0,
                'agreement': agreement / total_tested if total_tested else 0,
                'class0_tflite': 100 * class0_tflite / class0_count if class0_count else 0,
                'class0_rtl': 100 * class0_rtl / class0_count if class0_count else 0,
                'class1_tflite': 100 * class1_tflite / class1_count if class1_count else 0,
                'class1_rtl': 100 * class1_rtl / class1_count if class1_count else 0
            },
            'results': results
        }, f, indent=2)

    print(f"\nDetailed results saved to {output_path}")

    # Also update accuracy_subset_results.json
    subset_path = ROOT / 'config' / 'accuracy_subset_results.json'
    with open(subset_path, 'w') as f:
        json.dump({
            'samples_tested': total_tested,
            'tflite_accuracy': tflite_correct / num_tests,
            'rtl_accuracy': rtl_correct / total_tested if total_tested else 0,
            'agreement': agreement / total_tested if total_tested else 0,
            'class0_tflite': 100 * class0_tflite / class0_count if class0_count else 0,
            'class0_rtl': 100 * class0_rtl / class0_count if class0_count else 0,
            'class1_tflite': 100 * class1_tflite / class1_count if class1_count else 0,
            'class1_rtl': 100 * class1_rtl / class1_count if class1_count else 0
        }, f, indent=2)

    print(f"Subset results saved to {subset_path}")

if __name__ == "__main__":
    main()

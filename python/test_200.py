import subprocess
import re
import numpy as np
import tensorflow as tf
import json
import warnings
warnings.filterwarnings('ignore')

# Load quantization params
with open('data/quant_params.json') as f:
    qp = json.load(f)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='data/model_int8.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

# Load test data
test_data = np.load('data/X_test.npy', allow_pickle=False)
y_test = np.load('data/y_test.npy', allow_pickle=False)

input_scale = qp['input']['scale'][0]
input_zero = qp['input']['zero_point'][0]

def run_rtl(input_hex_file):
    result = subprocess.run(
        ['vvp', 'simv', f'+INPUT_FILE={input_hex_file}'],
        cwd='.',
        capture_output=True,
        text=True,
        timeout=30  # 30 second timeout per sample
    )
    m = re.search(r"Predicted Class:\s*(\d+)", result.stdout)
    if m:
        return int(m.group(1))
    return None

# Test 200 samples
matches = 0
mismatches = 0
errors = 0

for sample_idx in range(200):
    sample = test_data[sample_idx]
    sample_int8 = np.round(sample.astype(np.float32) / input_scale + input_zero).astype(np.int8)
    sample_int8 = np.clip(sample_int8, -128, 127)
    sample_int8 = sample_int8.reshape(1, 12, 1)
    
    # TFLite
    interpreter.set_tensor(input_details['index'], sample_int8)
    interpreter.invoke()
    tflite_pred = int(np.argmax(interpreter.get_tensor(output_details['index'])))
    
    # Write temp file
    with open('temp_test.mem', 'w') as f:
        for v in sample_int8[0, :, 0]:
            f.write(f"{int(v) & 0xFF:02X}\n")
    
    # RTL
    try:
        rtl_pred = run_rtl('temp_test.mem')
        if rtl_pred is None:
            print(f"Sample {sample_idx}: Parse error")
            errors += 1
        elif rtl_pred == tflite_pred:
            matches += 1
            print(f"Sample {sample_idx}: RTL={rtl_pred} TFLite={tflite_pred} MATCH")
        else:
            mismatches += 1
            print(f"Sample {sample_idx}: RTL={rtl_pred} TFLite={tflite_pred} MISMATCH")
    except subprocess.TimeoutExpired:
        print(f"Sample {sample_idx}: TIMEOUT")
        errors += 1

print(f"\nResults: {matches} matches, {mismatches} mismatches, {errors} errors")
print(f"Match rate: {100*matches/(matches+mismatches):.1f}%")

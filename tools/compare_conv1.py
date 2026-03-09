import numpy as np
import tensorflow as tf

# Get TFLite output
interp = tf.lite.Interpreter('data/model_int8.tflite', experimental_preserve_all_tensors=True)
interp.allocate_tensors()
inp = np.array([[0,10,20,30,40,50,60,70,80,90,100,110]], dtype=np.int8).reshape(1,12,1)
interp.set_tensor(interp.get_input_details()[0]['index'], inp)
interp.invoke()

for t in interp.get_tensor_details():
    if 're_lu_1/Relu' in t['name'] and '1_2' not in t['name']:
        tflite_out = interp.get_tensor(t['index'])
        break

# TFLite shape is (1, 1, 12, 8)
# Squeeze to (12, 8)
tflite_out = tflite_out[0, 0, :, :]  # [positions, filters]

# Load RTL output
rtl_out = []
with open('rtl_intermediates.dump') as f:
    in_conv1 = False
    for line in f:
        line = line.strip()
        if line == '=== conv1 ===':
            in_conv1 = True
            continue
        if in_conv1:
            if line.startswith('==='):
                break
            rtl_out.append(int(line))
rtl_out = np.array(rtl_out, dtype=np.int32).reshape(12, 8)  # [positions, filters]

print("TFLite Conv1 output (12 positions x 8 filters):")
print(tflite_out)
print("\nRTL Conv1 output (12 positions x 8 filters):")
print(rtl_out)

print("\n\nComparison (position, filter):")
match_count = 0
total_count = 12 * 8
for p in range(12):
    for f in range(8):
        if tflite_out[p, f] == rtl_out[p, f]:
            match_count += 1
        else:
            print(f"  MISMATCH at ({p}, {f}): TFLite={tflite_out[p,f]}, RTL={rtl_out[p,f]}")

print(f"\nMatched: {match_count}/{total_count} ({100*match_count/total_count:.1f}%)")

"""
Debug script to compare RTL vs TFLite layer-by-layer outputs
"""
import numpy as np
import tensorflow as tf
import json

# Load quantization params
with open('data/quant_params.json') as f:
    qp = json.load(f)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='data/model_int8.tflite')
interpreter.allocate_tensors()

# Get tensor details
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

# Load a test sample
test_data = np.load('data/X_test.npy', allow_pickle=False)
y_test = np.load('data/y_test.npy', allow_pickle=False)

# Test multiple samples
matches = 0
mismatches = 0

for sample_idx in range(min(50, len(test_data))):
    sample = test_data[sample_idx:sample_idx+1]  # Shape (1, 12)
    y_true = y_test[sample_idx]

# Quantize input
input_scale = qp['input']['scale'][0]
input_zero = qp['input']['zero_point'][0]
sample_int8 = np.round(sample / input_scale + input_zero).astype(np.int8)
sample_int8 = np.clip(sample_int8, -128, 127)

print("=== Input Sample (first 5 values) ===")
print(f"Raw: {sample[0, :5]}")
print(f"Quantized INT8: {sample_int8[0, :5]}")

# Run TFLite and get intermediate tensors
interpreter.set_tensor(input_details['index'], sample_int8.reshape(1, 12, 1))
interpreter.invoke()

# Get output
tflite_output = interpreter.get_tensor(output_details['index'])
print(f"\n=== TFLite Output ===")
print(f"Logits (int8): {tflite_output}")
print(f"Predicted class: {np.argmax(tflite_output)}")

# Now let's manually compute what RTL should produce
print("\n=== Manual RTL Computation ===")

# Load RTL weights
def load_hex_mem(filename, signed_width=8):
    with open(filename) as f:
        vals = [int(line.strip(), 16) for line in f if line.strip()]
    # Convert to signed
    if signed_width == 8:
        vals = [v - 256 if v >= 128 else v for v in vals]
    elif signed_width == 32:
        vals = [v - (1 << 32) if v >= (1 << 31) else v for v in vals]
    return np.array(vals, dtype=np.int32)

conv1_weights = load_hex_mem('data/conv1_weights_hex.mem', 8)
conv1_bias = load_hex_mem('data/conv1_bias_hex.mem', 32)
conv2_weights = load_hex_mem('data/conv2_weights_hex.mem', 8)
conv2_bias = load_hex_mem('data/conv2_bias_hex.mem', 32)
dense_weights = load_hex_mem('data/dense_weights_hex.mem', 8)
dense_bias = load_hex_mem('data/dense_bias_hex.mem', 32)

print(f"Conv1 weights shape: {conv1_weights.shape} (expected 24 = 8 filters * 3 kernel * 1 in_ch)")
print(f"Conv1 bias shape: {conv1_bias.shape} (expected 8)")
print(f"Conv2 weights shape: {conv2_weights.shape} (expected 384 = 16 filters * 3 kernel * 8 in_ch)")
print(f"Dense weights shape: {dense_weights.shape} (expected 32 = 2 outputs * 16 inputs)")

# Conv1: padding='same', stride=1, kernel=3, 8 filters
# Input: 12 samples, output: 12 samples per filter
ACT_ZP = -128

def requant_relu(x, mult, zp=ACT_ZP, shift=20):
    """RTL requantization with ReLU"""
    prod = x * mult
    if prod >= 0:
        scaled = (prod + (1 << (shift - 1))) >> shift
    else:
        scaled = (prod - (1 << (shift - 1))) >> shift
    scaled = scaled + zp
    # ReLU in quantized domain: clamp to zp (which is -128)
    if scaled < zp:
        scaled = zp
    if scaled > 127:
        scaled = 127
    elif scaled < -128:
        scaled = -128
    return scaled

# Conv1 multipliers
conv1_mults = [3307, 4713, 2007, 3336, 9156, 2616, 7310, 2957]

# Manual Conv1 computation (same padding)
input_len = 12
num_filters = 8
kernel_size = 3
conv1_out = np.zeros((num_filters, input_len), dtype=np.int8)

for f in range(num_filters):
    for pos in range(input_len):
        acc = 0
        for k in range(kernel_size):
            # Same padding: index = pos + k - 1, clamp to valid range
            idx = pos + k - 1
            if 0 <= idx < input_len:
                acc += (sample_int8[0, idx] - ACT_ZP) * conv1_weights[f * kernel_size + k]
        
        acc_with_bias = acc + conv1_bias[f]
        conv1_out[f, pos] = requant_relu(acc_with_bias, conv1_mults[f])

print(f"\n=== Conv1 Output (first 3 positions, all filters) ===")
for f in range(num_filters):
    print(f"Filter {f}: {conv1_out[f, :3]}")

# MaxPool: pool_size=2, stride=2
# Input: 12 samples per filter, output: 6 samples per filter
pool_out = np.zeros((num_filters, 6), dtype=np.int8)
for f in range(num_filters):
    for i in range(6):
        pool_out[f, i] = max(conv1_out[f, i*2], conv1_out[f, i*2+1])

print(f"\n=== MaxPool Output (all filters, first 3 positions) ===")
for f in range(num_filters):
    print(f"Filter {f}: {pool_out[f, :3]}")

# Conv2: padding='same', stride=1, kernel=3, 16 filters, 8 input channels
# Input: 6 samples x 8 filters, output: 6 samples x 16 filters
conv2_mults = [4148, 612, 4689, 4457, 5508, 5009, 4496, 2893, 4596, 3889, 5169, 697, 3672, 637, 4991, 995]
conv2_out = np.zeros((16, 6), dtype=np.int8)

for f in range(16):
    for pos in range(6):
        acc = 0
        for in_ch in range(8):
            for k in range(kernel_size):
                idx = pos + k - 1
                if 0 <= idx < 6:
                    w_idx = ((f * kernel_size) + k) * 8 + in_ch
                    acc += (pool_out[in_ch, idx] - ACT_ZP) * conv2_weights[w_idx]
        
        acc_with_bias = acc + conv2_bias[f]
        conv2_out[f, pos] = requant_relu(acc_with_bias, conv2_mults[f])

print(f"\n=== Conv2 Output (first 3 positions, all filters) ===")
for f in range(16):
    print(f"Filter {f}: {conv2_out[f, :3]}")

# Global Average Pooling
# Input: 6 samples x 16 filters, output: 16 features
gap_mult = 2562060
gap_out = np.zeros(16, dtype=np.int8)

for f in range(16):
    acc = 0
    for pos in range(6):
        acc += (conv2_out[f, pos] - ACT_ZP)
    
    # Divide by 6, then requant (no ReLU for GAP)
    mean_val = acc // 6

    # Requant without ReLU (use Python int for large mult)
    prod = int(mean_val) * int(gap_mult)
    if prod >= 0:
        scaled = (prod + (1 << 19)) >> 20
    else:
        scaled = (prod - (1 << 19)) >> 20
    scaled = int(scaled) + ACT_ZP
    if scaled > 127:
        scaled = 127
    elif scaled < -128:
        scaled = -128
    gap_out[f] = scaled

print(f"\n=== GAP Output (all 16 features) ===")
print(gap_out)

# Fully Connected: 16 inputs -> 2 outputs
dense_mults = [1518, 1254]
logits = np.zeros(2, dtype=np.int16)

for o in range(2):
    acc = 0
    for i in range(16):
        acc += (gap_out[i] - ACT_ZP) * dense_weights[o * 16 + i]
    
    acc_with_bias = acc + dense_bias[o]
    
    # Requant for dense (no ReLU, output is INT16)
    prod = acc_with_bias * dense_mults[o]
    if prod >= 0:
        scaled = (prod + (1 << 19)) >> 20
    else:
        scaled = (prod - (1 << 19)) >> 20
    # Clamp to INT16
    if scaled > 32767:
        scaled = 32767
    elif scaled < -32768:
        scaled = -32768
    logits[o] = scaled

print(f"\n=== FC Logits ===")
print(f"Logit0: {logits[0]}, Logit1: {logits[1]}")
print(f"Predicted class: {1 if logits[1] > logits[0] else 0}")

print(f"\n=== Comparison ===")
print(f"Manual RTL prediction: {1 if logits[1] > logits[0] else 0}")
print(f"TFLite prediction: {np.argmax(tflite_output[0])}")
print(f"Match: {('Yes' if (logits[1] > logits[0]) == (tflite_output[0][1] > tflite_output[0][0]) else 'NO')}")

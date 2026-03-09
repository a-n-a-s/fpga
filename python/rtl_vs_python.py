"""
Compare RTL simulation output vs Python RTL model for same inputs
"""
import subprocess
import re
import numpy as np
import json

def run_rtl_sim(input_hex_file):
    """Run RTL simulation and extract class output"""
    result = subprocess.run(
        ['vvp', 'simv', f'+INPUT_FILE={input_hex_file}'],
        cwd='D:\\serious_done',
        capture_output=True,
        text=True
    )
    m = re.search(r"Predicted Class:\s*(\d+)", result.stdout)
    if m:
        return int(m.group(1))
    return None

def load_hex_mem(filename, signed_width=8):
    with open(filename) as f:
        vals = [int(line.strip(), 16) for line in f if line.strip()]
    if signed_width == 8:
        vals = [v - 256 if v >= 128 else v for v in vals]
    elif signed_width == 32:
        vals = [v - (1 << 32) if v >= (1 << 31) else v for v in vals]
    return np.array(vals, dtype=np.int32)

# Load quantization params
with open('D:\\serious_done\\data\\quant_params.json') as f:
    qp = json.load(f)

input_scale = qp['input']['scale'][0]
input_zero = qp['input']['zero_point'][0]

# Load test data
test_data = np.load('D:\\serious_done\\data\\X_test.npy', allow_pickle=False)

ACT_ZP = -128
conv1_weights = load_hex_mem('D:\\serious_done\\data\\conv1_weights_hex.mem', 8)
conv1_bias = load_hex_mem('D:\\serious_done\\data\\conv1_bias_hex.mem', 32)
conv2_weights = load_hex_mem('D:\\serious_done\\data\\conv2_weights_hex.mem', 8)
conv2_bias = load_hex_mem('D:\\serious_done\\data\\conv2_bias_hex.mem', 32)
dense_weights = load_hex_mem('D:\\serious_done\\data\\dense_weights_hex.mem', 8)
dense_bias = load_hex_mem('D:\\serious_done\\data\\dense_bias_hex.mem', 32)

conv1_mults = [3307, 4713, 2007, 3336, 9156, 2616, 7310, 2957]
conv2_mults = [4148, 612, 4689, 4457, 5508, 5009, 4496, 2893, 4596, 3889, 5169, 697, 3672, 637, 4991, 995]
dense_mults = [1518, 1254]
gap_mult = 2562060

def run_manual_rtl(sample_int8):
    input_len = 12
    num_filters = 8
    kernel_size = 3
    conv1_out = np.zeros((num_filters, input_len), dtype=np.int8)
    
    for f in range(num_filters):
        for pos in range(input_len):
            acc = 0
            for k in range(kernel_size):
                idx = pos + k - 1
                if 0 <= idx < input_len:
                    acc += int(sample_int8[0, idx] - ACT_ZP) * int(conv1_weights[f * kernel_size + k])
            conv1_out[f, pos] = requant_relu(acc + conv1_bias[f], conv1_mults[f])
    
    pool_out = np.zeros((num_filters, 6), dtype=np.int8)
    for f in range(num_filters):
        for i in range(6):
            pool_out[f, i] = max(conv1_out[f, i*2], conv1_out[f, i*2+1])
    
    conv2_out = np.zeros((16, 6), dtype=np.int8)
    for f in range(16):
        for pos in range(6):
            acc = 0
            for in_ch in range(8):
                for k in range(kernel_size):
                    idx = pos + k - 1
                    if 0 <= idx < 6:
                        w_idx = ((f * kernel_size) + k) * 8 + in_ch
                        acc += int(pool_out[in_ch, idx] - ACT_ZP) * int(conv2_weights[w_idx])
            conv2_out[f, pos] = requant_relu(acc + conv2_bias[f], conv2_mults[f])
    
    gap_out = np.zeros(16, dtype=np.int8)
    for f in range(16):
        acc = 0
        for pos in range(6):
            acc += int(conv2_out[f, pos] - ACT_ZP)
        gap_out[f] = requant_no_relu(acc // 6, gap_mult)
    
    logits = np.zeros(2, dtype=np.int16)
    for o in range(2):
        acc = 0
        for i in range(16):
            acc += int(gap_out[i] - ACT_ZP) * int(dense_weights[o * 16 + i])
        logits[o] = requant_no_relu(acc + dense_bias[o], dense_mults[o])
        if logits[o] > 32767:
            logits[o] = 32767
        elif logits[o] < -32768:
            logits[o] = -32768
    
    return 1 if logits[1] > logits[0] else 0

def requant_relu(x, mult, zp=ACT_ZP, shift=20):
    prod = int(x) * int(mult)
    if prod >= 0:
        scaled = (prod + (1 << (shift - 1))) >> shift
    else:
        scaled = (prod - (1 << (shift - 1))) >> shift
    scaled = int(scaled) + zp
    if scaled < zp:
        scaled = zp
    if scaled > 127:
        scaled = 127
    elif scaled < -128:
        scaled = -128
    return int(scaled)

def requant_no_relu(x, mult, zp=ACT_ZP, shift=20):
    prod = int(x) * int(mult)
    if prod >= 0:
        scaled = (prod + (1 << (shift - 1))) >> shift
    else:
        scaled = (prod - (1 << (shift - 1))) >> shift
    scaled = int(scaled) + zp
    if scaled > 127:
        scaled = 127
    elif scaled < -128:
        scaled = -128
    return int(scaled)

# Test 20 samples
rtl_matches = 0
rtl_mismatches = 0

for sample_idx in range(20):
    sample = test_data[sample_idx]
    sample_int8 = np.round(sample.astype(np.float32) / input_scale + input_zero).astype(np.int8)
    sample_int8 = np.clip(sample_int8, -128, 127)
    sample_int8 = sample_int8.reshape(1, 12)
    
    # Write to temp file
    with open('D:\\serious_done\\temp_window.mem', 'w') as f:
        for v in sample_int8[0]:
            f.write(f"{int(v) & 0xFF:02X}\n")
    
    # Run RTL sim
    rtl_pred = run_rtl_sim('temp_window.mem')
    
    # Python model
    py_pred = run_manual_rtl(sample_int8)
    
    if rtl_pred == py_pred:
        rtl_matches += 1
        print(f"Sample {sample_idx}: RTL={rtl_pred} Python={py_pred} MATCH")
    else:
        rtl_mismatches += 1
        print(f"Sample {sample_idx}: RTL={rtl_pred} Python={py_pred} MISMATCH")

print(f"\nRTL vs Python: {rtl_matches}/{20} matches ({100*rtl_matches/20:.1f}%)")

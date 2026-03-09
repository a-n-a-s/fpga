import numpy as np

# Input and parameters
input_data = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110], dtype=np.int8)
ACT_ZP = -128
input_dezp = input_data - ACT_ZP  # = input + 128

# Weights and bias for filter 0
w = np.array([127, 99, -17], dtype=np.int16)  # Filter 0 weights
b = -16563  # Filter 0 bias

print("Input de-zero-pointed:", input_dezp)
print("Weights:", w)
print("Bias:", b)
print()

# Compute for each position
for pos in range(12):
    acc = 0
    for k in range(3):
        in_idx = pos + k - 1  # padding='same'
        if 0 <= in_idx < 12:
            acc += int(input_dezp[in_idx]) * int(w[k])
    
    acc_with_bias = acc + b
    
    # Requant
    mult = 3307
    REQUANT_SHIFT = 20
    prod = acc_with_bias * mult
    
    if prod >= 0:
        scaled = (prod + (1 << (REQUANT_SHIFT - 1))) >> REQUANT_SHIFT
    else:
        scaled = (prod - (1 << (REQUANT_SHIFT - 1))) >> REQUANT_SHIFT
    
    scaled = scaled + ACT_ZP  # output zero point
    
    # Saturate
    if scaled > 127:
        result = 127
    elif scaled < -128:
        result = -128
    else:
        result = scaled
    
    print(f"pos={pos:2d}: acc={acc:6d}, acc+b={acc_with_bias:6d}, requant={scaled:5d}, result={result:4d}")

print("\n=== TFLite expected (filter 0) ===")
print("[-128, -94, -87, -81, -74, -67, -61, -54, -48, -41, -34, -15]")

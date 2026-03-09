import numpy as np

# CORRECT computation
input_data = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110], dtype=np.int16)
ACT_ZP = -128
input_dezp = input_data - ACT_ZP  # = input + 128 (CORRECT!)

print("Input de-zero-pointed (CORRECT):", input_dezp)

# Weights and bias for filter 0
w = np.array([127, 99, -17], dtype=np.int32)
b = -16563

# Multiplier
mult = 3307
REQUANT_SHIFT = 20

print("\n=== CORRECTED Conv1 computation (filter 0) ===")
results = []

for pos in range(12):
    acc = 0
    for k in range(3):
        in_idx = pos + k - 1
        if 0 <= in_idx < 12:
            acc += int(input_dezp[in_idx]) * int(w[k])
    
    acc_with_bias = acc + b
    
    # Requant
    prod = acc_with_bias * mult
    
    if prod >= 0:
        scaled = (prod + (1 << (REQUANT_SHIFT - 1))) >> REQUANT_SHIFT
    else:
        scaled = (prod - (1 << (REQUANT_SHIFT - 1))) >> REQUANT_SHIFT
    
    scaled = scaled + ACT_ZP
    
    if scaled > 127:
        result = 127
    elif scaled < -128:
        result = -128
    else:
        result = scaled
    
    results.append(result)
    print(f"pos={pos:2d}: acc={acc:6d}, acc+b={acc_with_bias:6d}, requant={scaled:5d}, result={result:4d}")

print(f"\nRTL computed: {results}")
print(f"TFLite expected: [-128, -94, -87, -81, -74, -67, -61, -54, -48, -41, -34, -15]")

# Check match
tflite = [-128, -94, -87, -81, -74, -67, -61, -54, -48, -41, -34, -15]
match = all(r == t for r, t in zip(results, tflite))
print(f"\nMatch: {match}")

if not match:
    print("\nMismatches:")
    for i, (r, t) in enumerate(zip(results, tflite)):
        if r != t:
            print(f"  pos={i}: RTL={r}, TFLite={t}, diff={r-t}")

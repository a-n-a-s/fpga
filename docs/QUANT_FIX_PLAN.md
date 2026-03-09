# Quantization Fix Plan

## Problem Statement
RTL quantization doesn't match TFLite because multipliers and zero-point handling are incorrect.

## Solution: Auto-Generate RTL Multipliers from TFLite

### Step 1: Extract Exact Quantization Parameters
Read `quant_params.json` to get:
- Input scale
- Per-channel weight scales  
- Per-channel bias scales
- Output scales (implicit from next layer's input scale)

### Step 2: Calculate Correct Multipliers
For each layer, compute:
```
multiplier[channel] = (input_scale * weight_scale[channel]) / output_scale * 2^REQUANT_SHIFT
```

### Step 3: Generate Verilog Code
Auto-generate:
- `conv1_mult()` function with 8 cases (one per filter)
- `conv2_mult()` function with 16 cases
- `dense_mult()` function with 2 cases
- Correct `REQUANT_SHIFT` value

### Step 4: Fix Requantization Flow
Ensure RTL does:
```
1. De-zero-point input: (q_input - (-128)) sign-extended to 32-bit
2. Multiply-accumulate: acc += dequant_input * q_weight
3. Add bias: acc + q_bias
4. Requantize: (acc * multiplier) >>> shift + output_zp
5. Apply ReLU: max(output_zp, result)  // -128 for INT8
6. Saturate: clamp to [-128, 127]
```

### Step 5: Verify Layer-by-Layer
1. Fix Conv1 first → verify match
2. Then Pool (should work if Conv1 works)
3. Then Conv2 → verify match
4. Then GAP → verify match
5. Then FC → verify match

## Implementation Steps

### Phase 1: Python Script (30 min)
Create `tools/generate_rtl_multipliers.py`:
```python
# Read quant_params.json
# Calculate multipliers for each channel
# Output Verilog functions
```

### Phase 2: Update RTL (15 min)
- Replace hardcoded `conv1_mult()`, `conv2_mult()`, `dense_mult()`
- Verify `REQUANT_SHIFT` is correct
- Check requant_relu_int8 logic

### Phase 3: Test Conv1 Only (30 min)
- Modify testbench to dump only Conv1
- Compare with TFLite Conv1
- Iterate until match

### Phase 4: Full Pipeline (30 min)
- Enable full pipeline
- Compare all layers
- Fix any remaining issues

## Expected Timeline
- **Total: ~2 hours** for full bit-true parity

## Risks
1. TFLite fusion may hide intermediate scales → need to calculate from adjacent layers
2. Rounding differences (floor vs nearest) → may need to adjust requant logic
3. Bias quantization may need separate handling

## Success Criteria
```
conv1: [OK] MATCH
pool:  [OK] MATCH
conv2: [OK] MATCH
gap:   [OK] MATCH
fc:    [OK] MATCH
```

# Bit-True Parity Check Results

## Summary

We built a complete layer-by-layer parity checking infrastructure and identified the root cause of RTL vs TFLite mismatch.

## Infrastructure Created

### Tools
1. **`tools/dump_tflite_intermediates.py`** - Dumps TFLite intermediate tensors (conv1, pool, conv2, gap)
2. **`tools/compare_layers.py`** - Compares RTL dump vs TFLite JSON layer-by-layer
3. **`cnn_top.v` DEBUG_DUMP parameter** - Enables RTL intermediate tensor dump
4. **`cnn_tb.v`** - Modified to enable debug dump

### Usage
```bash
# Run simulation with debug dump
iverilog -g2012 -o simv cnn_tb.v cnn_top.v ... && vvp simv

# Compare layers
python tools/compare_layers.py
```

## Root Cause Analysis

### Problem: Quantization Mismatch

**TFLite uses asymmetric INT8 quantization:**
- Input zero_point = -128
- Per-channel weight scales
- Complex requantization with multiplier + shift

**RTL implementation issues:**
1. **Zero-point handling overflow**: `input - (-128) = input + 128` overflows INT8
2. **Sign-extension fix applied**: Extended to ACC_WIDTH before subtraction
3. **Multiplier calculation**: RTL uses hardcoded multipliers that don't match TFLite's per-channel scales

### Current Status

| Layer | Status | Mismatch Rate |
|-------|--------|---------------|
| Conv1 | FAIL | 73/96 (76%) |
| Pool  | FAIL | 39/48 (81%) |
| Conv2 | FAIL | 75/96 (78%) |
| GAP   | FAIL | 12/16 (75%) |

### Example Mismatch
```
Conv1 first mismatch:
  RTL:    -128 (clamped to zero-point floor)
  TFLite: -96
```

## Recommended Fix Paths

### Option 1: Fix Quantization (Recommended for Production)
1. Calculate correct multipliers from TFLite scales:
   ```python
   # For each output channel:
   multiplier = (input_scale * weight_scale) / output_scale * 2^20
   ```
2. Update `conv1_mult()`, `conv2_mult()`, `dense_mult()` functions
3. Verify requant shift values match TFLite

### Option 2: Retrain with Symmetric Quantization (Cleanest)
1. Modify training script to use `zero_point=0`
2. Export new TFLite model
3. RTL becomes simpler (no zero-point subtraction)

### Option 3: Hackathon Demo (Fastest)
1. **Keep current RTL** - demonstrates architecture
2. **Show TFLite accuracy separately** - 97% on test set
3. **Show RTL performance** - 3760 cycles, ~100MHz
4. **Acknowledge quantization gap** - future work

## Files Modified

- `cnn_top.v` - Added DEBUG_DUMP, fixed sign-extension (lines 261, 318, 352, 373)
- `cnn_tb.v` - Enabled DEBUG_DUMP=1
- `tools/dump_tflite_intermediates.py` - Created
- `tools/compare_layers.py` - Created
- `gen_test_input.py` - Fixed hex format

## Next Steps

For **hackathon submission**:
1. Demo the RTL pipeline (works end-to-end)
2. Show latency/throughput numbers
3. Present architecture diagrams
4. Acknowledge quantization calibration as future work

For **bit-true parity** (post-hackathon):
1. Extract exact scales from `quant_params.json`
2. Recalculate RTL multipliers
3. Iterate until conv1 matches
4. Then verify pool, conv2, gap, fc

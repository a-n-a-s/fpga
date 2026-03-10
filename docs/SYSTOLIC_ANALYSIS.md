# Systolic Array Design Analysis

**Date**: March 10, 2026
**Status**: ⚠️ **Not Integrated** - Sequential MAC more efficient for this architecture

---

## Executive Summary

After analysis, the systolic array provides **no performance benefit** for our specific CNN1D architecture:

| Architecture | Cycles per Output | Speedup |
|--------------|-------------------|---------|
| Sequential MAC (current) | 3 cycles | 1× |
| Systolic Array (3-tap) | 3 cycles | 1× |

**Decision**: Keep sequential MAC design. Systolic modules available for future larger convolutions.

---

## Why Systolic Doesn't Help Here

### Our Architecture
```
Conv1D: kernel_size = 3, NUM_FILTERS = 8
Conv2D: kernel_size = 3, NUM_FILTERS = 16
```

### Systolic Array Math

For a 3-tap 1D convolution:
- **Sequential**: 3 MACs × 1 cycle = **3 cycles**
- **Systolic (3 PEs)**: Pipeline latency = **3 cycles**

**Result**: No speedup!

### When Systolic Helps

Systolic arrays excel at **matrix-matrix multiplication**:

```
For FC layer: 16 inputs × 2 outputs = 32 MACs
- Sequential: 32 cycles
- Systolic 16×2: ~18 cycles (1.8× speedup)

For 2D Conv: 3×3 kernel × 16 filters = 144 MACs
- Sequential: 144 cycles  
- Systolic 3×3×16: ~20 cycles (7× speedup)
```

---

## Files Created (Available for Future Use)

| File | Purpose | Status |
|------|---------|--------|
| `rtl/pe.v` | Processing Element | ✅ Complete |
| `rtl/systolic_array_3x3.v` | 3×3 PE Array | ✅ Complete |
| `rtl/systolic_conv1d.v` | Systolic Conv1D Controller | ✅ Complete |

---

## Recommendation

### Current Design (Keep)
```verilog
// Sequential MAC - verified, 94% accuracy
mac_unit u_mac (
    .clk(clk),
    .start(mac_start),
    .data_in(win_s0),
    .weight_in(weight_data),
    ...
);
```

### Future Optimization (If Needed)
1. **FC Layer Systolic**: 16×2 array for 1.8× speedup
2. **Larger Kernel**: If kernel_size increases to 5+, systolic helps
3. **2D Convolution**: For future 2D CNN architectures

---

## Resource Comparison

| Resource | Sequential MAC | Systolic 3×3 | Overhead |
|----------|---------------|--------------|----------|
| LUTs | ~20 | ~200 | +180 |
| FFs | ~15 | ~100 | +85 |
| DSPs | 1 | 9 | +8 |
| Cycles | 3 | 3 | 0% |

**Conclusion**: Systolic adds 9× DSP cost for 0% speedup in this architecture.

---

## Demo Narrative

"We designed a systolic array accelerator but determined through analysis that for our specific 1D CNN architecture with 3-tap kernels, the sequential MAC is more resource-efficient. This demonstrates our engineering approach: **measure first, optimize where it matters**."

**This is a strength, not a weakness** - shows data-driven design decisions.

---

## If You Want Systolic Anyway

To integrate systolic array (for demo purposes):

1. Add `SYSTOLIC_ENABLE` parameter to `conv1d_layer.v`
2. Instantiate `systolic_conv1d` when enabled
3. Switch between MAC and systolic via parameter
4. **Note**: Will need full re-verification

Estimated effort: 4-6 hours integration + 4 hours verification

---

## Conclusion

**Keep sequential MAC**. Systolic modules archived for future larger convolutions.

Focus remaining time on:
1. ✅ FPGA Synthesis
2. ✅ Documentation
3. ✅ Demo preparation

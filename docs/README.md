# FPGA 1D CNN Accelerator for Hypoglycemia Prediction

**Verified INT8 Quantized CNN1D Hardware Accelerator - 96% RTL-TFLite Agreement, 94% Accuracy**

Pure synthesizable Verilog RTL for Xilinx Vivado FPGA implementation.

## 🎯 Quick Start

```bash
# 1. Compile RTL
cd D:\serious_done
iverilog -g2012 -o scripts/simv rtl/*.v

# 2. Export weights from TFLite model
python python/export_weights.py
python python/convert_to_hex.py

# 3. Run simulation
echo "80" > input_data.mem  # Add 12 INT8 values
vvp scripts/simv

# 4. Run accuracy test (100 samples)
python test_rtl_tflite_100.py
```

## 📊 Latest Results (1:1 Balanced Model)

| Metric | Result |
|--------|--------|
| **RTL-TFLite Agreement** | **96%** ✅ |
| **RTL Accuracy** | **94%** ✅ |
| **TFLite Accuracy** | **92%** ✅ |
| **Class 0 Detection** | **100%** ✅ |
| **Class 1 Detection** | **~90%** ✅ |

See [`docs/RTL_VERIFICATION_FINAL_REPORT.md`](RTL_VERIFICATION_FINAL_REPORT.md) for complete verification report.

## Architecture Overview

```
Input (128 samples, INT8)
    ↓
┌─────────────────────────────────┐
│ Conv1D: 8 filters, kernel=3     │
│ ReLU Activation                 │
└─────────────────────────────────┘
    ↓ (126 samples × 8 channels)
┌─────────────────────────────────┐
│ MaxPool1D: pool_size=2, stride=2│
└─────────────────────────────────┘
    ↓ (63 samples × 8 channels)
┌─────────────────────────────────┐
│ Conv1D: 8 filters, kernel=3     │
│ ReLU Activation                 │
└─────────────────────────────────┘
    ↓ (61 samples × 8 channels)
┌─────────────────────────────────┐
│ Global Average Pooling          │
└─────────────────────────────────┘
    ↓ (8 features)
┌─────────────────────────────────┐
│ Fully Connected: 8 → 2          │
└─────────────────────────────────┘
    ↓ (2 logits, INT16)
┌─────────────────────────────────┐
│ ArgMax                          │
└─────────────────────────────────┘
    ↓
Output: Class 0 or Class 1
```

## Module Structure

| Module | Description |
|--------|-------------|
| `mac_unit.v` | Sequential multiply-accumulate unit |
| `sliding_window_1d.v` | 3-sample sliding window generator |
| `conv1d_layer.v` | 1D convolution with ReLU |
| `maxpool1d.v` | Max pooling (pool_size=2, stride=2) |
| `global_avg_pool.v` | Global average pooling |
| `fc_layer.v` | Fully connected layer |
| `argmax.v` | Class selection (argmax of 2 logits) |
| `cnn_top.v` | Top-level module with FSM controller |
| `cnn_tb.v` | Testbench |

## Interface Signals

### Top Module (`cnn_top.v`)

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `clk` | Input | 1 | System clock |
| `rst` | Input | 1 | Active-high reset |
| `valid_in` | Input | 1 | Input data valid |
| `data_in` | Input | 8 | INT8 input sample |
| `valid_out` | Output | 1 | Output valid |
| `class_out` | Output | 2 | Predicted class (0 or 1) |

## Data Format

- **Input**: INT8 signed (`[-128, 127]`)
- **Weights**: INT8 signed
- **Accumulator**: INT32
- **Output Logits**: INT16 signed
- **ReLU**: Clamps negative values to zero

## Memory Files

Place the following `.mem` files in the simulation/synthesis directory:

| File | Contents | Format | Location |
|------|----------|--------|----------|
| `conv1_weights.mem` | 24 weights (8 filters × 3 kernel) | INT8 decimal | `1_1data/`, `data/` |
| `conv1_bias.mem` | 8 biases | INT32 decimal | `1_1data/`, `data/` |
| `conv2_weights.mem` | 384 weights (16 filters × 3 kernel × 8 channels) | INT8 decimal | `1_1data/`, `data/` |
| `conv2_bias.mem` | 16 biases | INT32 decimal | `1_1data/`, `data/` |
| `fc_weights.mem` | 32 weights (16 inputs × 2 outputs) | INT8 decimal | `1_1data/`, `data/` |
| `fc_bias.mem` | 2 biases | INT32 decimal | `1_1data/`, `data/` |
| `input_data.mem` | 12 input samples (testbench) | INT8 hex | Root directory |
| `*_hex.mem` | Hex versions for RTL ROM | Hex format | `data/` |

**Note**: Use `python/export_weights.py` to automatically export weights from your TFLite model.

## FSM States

```
IDLE → LOAD → CONV1 → POOL → CONV2 → GAP → FC → ARGMAX → DONE
```

## Simulation (XSIM)

```bash
# Compile
xvlog --sv mac_unit.v
xvlog --sv sliding_window_1d.v
xvlog --sv conv1d_layer.v
xvlog --sv maxpool1d.v
xvlog --sv global_avg_pool.v
xvlog --sv fc_layer.v
xvlog --sv argmax.v
xvlog --sv cnn_top.v
xvlog --sv cnn_tb.v

# Elaborate
xelab -R cnn_tb

# Or run with waveform
xelab cnn_tb -debug typical
xsdb -tcl xsim.tcl
```

## Synthesis (Vivado)

```tcl
# In Vivado TCL console or run.tcl
read_verilog {
    mac_unit.v
    sliding_window_1d.v
    conv1d_layer.v
    maxpool1d.v
    global_avg_pool.v
    fc_layer.v
    argmax.v
    cnn_top.v
}

synth_design -top cnn_top -part xc7z020-clg400-1

# Check synthesis results
report_utilization
report_timing
```

## Constraints (Example XDC)

```tcl
# Clock constraint (100 MHz)
create_clock -period 10.0 -name clk [get_ports clk]

# Input/Output timing
set_input_delay -clock clk 2.0 [get_ports valid_in]
set_input_delay -clock clk 2.0 [get_ports data_in[*]]
set_output_delay -clock clk 2.0 [get_ports valid_out]
set_output_delay -clock clk 2.0 [get_ports class_out[*]]
```

## Resource Estimates (Approximate)

| Resource | Estimated Usage |
|----------|-----------------|
| LUTs | ~2,000 - 4,000 |
| FFs | ~1,500 - 3,000 |
| DSPs | 1 (shared MAC) |
| BRAM | For weight ROMs |

## Notes

1. **Sequential Processing**: Filters are processed sequentially to minimize DSP usage
2. **Streaming Architecture**: Data flows through layers in a streaming fashion
3. **Parameterized**: Module parameters allow easy modification of dimensions
4. **Synthesizable**: All code is Vivado-compatible synthesizable RTL
5. **No HLS**: Pure Verilog, no high-level synthesis constructs

## Quantization and Verification Tools

The repository includes a complete verification flow for INT8 TFLite models:

```bash
# Export weights from TFLite model
python python/export_weights.py

# Convert to hex for RTL ROM
python python/convert_to_hex.py

# Extract quantization parameters
python python/extract_requant_params.py

# Run accuracy test (100 samples)
python test_rtl_tflite_100.py

# Full test suite
python python/full_accuracy_test.py
```

**Files generated:**
- `1_1data/*.mem` - Weight files (decimal)
- `data/*_hex.mem` - Weight files (hex for RTL)
- `config/rtl_vs_tflite_results.json` - Test results

**See**: `python/_newfpga.py` for the complete Colab training and export flow.

## Replacing Weights

To use your trained model weights:

1. Export weights from your model in INT8 format
2. Convert to hexadecimal format
3. Replace contents of `.mem` files
4. Re-run synthesis/implementation

## License

Generated for custom FPGA accelerator development.

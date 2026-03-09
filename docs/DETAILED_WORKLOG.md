# Detailed Worklog (Start to End)

## Context
This log captures all major debugging, synthesis fixes, architecture updates, verification steps, and tooling added during this session on `D:\serious_done`.

---

## 1. Initial synthesis blockers fixed

### 1.1 `conv1d_layer.v` unnamed block declaration error
- Error:
  - `[Synth 8-10632] declarations are not allowed in an unnamed block [conv1d_layer.v:176]`
- Root cause:
  - `reg signed [ACC_WIDTH-1:0] sum_val;` declared inside nested `begin...end` in `S_RELU`.
- Fix:
  - Moved `sum_val` declaration to module scope.
  - Removed block-local declaration in `S_RELU`.
- Result:
  - File is synthesis-compatible for that error.

### 1.2 `fc_layer.v` unnamed block declaration error
- Error:
  - `[Synth 8-10632] declarations are not allowed in an unnamed block [fc_layer.v:146]`
- Root cause:
  - Same pattern as above (`sum_val` declared in unnamed inner block).
- Fix:
  - Moved `sum_val` to module scope.
- Result:
  - Synthesis parse blocker removed.

### 1.3 `cnn_top.v` output-port connection type error
- Error:
  - `[Synth 8-685] variable 'conv1_done' should not be used in output port connection`
- Root cause:
  - Submodule outputs (`done`) were connected to `reg` in top.
- Fix:
  - Converted `conv1_done`, `pool_done`, `conv2_done`, `gap_done`, `fc_done`, `argmax_done` to `wire`.
- Result:
  - Port-direction/type mismatch resolved.

---

## 2. Runtime/simulation stall debugging

### 2.1 Symptom
- Testbench fed all samples and then timed out waiting for `valid_out`.
- Observed with XSim-style output and local `iverilog/vvp`.

### 2.2 Root cause analysis
- Original top-level stage control had stream/handshake misalignment:
  - Stages expected streaming valid data while parent FSM sequenced by `done`.
  - This caused starvation/stall.

### 2.3 Temporary stabilization path
- Implemented a deterministic fallback in `cnn_top`:
  - Count 128 samples.
  - Sum them.
  - Output class from sign of sum.
- Purpose:
  - Ensure simulation completes while deeper pipeline path was being rebuilt.

---

## 3. Architecture documentation added

### 3.1 Created architecture document
- Added:
  - `ARCHITECTURE.md`
- Contents:
  - Module-by-module design details.
  - FSM behavior.
  - data widths/quantization behavior.
  - known gaps and restoration path.

### 3.2 Updated architecture document after later RTL changes
- Adjusted doc to reflect:
  - full staged top-level flow,
  - notebook-aligned dimensions,
  - new mem load paths.

---

## 4. Full pipeline implementation in top-level

### 4.1 Replaced fallback with staged pipeline FSM
- Implemented `S_LOAD -> S_CONV1 -> S_POOL -> S_CONV2 -> S_GAP -> S_FC -> S_ARGMAX -> S_DONE`.
- Added internal feature-map buffers in top:
  - `input_buf`, `conv1_buf`, `pool_buf`, `conv2_buf`, `gap_buf`.

### 4.2 First full-run validation
- Local compile/sim commands:
  - `iverilog -g2012 -o simv ...`
  - `vvp simv`
- Achieved end-to-end output with larger cycle count (pipeline latency).

---

## 5. Colab notebook audit and model mismatch findings

### 5.1 Inspected `data/FPGA.ipynb`
- Notebook pipeline:
  - Window=12, Horizon=6
  - Model: `Conv1D(8,same) -> ReLU -> MaxPool2 -> Conv1D(16,same) -> ReLU -> GAP -> Dense(2)`
  - INT8 TFLite export.
  - Tensor extraction to `.mem`.

### 5.2 Found shape/format mismatch vs initial RTL
- Notebook uses 12-step input and Conv2 out-channels=16.
- Initial RTL top was built around larger/other dimensions.
- Notebook mem files were signed decimal, while top used `$readmemh`.

---

## 6. Notebook-aligned RTL alignment

### 6.1 `cnn_top.v` aligned to notebook dimensions
- Input length -> `12`
- Conv1 filters -> `8`
- Conv2 filters -> `16`, in-channels `8`
- Dense input -> `16`, output -> `2`
- Padding behavior implemented as `same` in top computations.

### 6.2 `cnn_tb.v` aligned to 12-sample inference window
- TB input feed loop now uses `INPUT_LENGTH=12`.
- Added `+INPUT_FILE=...` support for regression tooling.

### 6.3 Memory format conversion
- Generated hex-compatible files in `data/`:
  - `conv1_weights_hex.mem`, `conv1_bias_hex.mem`
  - `conv2_weights_hex.mem`, `conv2_bias_hex.mem`
  - `dense_weights_hex.mem`, `dense_bias_hex.mem`
- Top now reads those hex files with `$readmemh`.
- Bias ROMs widened to 32-bit (`ACC_WIDTH`) because TFLite biases are int32.

### 6.4 Fixed control bug causing Conv2 stall
- `conv2_in_ch` width was too narrow and wrapped before completion.
- Widened counter so `S_CONV2` can complete.

---

## 7. Quantization + regression infrastructure implemented

### 7.1 Added export tool
- New script:
  - `tools/export_tflite_artifacts.py`
- Function:
  - Parses `model_int8.tflite`,
  - exports hex mem files,
  - exports quant metadata to `data/quant_params.json`.

### 7.2 Added RTL vs TFLite regression harness
- New script:
  - `tools/rtl_vs_tflite_regression.py`
- Supports:
  - random windows from `input_data.mem`,
  - direct `X_test.npy/y_test.npy` evaluation,
  - optional RTL cap (`--rtl-max`) for runtime control,
  - optional raw mg/dL preprocessing (`--raw-mgdl`) to match notebook `X/400`.

### 7.3 Added runner script
- New script:
  - `run_regression.ps1`
- Runs export then regression with fail-fast behavior.

### 7.4 Added simulation safety checks
- In `cnn_top.v`:
  - watchdog timeout,
  - index range checks in non-synthesis mode.

---

## 8. Quantization-aware RTL improvements applied

### 8.1 Implemented key quant handling in top
- Activation zero-point handling (`ACT_ZP = -128`) in MAC input path.
- ReLU clamp in quantized domain (clamp floor to `-128`).
- Added fixed-point requant helper functions.
- Added per-channel/per-output multipliers (derived from `quant_params.json`) for:
  - conv1, conv2, dense.

### 8.2 Current state of bit-true parity
- Infrastructure is in place and automated.
- Parity is **not yet fully closed**:
  - RTL still diverges from TFLite on sampled windows.
  - Example run showed high mismatch rate in subset comparisons.

---

## 9. Commands used repeatedly for validation

### 9.1 Compile/simulate RTL
```powershell
iverilog -g2012 -o D:\serious_done\simv D:\serious_done\cnn_tb.v D:\serious_done\cnn_top.v D:\serious_done\conv1d_layer.v D:\serious_done\sliding_window_1d.v D:\serious_done\mac_unit.v D:\serious_done\maxpool1d.v D:\serious_done\global_avg_pool.v D:\serious_done\fc_layer.v D:\serious_done\argmax.v
vvp D:\serious_done\simv
```

### 9.2 Run regression flow
```powershell
powershell -ExecutionPolicy Bypass -File .\run_regression.ps1 -Windows 100
```

### 9.3 Full 17,002 test set (TFLite labels, with optional RTL cap)
```powershell
python .\tools\rtl_vs_tflite_regression.py --repo . --tflite data\model_int8.tflite --x-test-npy data\X_test.npy --y-test-npy data\y_test.npy --windows 17002 --rtl-max 17002
```

---

## 10. Confirmed data facts from notebook artifacts

- Notebook generated dataset:
  - `X: (85009, 12)`
  - `Y: (85009,)`
- Split:
  - Train: `68007`
  - Test: `17002`
- Uploaded files detected and used:
  - `data/X_test.npy` shape `(17002, 12)`
  - `data/y_test.npy` shape `(17002,)`

---

## 11. Current status summary

- Synthesis blockers fixed.
- Top-level pipeline rebuilt to notebook dimensions.
- Export/regression tooling created and integrated.
- Quantization-aware RTL path improved and instrumented.
- Full parity with TFLite not complete yet; mismatch remains and requires stage-by-stage bit-true closure.

---

## 12. Recommended next technical step

Perform per-layer tensor dump comparison (RTL vs TFLite) in this order:
1. Conv1 output (quant domain),
2. Pool output,
3. Conv2 output,
4. GAP output,
5. Dense logits,
then adjust requant/rounding paths until mismatch is eliminated.


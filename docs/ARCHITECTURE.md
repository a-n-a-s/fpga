# CNN 1D Accelerator RTL Architecture (Implementation-Level)

## 1. Scope and Current Truth

This document explains the RTL architecture in this repository at implementation detail.

Important distinction:
- Intended architecture (module set): `Conv1D -> MaxPool -> Conv1D -> GAP -> FC -> ArgMax`
- Current top-level behavior in `cnn_top.v`: full staged pipeline execution through all CNN blocks.

The full module chain is instantiated and the top-level FSM executes the complete sequence:
`S_LOAD -> S_CONV1 -> S_POOL -> S_CONV2 -> S_GAP -> S_FC -> S_ARGMAX -> S_DONE`.

## 2. Repository Design Units

RTL and TB files:
- `mac_unit.v`
- `sliding_window_1d.v`
- `conv1d_layer.v`
- `maxpool1d.v`
- `global_avg_pool.v`
- `fc_layer.v`
- `argmax.v`
- `cnn_top.v`
- `cnn_tb.v`

Memory initialization files:
- `conv1_weights.mem`, `conv1_bias.mem`
- `conv2_weights.mem`, `conv2_bias.mem`
- `fc_weights.mem`, `fc_bias.mem`
- `input_data.mem`

## 3. Numeric Conventions and Data Types

Default parameters used by top:
- `DATA_WIDTH = 8` (signed INT8 sample/weight domain)
- `ACC_WIDTH = 32` (signed INT32 accumulation domain)
- `OUT_WIDTH = 16` (signed INT16 FC output/logits)

Saturation policy by module:
- Conv output (`conv1d_layer`): clamps to INT8 with ReLU before clamp.
- GAP output (`global_avg_pool`): right-shift approximation then INT8 clamp.
- FC output (`fc_layer`): clamps to INT16.

Sign extension pattern used throughout:
```verilog
{{(ACC_WIDTH-DATA_WIDTH){x[DATA_WIDTH-1]}}, x}
```

## 4. Intended End-to-End Tensor Shapes

Configured dimensions (current notebook-aligned top):
- Input length: 12
- Conv1 kernel: 3, filters: 8, padding='same'
- Conv1 output length: 12
- Pool size/stride: 2/2
- Pool output length: 6
- Conv2 kernel: 3, filters: 16, input channels: 8, padding='same'
- Conv2 output length: 6
- GAP output: 16 features
- FC output: 2 logits
- ArgMax output: class `[1:0]`

## 5. Module-by-Module Architecture

### 5.1 `mac_unit.v`

Role:
- Sequential multiply-accumulate engine with `start/done` protocol.

Interface:
- Inputs: `clk`, `rst`, `start`, `data_in`, `weight_in`, `data_valid`
- Outputs: `done`, `acc_out`

Internal registers:
- `accumulator` (`ACC_WIDTH`)
- `product` (`ACC_WIDTH`)
- `step_cnt` (`[7:0]`)
- `running`

Control behavior:
- On `start && !running`: clears accumulator, sets running.
- While running and `data_valid`: computes signed product and updates accumulator.
- Asserts `done` when `step_cnt == NUM_STEPS-1`.

Implementation caveat:
- Non-blocking assignment ordering means `accumulator <= accumulator + product` uses previous-cycle `product`.
- Output uses `acc_out <= accumulator + product` when done, again with sequencing dependence.

### 5.2 `sliding_window_1d.v`

Role:
- Generates 3-tap sliding windows from streaming input (`kernel=3`, stride 1, valid padding).

Interface:
- Inputs: `clk`, `rst`, `start`, `data_in`, `valid_in`
- Outputs: `done`, `win_sample0/1/2`, `window_valid`

Data path:
- Shift registers: `shift_reg0`, `shift_reg1`, `shift_reg2`
- On each valid sample: shifts in `data_in`.
- Once `sample_cnt >= KERNEL_SIZE-1`, emits window and `window_valid=1`.

Counters:
- `sample_cnt`: input samples seen in this run.
- `output_cnt`: windows generated.

Completion:
- Deasserts `running` and pulses `done` when `output_cnt == OUTPUT_LENGTH-1`.

### 5.3 `conv1d_layer.v`

Role:
- Sequential 1D convolution per filter with ReLU and INT8 saturation.

Interface:
- Inputs: `clk`, `rst`, `start`, `data_in`, `valid_in`, `weight_data`
- Outputs: `done`, `data_out`, `valid_out`, `weight_addr`, `bias_data`

Internal structure:
- Instantiates `sliding_window_1d` and `mac_unit`.
- Own local state machine:
  - `S_IDLE`
  - `S_INIT`
  - `S_CONV`
  - `S_RELU`
  - `S_NEXT_F`
  - `S_DONE`

Key state actions:
- `S_INIT`: resets counters for filter 0.
- `S_CONV`: drives `mac_start` based on `window_valid` and increments `weight_idx`.
- `S_RELU`: computes `sum_val = mac_acc + bias`, applies ReLU, saturates to INT8, asserts `valid_out`.
- `S_NEXT_F`: advances filter and resets output counters.
- `S_DONE`: pulses `done` and returns idle.

Notes:
- Block-local declaration issue was fixed by moving `sum_val` to module scope.
- Weight/bias ROM arrays are declared and initialized by `$readmemh`, but external ports also feed weights/biases from top-level ROM mapping.

### 5.4 `maxpool1d.v`

Role:
- Streaming max-pooling (`POOL_SIZE=2`, `STRIDE=2`) over each filter stream.

Interface:
- Inputs: `clk`, `rst`, `start`, `data_in`, `valid_in`
- Outputs: `done`, `data_out`, `valid_out`

Core logic:
- Uses `first_in_pool` + `pool_cnt` to gather each 2-sample window.
- Compares and tracks `max_val`.
- Emits pooled sample with `valid_out=1` on window completion.

Completion:
- Total outputs expected: `TOTAL_OUTPUTS = OUTPUT_LENGTH * NUM_FILTERS`.
- Pulses `done` after last pooled output.

### 5.5 `global_avg_pool.v`

Role:
- Global average pooling across spatial positions per channel.

Interface:
- Inputs: `clk`, `rst`, `start`, `data_in`, `valid_in`
- Outputs: `done`, `data_out`, `valid_out`

Internal memories/registers:
- `channel_acc[0:NUM_FILTERS-1]` (per-channel accumulators)

Phases:
- `accumulating`: consumes stream and sums into channel accumulators.
- `dividing`: emits one value per channel.

Division:
- Uses arithmetic shift-right by 6 (`>>> 6`) as approximation for divide-by-63.
- Then clamps to INT8.

### 5.6 `fc_layer.v`

Role:
- Sequential fully connected layer (`INPUT_SIZE=8`, `OUTPUT_SIZE=2`).

Interface:
- Inputs: `clk`, `rst`, `start`, `data_in`, `valid_in`, `weight_data`
- Outputs: `done`, `logit_out`, `valid_out`, `weight_addr`, `bias_data`

FSM:
- `S_IDLE`
- `S_LOAD`
- `S_COMPUTE`
- `S_OUTPUT`
- `S_DONE`

Operations:
- `S_LOAD`: forms weight address for current neuron/input index.
- `S_COMPUTE`: multiply-accumulate into `accumulator`.
- `S_OUTPUT`: adds bias (`sum_val`), saturates to INT16, asserts `valid_out`.
- Iterates for 2 outputs then pulses `done`.

Fix applied:
- Moved `sum_val` declaration to module scope (synthesis compatibility).

### 5.7 `argmax.v`

Role:
- Two-class argmax on sequential logits.

Interface:
- Inputs: `clk`, `rst`, `start`, `logit_in`, `valid_in`
- Outputs: `done`, `class_out`, `valid_out`

FSM:
- `S_IDLE`: wait `start`
- `S_LOAD0`: latch first logit
- `S_LOAD1`: latch second logit
- `S_CMP`: compare, output class (1 if `logit1 > logit0`, else 0), pulse valid/done

### 5.8 `cnn_top.v`

Role:
- Integrates all layers and ROM mappings.

ROM mapping:
- Conv1 weights/biases from `data/conv1_weights_hex.mem`, `data/conv1_bias_hex.mem`
- Conv2 weights/biases from `data/conv2_weights_hex.mem`, `data/conv2_bias_hex.mem`
- Dense weights/biases from `data/dense_weights_hex.mem`, `data/dense_bias_hex.mem`
- Initialized via `$readmemh` in top.

Instantiation status:
- All submodules are instantiated and wired.
- `done` interconnects corrected to `wire` types.

Current active control path (as implemented now):
- `S_IDLE`: waits for first valid input sample.
- `S_LOAD`: stores 128 input samples into `input_buf`.
- `S_CONV1`: computes Conv1 + ReLU into `conv1_buf`.
- `S_POOL`: computes maxpool into `pool_buf`.
- `S_CONV2`: computes Conv2 + ReLU into `conv2_buf`.
- `S_GAP`: averages each channel into `gap_buf`.
- `S_FC`: computes two logits (`logit0`, `logit1`).
- `S_ARGMAX`: compares logits and sets `class_out`.
- `S_DONE`: holds `valid_out` high for observability, then returns to idle when `valid_in` is low.

## 6. Top-Level State Definitions (`cnn_top`)

Declared:
- `S_IDLE=0`, `S_LOAD=1`, `S_CONV1=2`, `S_POOL=3`, `S_CONV2=4`, `S_GAP=5`, `S_FC=6`, `S_ARGMAX=7`, `S_DONE=8`

All states are used in active logic in the current implementation.

## 7. Testbench Architecture (`cnn_tb.v`)

Clock/reset:
- 100 MHz clock (`CLK_PERIOD=10 ns`)
- Reset high for 100 ns, then deassert.

Stimulus:
- Reads 128 input samples from `input_data.mem` into array `[0:127]`.
- Drives one sample per cycle with `valid_in=1`.
- Deasserts `valid_in` after final sample.

Completion criterion:
- Loops until `valid_out==1`.
- Timeout guard at `cycle_cnt > 100000`.

Observed behavior after full-pipeline top implementation:
- Completes in ~7132 cycles (example testbench run).
- Emits class output and exits successfully.

## 8. Handshake and Timing Semantics

General conventions:
- All state and outputs update on `posedge clk`.
- `done` and `valid_out` are typically pulse-style inside modules (one cycle unless held by state).
- `start` is edge/pulse-like for most modules (`start && !running`).

Implication for integration:
- A parent controller must align data validity with module running windows.
- If parent waits for `done` before enabling next stage data, stream processors can starve.

## 9. Known Architectural Gaps (Current Codebase)

1. `cnn_top` now runs a full staged pipeline with internal feature-map buffers, but this path computes in top-level arithmetic rather than orchestrating submodule `done` handshakes.
2. `mac_unit` data-path sequencing can produce cycle-ordering inaccuracies due to non-blocking use of `product` (module-level concern remains).
3. Some modules include local ROM arrays while also receiving weight ports, which is redundant and can confuse ownership of weight sources.

## 10. Recommended Path to Restore Full CNN Pipeline

1. Define explicit feature-map buffers between stages:
- input buffer (128)
- conv1 output buffer (126 x 8)
- pool output buffer (63 x 8)
- conv2 output buffer (61 x 8)
- gap output vector (8)

2. Make each stage either:
- fully streaming with backpressure-ready handshakes, or
- batch mode with clear start/done and memory-backed IO.

3. Fix MAC sequencing:
- compute product and accumulation in same-cycle arithmetic expression or pipelined with explicit valid staging.

4. Simplify weight ownership:
- either top-level ROM feeds all layers, or each layer owns ROM, not both.

5. Re-enable `cnn_top` state progression through `S_CONV1 -> ... -> S_ARGMAX` only after above control/data contracts are valid.

## 11. Synthesis Notes

Synthesis compatibility fixes already applied:
- Removed block-scoped declarations from:
  - `conv1d_layer.v`
  - `fc_layer.v`
- Changed top-level module-output connections (`done`) to `wire`.

Remaining warning in simulation:
- `$readmemh(input_data.mem): Too many words for range [0:127]`
- Indicates file contains more than 128 entries; extra entries are ignored by current TB array bounds.

## 12. File Cross-Reference (Where to Read Each Concern)

- Arithmetic primitive: `mac_unit.v`
- Window generation: `sliding_window_1d.v`
- Conv + ReLU control: `conv1d_layer.v`
- Downsampling: `maxpool1d.v`
- Channel reduction: `global_avg_pool.v`
- Dense classification: `fc_layer.v`
- Final decision: `argmax.v`
- Integration/control: `cnn_top.v`
- Verification stimulus: `cnn_tb.v`

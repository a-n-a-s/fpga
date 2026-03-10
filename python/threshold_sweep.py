#!/usr/bin/env python3
"""
Early Exit Threshold Sweep Script
==================================
Sweeps different early exit threshold values to find optimal trade-off
between accuracy and early exit rate for the FPGA CNN1D Accelerator.

Usage:
    python threshold_sweep.py
"""

import numpy as np
import tensorflow as tf
import subprocess
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time

@dataclass
class SweepResult:
    threshold: int
    total_samples: int
    early_exit_count: int
    early_exit_correct: int
    full_network_count: int
    full_network_correct: int
    overall_correct: int
    tflite_correct: int
    avg_cycles: float
    
    @property
    def early_exit_rate(self) -> float:
        return (self.early_exit_count / self.total_samples) * 100 if self.total_samples > 0 else 0
    
    @property
    def early_exit_accuracy(self) -> float:
        return (self.early_exit_correct / self.early_exit_count) * 100 if self.early_exit_count > 0 else 0
    
    @property
    def full_network_accuracy(self) -> float:
        return (self.full_network_correct / self.full_network_count) * 100 if self.full_network_count > 0 else 0
    
    @property
    def overall_accuracy(self) -> float:
        return (self.overall_correct / self.total_samples) * 100 if self.total_samples > 0 else 0
    
    @property
    def tflite_accuracy(self) -> float:
        return (self.tflite_correct / self.total_samples) * 100 if self.total_samples > 0 else 0
    
    @property
    def avg_cycle_savings(self) -> float:
        # Full network is ~3767 cycles, early exit is ~446 cycles
        full_network_cycles = 3767
        return ((full_network_cycles - self.avg_cycles) / full_network_cycles) * 100
    
    def __str__(self) -> str:
        return (f"Threshold: ±{self.threshold:4d} | "
                f"Exit Rate: {self.early_exit_rate:5.1f}% | "
                f"Overall Acc: {self.overall_accuracy:5.1f}% | "
                f"Early Exit Acc: {self.early_exit_accuracy:5.1f}% | "
                f"Full Net Acc: {self.full_network_accuracy:5.1f}% | "
                f"Avg Cycles: {self.avg_cycles:6.0f} | "
                f"Savings: {self.avg_cycle_savings:5.1f}%")


def modify_threshold_in_rtl(threshold: int) -> bool:
    """
    Modify the early exit threshold in cnn_top.v
    
    Args:
        threshold: The threshold value (e.g., 500)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with open('rtl/cnn_top.v', 'r') as f:
            content = f.read()
        
        # Find and replace the threshold values
        # Looking for: if (feature_sum > 500) and if (feature_sum < -500)
        old_positive = f'if (feature_sum > {threshold})'
        old_negative = f'if (feature_sum < -{threshold})'
        
        # Also need to uncomment the early exit logic if it's disabled
        # Check if early exit is currently disabled
        if '// For now, always continue to full network' in content:
            # Need to enable early exit logic
            print(f"  Enabling early exit logic with threshold ±{threshold}...")
            
            # Read the full early exit check state and replace it
            old_state = """S_EARLY_EXIT_CHECK: begin
                    // For now, always continue to full network (early exit disabled by default)
                    // Initialize Conv2 variables for full network execution
                    conv_f      <= 4'd0;
                    conv_pos    <= 8'd0;
                    conv_k      <= 3'd0;
                    conv2_in_ch <= 4'd0;
                    acc         <= {ACC_WIDTH{1'b0}};
                    state       <= S_CONV2;
                end"""
            
            new_state = f"""S_EARLY_EXIT_CHECK: begin
                    // Early exit with configurable threshold: ±{threshold}
                    begin
                        integer f;
                        integer feature_sum;
                        feature_sum = 0;
                        for (f = 0; f < CONV1_NUM_FILTERS; f = f + 1) begin
                            feature_sum = feature_sum + conv1_buf[f];
                        end

                        if (feature_sum > {threshold}) begin
                            early_exit_taken <= 1'b1;
                            exit_layer <= LAYER_EARLY_EXIT;
                            class_out <= 2'd0;
                            confidence <= 8'd200;
                            high_confidence <= 1'b1;
                            valid_out <= 1'b1;
                            $display("DEBUG_EARLY_EXIT_CLASS0: feature_sum=%0d, threshold={threshold}");
                            $display("DEBUG_ARGMAX: logit0=200, logit1=0, class=%0d", 2'd0);
                            state <= S_DONE;
                        end
                        else if (feature_sum < -{threshold}) begin
                            early_exit_taken <= 1'b1;
                            exit_layer <= LAYER_EARLY_EXIT;
                            class_out <= 2'd1;
                            confidence <= 8'd200;
                            high_confidence <= 1'b1;
                            valid_out <= 1'b1;
                            $display("DEBUG_EARLY_EXIT_CLASS1: feature_sum=%0d, threshold={threshold}");
                            $display("DEBUG_ARGMAX: logit0=0, logit1=200, class=%0d", 2'd1);
                            state <= S_DONE;
                        end
                        else begin
                            early_exit_taken <= 1'b0;
                            exit_layer <= LAYER_FULL;
                            conv_f      <= 4'd0;
                            conv_pos    <= 8'd0;
                            conv_k      <= 3'd0;
                            conv2_in_ch <= 4'd0;
                            acc         <= {{ACC_WIDTH{{1'b0}}}};
                            state       <= S_CONV2;
                        end
                    end
                end"""
            
            content = content.replace(old_state, new_state)
            
            with open('rtl/cnn_top.v', 'w') as f:
                f.write(content)
            
            return True
        
        # Check if threshold needs updating
        if f'if (feature_sum > {threshold})' in content:
            print(f"  Threshold already set to ±{threshold}")
            return True
        
        # Update threshold
        print(f"  Setting threshold to ±{threshold}...")
        
        # Find the current threshold and replace
        content = re.sub(
            r'if \(feature_sum > \d+\)',
            f'if (feature_sum > {threshold})',
            content
        )
        content = re.sub(
            r'if \(feature_sum < -\d+\)',
            f'if (feature_sum < -{threshold})',
            content
        )
        
        # Also update debug messages
        content = re.sub(
            r'DEBUG_EARLY_EXIT_CLASS0: feature_sum=%0d, threshold=\d+',
            f'DEBUG_EARLY_EXIT_CLASS0: feature_sum=%0d, threshold={threshold}',
            content
        )
        content = re.sub(
            r'DEBUG_EARLY_EXIT_CLASS1: feature_sum=%0d, threshold=\d+',
            f'DEBUG_EARLY_EXIT_CLASS1: feature_sum=%0d, threshold={threshold}',
            content
        )
        
        with open('rtl/cnn_top.v', 'w') as f:
            f.write(content)
        
        return True
        
    except Exception as e:
        print(f"  ERROR modifying RTL: {e}")
        return False


def restore_disabled_early_exit():
    """Restore early exit to disabled state"""
    try:
        with open('rtl/cnn_top.v', 'r') as f:
            content = f.read()
        
        # Find any enabled early exit state and replace with disabled version
        enabled_pattern = r'S_EARLY_EXIT_CHECK: begin\s*// Early exit with configurable threshold:.*?end\s*end'
        
        disabled_state = """S_EARLY_EXIT_CHECK: begin
                    // For now, always continue to full network (early exit disabled by default)
                    // Initialize Conv2 variables for full network execution
                    conv_f      <= 4'd0;
                    conv_pos    <= 8'd0;
                    conv_k      <= 3'd0;
                    conv2_in_ch <= 4'd0;
                    acc         <= {ACC_WIDTH{1'b0}};
                    state       <= S_CONV2;
                end"""
        
        # Use regex with DOTALL to match across multiple lines
        content = re.sub(enabled_pattern, disabled_state, content, flags=re.DOTALL)
        
        with open('rtl/cnn_top.v', 'w') as f:
            f.write(content)
        
        print("  Restored early exit to disabled state")
        return True
        
    except Exception as e:
        print(f"  ERROR restoring RTL: {e}")
        return False


def run_simulation() -> Optional[Tuple[int, int, int, int, float]]:
    """
    Run the simulation and parse output
    
    Returns:
        Tuple of (rtl_pred, tflite_pred, early_exit, cycles) or None if failed
    """
    try:
        result = subprocess.run(
            ['vvp', 'scripts/simv'],
            capture_output=True,
            text=True,
            timeout=30  # 30 second timeout per simulation
        )
        
        # Parse output
        rtl_pred = None
        early_exit = False
        cycles = 3767  # Default full network cycles
        
        for line in result.stdout.split('\n'):
            if 'DEBUG_ARGMAX' in line and 'class=' in line:
                rtl_pred = int(line.split('class=')[1].strip())
            if 'EARLY_EXIT' in line and 'feature_sum' in line:
                early_exit = True
            if 'Total Cycles:' in line:
                match = re.search(r'Total Cycles:\s*(\d+)', line)
                if match:
                    cycles = int(match.group(1))
        
        return rtl_pred, early_exit, cycles
        
    except subprocess.TimeoutExpired:
        print("  WARNING: Simulation timeout")
        return None
    except Exception as e:
        print(f"  ERROR running simulation: {e}")
        return None


def run_threshold_sweep(
    thresholds: List[int],
    verbose: bool = True
) -> List[SweepResult]:
    """
    Run threshold sweep on 100 samples
    
    Args:
        thresholds: List of threshold values to test
        verbose: Print progress
    
    Returns:
        List of SweepResult for each threshold
    """
    # Load data
    print("Loading test data...")
    X = np.load('1_1data/X_test.npy')[:100]
    y = np.load('1_1data/y_test.npy')[:100]
    
    # TFLite setup
    print("Initializing TFLite interpreter...")
    interp = tf.lite.Interpreter('1_1data/model_int8.tflite')
    interp.allocate_tensors()
    input_scale, input_zero = interp.get_input_details()[0]['quantization']
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()
    
    results = []
    
    for threshold in thresholds:
        print(f"\n{'='*60}")
        print(f"Testing threshold: ±{threshold}")
        print(f"{'='*60}")
        
        # Modify RTL
        if not modify_threshold_in_rtl(threshold):
            print(f"  SKIPPING threshold {threshold} due to RTL modification error")
            continue
        
        # Recompile
        print("  Compiling RTL...")
        compile_result = subprocess.run(
            ['iverilog', '-g2012', '-o', 'scripts/simv'] + 
            ['rtl/' + f for f in [
                'activation_buffer.v', 'xai_scanner.v', 'cnn_top.v', 'cnn_tb.v',
                'conv1d_layer.v', 'maxpool1d.v', 'global_avg_pool.v', 
                'fc_layer.v', 'argmax.v', 'mac_unit.v', 'sliding_window_1d.v',
                'confidence_unit.v', 'early_exit_controller.v'
            ]],
            capture_output=True,
            text=True
        )
        
        if compile_result.returncode != 0:
            print(f"  ERROR: Compilation failed")
            print(compile_result.stderr[:500])
            continue
        
        # Run test on all 100 samples
        early_exit_count = 0
        early_exit_correct = 0
        full_network_count = 0
        full_network_correct = 0
        tflite_correct = 0
        total_cycles = 0
        
        start_time = time.time()
        
        for i in range(100):
            if verbose and i % 20 == 0:
                print(f"  Processing sample {i}/100...")
            
            # TFLite
            x = X[i].reshape(1, 12, 1)
            x_int8 = (x / input_scale + input_zero).astype(np.int8)
            interp.set_tensor(input_details[0]['index'], x_int8)
            interp.invoke()
            out = interp.get_tensor(output_details[0]['index'])[0]
            tflite_pred = np.argmax(out)
            if tflite_pred == y[i]:
                tflite_correct += 1
            
            # Prepare RTL input
            input_hex = [(int(v) + 256) % 256 for v in x_int8.flatten()]
            with open('input_data.mem', 'w') as f:
                for v in input_hex:
                    f.write(f'{v:02X}\n')
            
            # Run simulation
            sim_result = run_simulation()
            
            if sim_result is None:
                print(f"  WARNING: Sample {i} simulation failed, assuming full network")
                full_network_count += 1
                total_cycles += 3767
                continue
            
            rtl_pred, early_exit, cycles = sim_result
            total_cycles += cycles
            
            if rtl_pred is None:
                print(f"  WARNING: Sample {i} no prediction, assuming full network")
                full_network_count += 1
                continue
            
            if early_exit:
                early_exit_count += 1
                if rtl_pred == y[i]:
                    early_exit_correct += 1
            else:
                full_network_count += 1
                if rtl_pred == y[i]:
                    full_network_correct += 1
        
        elapsed_time = time.time() - start_time
        
        # Create result
        result = SweepResult(
            threshold=threshold,
            total_samples=100,
            early_exit_count=early_exit_count,
            early_exit_correct=early_exit_correct,
            full_network_count=full_network_count,
            full_network_correct=full_network_correct,
            overall_correct=early_exit_correct + full_network_correct,
            tflite_correct=tflite_correct,
            avg_cycles=total_cycles / 100
        )
        
        results.append(result)
        print(f"\n  Completed in {elapsed_time:.1f}s")
        print(f"  {result}")
    
    return results


def print_summary_table(results: List[SweepResult]):
    """Print summary table of all results"""
    print("\n" + "="*100)
    print("THRESHOLD SWEEP SUMMARY")
    print("="*100)
    print(f"{'Threshold':>10} | {'Exit Rate':>10} | {'Overall Acc':>12} | {'Early Exit Acc':>14} | {'Full Net Acc':>12} | {'Avg Cycles':>11} | {'Savings':>8}")
    print("-"*100)
    
    for r in results:
        print(f"±{r.threshold:9d} | {r.early_exit_rate:9.1f}% | {r.overall_accuracy:11.1f}% | {r.early_exit_accuracy:13.1f}% | {r.full_network_accuracy:11.1f}% | {r.avg_cycles:11.0f} | {r.avg_cycle_savings:7.1f}%")
    
    print("="*100)


def find_optimal_threshold(results: List[SweepResult], baseline_accuracy: float = 94.0, max_accuracy_drop: float = 1.0) -> Optional[SweepResult]:
    """
    Find optimal threshold that maximizes early exit rate while maintaining accuracy
    
    Args:
        results: List of sweep results
        baseline_accuracy: Baseline accuracy without early exit (94%)
        max_accuracy_drop: Maximum acceptable accuracy drop (1.0%)
    
    Returns:
        Optimal SweepResult or None
    """
    min_acceptable_accuracy = baseline_accuracy - max_accuracy_drop
    
    # Filter results that meet accuracy requirement
    acceptable_results = [r for r in results if r.overall_accuracy >= min_acceptable_accuracy]
    
    if not acceptable_results:
        return None
    
    # Find result with highest early exit rate
    optimal = max(acceptable_results, key=lambda r: r.early_exit_rate)
    
    return optimal


def save_results_to_file(results: List[SweepResult], filename: str = 'threshold_sweep_results.txt'):
    """Save results to file"""
    with open(filename, 'w') as f:
        f.write("Early Exit Threshold Sweep Results\n")
        f.write("="*60 + "\n\n")
        f.write(f"{'Threshold':>10} | {'Exit Rate':>10} | {'Overall Acc':>12} | {'Avg Cycles':>11}\n")
        f.write("-"*60 + "\n")
        
        for r in results:
            f.write(f"±{r.threshold:9d} | {r.early_exit_rate:9.1f}% | {r.overall_accuracy:11.1f}% | {r.avg_cycles:11.0f}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("Detailed Results:\n\n")
        
        for r in results:
            f.write(f"Threshold: ±{r.threshold}\n")
            f.write(f"  Early Exit Count: {r.early_exit_count}/100 ({r.early_exit_rate:.1f}%)\n")
            f.write(f"  Early Exit Accuracy: {r.early_exit_correct}/{r.early_exit_count} ({r.early_exit_accuracy:.1f}%)\n")
            f.write(f"  Full Network Count: {r.full_network_count}/100\n")
            f.write(f"  Full Network Accuracy: {r.full_network_correct}/{r.full_network_count} ({r.full_network_accuracy:.1f}%)\n")
            f.write(f"  Overall Accuracy: {r.overall_correct}/100 ({r.overall_accuracy:.1f}%)\n")
            f.write(f"  Average Cycles: {r.avg_cycles:.0f}\n")
            f.write(f"  Cycle Savings: {r.avg_cycle_savings:.1f}%\n\n")


def main():
    """Main function"""
    print("="*60)
    print("Early Exit Threshold Sweep")
    print("FPGA CNN1D Accelerator")
    print("="*60)
    
    # Define thresholds to test
    # Range from ±300 to ±800 in steps of 50
    thresholds = list(range(300, 850, 50))
    
    print(f"\nTesting {len(thresholds)} threshold values: ±{min(thresholds)} to ±{max(thresholds)}")
    
    # Run sweep
    results = run_threshold_sweep(thresholds, verbose=True)
    
    if not results:
        print("\nERROR: No results obtained!")
        return
    
    # Print summary
    print_summary_table(results)
    
    # Save results
    save_results_to_file(results)
    print(f"\nResults saved to: threshold_sweep_results.txt")
    
    # Find optimal threshold
    optimal = find_optimal_threshold(results, baseline_accuracy=94.0, max_accuracy_drop=1.0)
    
    if optimal:
        print("\n" + "="*60)
        print("RECOMMENDED THRESHOLD")
        print("="*60)
        print(f"Threshold: ±{optimal.threshold}")
        print(f"Early Exit Rate: {optimal.early_exit_rate:.1f}%")
        print(f"Overall Accuracy: {optimal.overall_accuracy:.1f}%")
        print(f"Average Cycles: {optimal.avg_cycles:.0f}")
        print(f"Cycle Savings: {optimal.avg_cycle_savings:.1f}%")
        print("="*60)
    else:
        print("\n⚠️  WARNING: No threshold meets accuracy requirements!")
        print("Consider relaxing accuracy constraints or investigating early exit classifier.")
    
    # Restore RTL to disabled state
    print("\nRestoring early exit to disabled state...")
    restore_disabled_early_exit()
    
    print("\nDone!")


if __name__ == '__main__':
    # Suppress TensorFlow warnings
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    main()

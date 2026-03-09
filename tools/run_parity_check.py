#!/usr/bin/env python3
"""
Run complete RTL vs TFLite parity check.
1. Dump TFLite intermediates
2. Run RTL simulation
3. Compare intermediates
4. Report first mismatch layer
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def run_command(cmd, cwd=None, check=True):
    """Run shell command and return result."""
    print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    result = subprocess.run(
        cmd if isinstance(cmd, list) else cmd.split(),
        cwd=cwd,
        capture_output=False,
        text=True,
        check=check
    )
    return result


def dump_tflite_intermediates(tflite_path, input_data, output_dir):
    """Run TFLite intermediate dump."""
    cmd = [
        sys.executable,
        "tools/dump_intermediates.py",
        "--tflite", str(tflite_path),
        "--output-dir", str(output_dir)
    ]
    
    if input_data:
        cmd.extend(["--input", str(input_data)])
    
    run_command(cmd)
    return output_dir / "tflite_intermediates.json"


def run_rtl_simulation(input_mem, output_dir):
    """Compile and run RTL simulation."""
    repo_dir = output_dir.parent if output_dir.name == "data" else output_dir
    
    # Compile
    rtl_files = [
        "cnn_tb.v",
        "cnn_top.v",
        "conv1d_layer.v",
        "sliding_window_1d.v",
        "mac_unit.v",
        "maxpool1d.v",
        "global_avg_pool.v",
        "fc_layer.v",
        "argmax.v"
    ]
    
    compile_cmd = ["iverilog", "-g2012", "-o", str(output_dir / "simv")]
    for f in rtl_files:
        compile_cmd.append(str(repo_dir / f))
    
    print("\n=== Compiling RTL ===")
    run_command(compile_cmd, cwd=repo_dir)
    
    # Run simulation
    print("\n=== Running RTL Simulation ===")
    run_cmd = ["vvp", str(output_dir / "simv")]
    run_command(run_cmd, cwd=repo_dir)
    
    rtl_dump = output_dir / "rtl_intermediates.dump"
    if not rtl_dump.exists():
        # Try parent directory
        rtl_dump = repo_dir / "rtl_intermediates.dump"
    
    return rtl_dump


def compare_intermediates(rtl_dump, tflite_json, output_dir):
    """Run comparison script."""
    report_file = output_dir / "parity_report.json"
    
    cmd = [
        sys.executable,
        "tools/compare_rtl_tflite.py",
        "--rtl-dump", str(rtl_dump),
        "--tflite-json", str(tflite_json),
        "--output", str(report_file)
    ]
    
    print("\n=== Comparing RTL vs TFLite ===")
    result = run_command(cmd, check=False)
    
    return report_file, result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run complete RTL vs TFLite parity check")
    parser.add_argument("--tflite", default="data/model_int8.tflite",
                        help="Path to TFLite model")
    parser.add_argument("--input", default=None,
                        help="Input data file (npy or mem)")
    parser.add_argument("--output-dir", default="data",
                        help="Output directory")
    parser.add_argument("--skip-tflite", action="store_true",
                        help="Skip TFLite dump (use existing)")
    parser.add_argument("--skip-rtl", action="store_true",
                        help="Skip RTL simulation (use existing dump)")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("RTL vs TFLite PARITY CHECK")
    print("="*60)
    
    # Step 1: Dump TFLite intermediates
    tflite_json = output_dir / "tflite_intermediates.json"
    if not args.skip_tflite or not tflite_json.exists():
        tflite_json = dump_tflite_intermediates(args.tflite, args.input, output_dir)
    else:
        print(f"Using existing TFLite intermediates: {tflite_json}")
    
    # Step 2: Run RTL simulation
    rtl_dump = output_dir / "rtl_intermediates.dump"
    if not args.skip_rtl or not rtl_dump.exists():
        rtl_dump = run_rtl_simulation(args.input, output_dir)
    else:
        print(f"Using existing RTL dump: {rtl_dump}")
    
    if not rtl_dump.exists():
        # Check repo root
        repo_dir = output_dir.parent if output_dir.name == "data" else output_dir
        rtl_dump = repo_dir / "rtl_intermediates.dump"
    
    if not rtl_dump.exists():
        print("ERROR: RTL dump not found after simulation")
        return 1
    
    # Step 3: Compare
    report_file, all_match = compare_intermediates(rtl_dump, tflite_json, output_dir)
    
    # Summary
    print("\n" + "="*60)
    print("PARITY CHECK COMPLETE")
    print("="*60)
    
    if report_file.exists():
        with open(report_file) as f:
            report = json.load(f)
        
        print(f"Matched: {report['matched']}/{report['total']} layers")
        
        if report['first_mismatch_layer']:
            print(f"\n⚠️  FIRST MISMATCH: {report['first_mismatch_layer']}")
            print("\nNext steps:")
            print(f"1. Examine {report['first_mismatch_layer']} quantization logic")
            print(f"2. Check requant shift values and zero-point handling")
            print(f"3. Compare rounding mode (floor vs nearest)")
        else:
            print("\n✓ ALL LAYERS MATCH - Bit-true parity achieved!")
    
    return 0 if all_match else 1


if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""
Compare RTL simulation outputs against TFLite intermediate tensors.
Identifies which layer first diverges.
"""

import argparse
import json
from pathlib import Path
import numpy as np


def load_rtl_dump(path):
    """Load RTL dump from simulation (text format with layer markers)."""
    layers = {}
    current_layer = None
    current_data = []
    
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Check for layer marker: === LAYER_NAME ===
            if line.startswith("===") and line.endswith("==="):
                if current_layer and current_data:
                    layers[current_layer] = np.array(current_data, dtype=np.int32)
                current_layer = line.strip("=").strip()
                current_data = []
            else:
                # Parse value (hex or decimal)
                try:
                    if line.startswith("0x") or line.startswith("0X"):
                        val = int(line, 16)
                        # Sign-extend if needed
                        if "8bit" in path.name.lower() or "8b" in line:
                            if val > 127:
                                val -= 256
                    elif any(c in line.upper() for c in ["A","B","C","D","E","F"]) and not line.isdigit():
                        val = int(line, 16)
                    else:
                        val = int(line)
                    current_data.append(val)
                except ValueError:
                    continue
        
        # Save last layer
        if current_layer and current_data:
            layers[current_layer] = np.array(current_data, dtype=np.int32)
    
    return layers


def load_tflite_intermediates(path):
    """Load TFLite intermediates from JSON."""
    with open(path) as f:
        data = json.load(f)
    
    layers = {}
    for name, values in data["intermediates"].items():
        layers[name] = np.array(values, dtype=np.int32).flatten()
    
    return layers


def compare_arrays(rtl, tflite, name=""):
    """Compare two arrays and report differences."""
    if rtl.shape != tflite.shape:
        return {
            "match": False,
            "error": f"Shape mismatch: RTL {rtl.shape} vs TFLite {tflite.shape}",
            "max_diff": None,
            "mismatch_count": None,
            "first_mismatch_idx": None
        }
    
    diff = rtl.astype(np.int32) - tflite.astype(np.int32)
    mismatch_mask = diff != 0
    mismatch_count = np.sum(mismatch_mask)
    
    if mismatch_count == 0:
        return {
            "match": True,
            "error": None,
            "max_diff": 0,
            "mismatch_count": 0,
            "first_mismatch_idx": None
        }
    
    max_diff = np.max(np.abs(diff))
    first_mismatch_idx = np.argmax(mismatch_mask)
    
    # Sample some mismatches
    mismatch_indices = np.where(mismatch_mask)[0]
    sample_indices = mismatch_indices[:5]
    samples = []
    for idx in sample_indices:
        samples.append({
            "index": int(idx),
            "rtl": int(rtl[idx]),
            "tflite": int(tflite[idx]),
            "diff": int(diff[idx])
        })
    
    return {
        "match": False,
        "error": None,
        "max_diff": int(max_diff),
        "mismatch_count": int(mismatch_count),
        "total_count": len(rtl),
        "mismatch_rate": float(mismatch_count / len(rtl)),
        "first_mismatch_idx": int(first_mismatch_idx),
        "samples": samples
    }


def main():
    parser = argparse.ArgumentParser(description="Compare RTL vs TFLite intermediates")
    parser.add_argument("--rtl-dump", required=True,
                        help="RTL dump file from simulation")
    parser.add_argument("--tflite-json", required=True,
                        help="TFLite intermediates JSON file")
    parser.add_argument("--output", default=None,
                        help="Output report file (optional)")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading RTL dump: {args.rtl_dump}")
    rtl_layers = load_rtl_dump(Path(args.rtl_dump))
    
    print(f"Loading TFLite intermediates: {args.tflite_json}")
    tflite_layers = load_tflite_intermediates(Path(args.tflite_json))
    
    print(f"\nRTL layers found: {list(rtl_layers.keys())}")
    print(f"TFLite layers found: {list(tflite_layers.keys())}")
    
    # Find common layers and compare
    results = {}
    first_mismatch_layer = None
    
    print("\n" + "="*60)
    print("LAYER-BY-LAYER COMPARISON")
    print("="*60)
    
    for layer_name in rtl_layers.keys():
        if layer_name not in tflite_layers:
            print(f"\n{layer_name}: SKIPPED (not in TFLite)")
            continue
        
        rtl_data = rtl_layers[layer_name]
        tflite_data = tflite_layers[layer_name]
        
        result = compare_arrays(rtl_data, tflite_data, layer_name)
        results[layer_name] = result
        
        if result["match"]:
            status = "✓ MATCH"
        else:
            status = "✗ MISMATCH"
            if first_mismatch_layer is None:
                first_mismatch_layer = layer_name
        
        print(f"\n{layer_name}: {status}")
        print(f"  RTL shape: {rtl_data.shape}, TFLite shape: {tflite_data.shape}")
        
        if not result["match"]:
            print(f"  Max diff: {result['max_diff']}")
            print(f"  Mismatches: {result['mismatch_count']}/{result['total_count']} ({result['mismatch_rate']*100:.1f}%)")
            print(f"  First mismatch at index: {result['first_mismatch_idx']}")
            if "samples" in result:
                print("  Sample mismatches:")
                for s in result["samples"]:
                    print(f"    [{s['index']}] RTL={s['rtl']}, TFLite={s['tflite']}, diff={s['diff']}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    matched = sum(1 for r in results.values() if r["match"])
    total = len(results)
    
    print(f"Matched layers: {matched}/{total}")
    
    if first_mismatch_layer:
        print(f"\n⚠️  FIRST MISMATCH: {first_mismatch_layer}")
        print("   This is the layer where divergence begins.")
        print("   Check quantization/requant logic in this layer.")
    else:
        print("\n✓ ALL LAYERS MATCH - Bit-true parity achieved!")
    
    # Write report
    if args.output:
        report = {
            "matched": matched,
            "total": total,
            "first_mismatch_layer": first_mismatch_layer,
            "results": {
                k: {kk: vv for kk, vv in v.items() if kk != "samples"}
                for k, v in results.items()
            }
        }
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {args.output}")
    
    # Return exit code
    return 0 if first_mismatch_layer is None else 1


if __name__ == "__main__":
    exit(main())

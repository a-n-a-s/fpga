#!/usr/bin/env python3
"""
Compare RTL dump against TFLite intermediates.
"""

import json
import numpy as np
from pathlib import Path


def load_rtl_dump(path):
    """Load RTL dump from simulation."""
    layers = {}
    current_layer = None
    current_data = []
    
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Layer marker: === layer_name ===
            if line.startswith("===") and line.endswith("==="):
                if current_layer and current_data:
                    layers[current_layer] = np.array(current_data, dtype=np.int32)
                current_layer = line.strip("=").strip()
                current_data = []
            else:
                try:
                    val = int(line)
                    current_data.append(val)
                except ValueError:
                    continue
        
        # Save last layer
        if current_layer and current_data:
            layers[current_layer] = np.array(current_data, dtype=np.int32)
    
    return layers


def load_tflite_json(path):
    """Load TFLite intermediates from JSON."""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    
    layers = {}
    for name, values in data.get("intermediates", {}).items():
        layers[name] = np.array(values, dtype=np.int32).flatten()
    
    return layers


def compare(rtl, tflite, name):
    """Compare two arrays."""
    if rtl.shape != tflite.shape:
        return {
            "match": False,
            "reason": f"Shape mismatch: RTL {rtl.shape} vs TFLite {tflite.shape}",
            "max_diff": None,
            "mismatch_rate": None
        }
    
    diff = rtl.astype(np.int64) - tflite.astype(np.int64)
    mismatch = np.sum(diff != 0)
    
    if mismatch == 0:
        return {"match": True, "reason": "Exact match", "max_diff": 0, "mismatch_rate": 0.0}
    
    return {
        "match": False,
        "reason": f"{mismatch}/{len(rtl)} mismatches",
        "max_diff": int(np.max(np.abs(diff))),
        "mismatch_rate": float(mismatch / len(rtl)),
        "first_mismatch": {
            "index": int(np.argmax(diff != 0)),
            "rtl": int(rtl[np.argmax(diff != 0)]),
            "tflite": int(tflite[np.argmax(diff != 0)])
        }
    }


def main():
    rtl_path = Path("rtl_intermediates.dump")
    tflite_path = Path("data/tflite_intermediates.json")
    
    if not rtl_path.exists():
        print(f"ERROR: RTL dump not found: {rtl_path}")
        return 1
    
    if not tflite_path.exists():
        print(f"ERROR: TFLite intermediates not found: {tflite_path}")
        return 1
    
    # Load data
    rtl_layers = load_rtl_dump(rtl_path)
    tflite_layers = load_tflite_json(tflite_path)
    
    print("="*60)
    print("RTL vs TFLite PARITY CHECK")
    print("="*60)
    print(f"\nRTL layers: {list(rtl_layers.keys())}")
    print(f"TFLite layers: {list(tflite_layers.keys())}")
    
    # Compare layer by layer
    results = {}
    first_mismatch = None
    
    print("\n" + "-"*60)
    
    for layer_name in ["conv1", "pool", "conv2", "gap"]:
        if layer_name not in rtl_layers:
            print(f"\n{layer_name}: SKIPPED (not in RTL)")
            continue
        if layer_name not in tflite_layers:
            print(f"\n{layer_name}: SKIPPED (not in TFLite)")
            continue
        
        rtl_data = rtl_layers[layer_name]
        tflite_data = tflite_layers[layer_name]
        
        result = compare(rtl_data, tflite_data, layer_name)
        results[layer_name] = result
        
        if result["match"]:
            status = "[OK] MATCH"
        else:
            status = "[FAIL] MISMATCH"
            if first_mismatch is None:
                first_mismatch = layer_name
        
        print(f"\n{layer_name}: {status}")
        print(f"  RTL shape:   {rtl_data.shape}")
        print(f"  TFLite shape: {tflite_data.shape}")
        print(f"  Reason: {result['reason']}")
        
        if not result["match"] and "first_mismatch" in result:
            fm = result["first_mismatch"]
            print(f"  First mismatch at index {fm['index']}: RTL={fm['rtl']}, TFLite={fm['tflite']}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    matched = sum(1 for r in results.values() if r["match"])
    total = len(results)
    
    print(f"Matched: {matched}/{total} layers")
    
    if first_mismatch:
        print(f"\nWARNING: FIRST MISMATCH: {first_mismatch}")
        print("\nAction items:")
        print(f"1. Check {first_mismatch} quantization parameters")
        print(f"2. Verify zero-point handling (ACT_ZP = -128)")
        print(f"3. Check requantization shift values")
    else:
        print("\nSUCCESS: ALL LAYERS MATCH - Bit-true parity achieved!")
    
    return 0 if first_mismatch is None else 1


if __name__ == "__main__":
    exit(main())

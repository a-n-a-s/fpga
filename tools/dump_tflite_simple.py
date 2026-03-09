#!/usr/bin/env python3
"""
Simple TFLite intermediate dump - runs inference and captures all accessible tensors.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf


def main():
    parser = argparse.ArgumentParser(description="Dump TFLite intermediates")
    parser.add_argument("--tflite", default="data/model_int8.tflite")
    parser.add_argument("--input", default="test_input.npy")
    parser.add_argument("--output", default="data/tflite_intermediates.json")
    
    args = parser.parse_args()
    
    # Load interpreter
    interp = tf.lite.Interpreter(model_path=str(args.tflite))
    interp.allocate_tensors()
    
    # Get I/O details
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()
    all_details = interp.get_tensor_details()
    
    # Load input
    if args.input.endswith(".npy"):
        input_data = np.load(args.input).astype(np.int8)
    else:
        with open(args.input) as f:
            input_data = np.array([int(line.strip()) for line in f if line.strip()], dtype=np.int8)
    
    # Reshape for TFLite
    if input_data.ndim == 1:
        input_data = input_data.reshape(1, 12, 1)
    
    # Set input and invoke
    interp.set_tensor(input_details[0]["index"], input_data)
    interp.invoke()
    
    # Get output
    output_data = interp.get_tensor(output_details[0]["index"])
    
    # Extract all accessible intermediate tensors
    intermediates = {}
    for t in all_details:
        # Skip input and output
        if t["index"] == input_details[0]["index"]:
            continue
        if t["index"] == output_details[0]["index"]:
            continue
        
        # Try to get tensor
        try:
            tensor_data = interp.get_tensor(t["index"])
            name = t["name"].replace("/", "_").replace(":", "_")
            intermediates[name] = {
                "data": tensor_data.flatten().tolist(),
                "shape": list(tensor_data.shape),
                "dtype": str(tensor_data.dtype)
            }
        except (ValueError, TypeError):
            # Skip inaccessible tensors
            continue
    
    # Save results
    result = {
        "input": input_data.flatten().tolist(),
        "output": output_data.flatten().tolist(),
        "intermediates": intermediates,
        "tensor_count": len(intermediates)
    }
    
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"Saved {len(intermediates)} intermediate tensors to {args.output}")
    print(f"\nTensor names:")
    for name, info in intermediates.items():
        print(f"  {name}: {info['shape']} {info['dtype']}")
    
    # Also dump key layers in simple format for comparison
    key_layers = ["conv1", "pool", "conv2", "gap", "fc", "logits"]
    simple_dump = {}
    
    for key in key_layers:
        # Find matching tensor
        for name, info in intermediates.items():
            if key in name.lower():
                simple_dump[key] = info["data"]
                break
    
    # Save simple format
    simple_path = Path(args.output).parent / "tflite_simple.json"
    with open(simple_path, "w") as f:
        json.dump(simple_dump, f, indent=2)
    
    print(f"\nSaved simplified dump to {simple_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Dump TFLite intermediates with correct layer mapping.
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
    
    # Load interpreter with XNNPACK disabled to access all intermediates
    interp = tf.lite.Interpreter(
        model_path=str(args.tflite),
        experimental_preserve_all_tensors=True
    )
    
    # Disable XNNPACK delegate to access all intermediates
    interp._interpreter.SetNumThreads(1)
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
    
    # Reshape for TFLite: (1, 12, 1)
    if input_data.ndim == 1:
        input_data = input_data.reshape(1, 12, 1)
    
    print(f"Input shape: {input_data.shape}")
    print(f"Input values: {input_data.flatten()}")
    
    # Set input and invoke
    interp.set_tensor(input_details[0]["index"], input_data)
    interp.invoke()
    
    # Get output
    output_data = interp.get_tensor(output_details[0]["index"])
    
    # Map tensor names to our layer names
    layer_map = {
        "conv1": ["re_lu_1/Relu", "conv1d_1/BiasAdd"],
        "pool": ["max_pooling1d_1/MaxPool1d/Squeeze"],
        "conv2": ["re_lu_1_2/Relu", "conv1d_1_2/BiasAdd"],
        "gap": ["global_average_pooling1d_1/Mean"],
        "fc": ["StatefulPartitionedCall"]
    }
    
    # Extract intermediates
    intermediates = {}
    
    for layer_name, keywords in layer_map.items():
        for t in all_details:
            name = t["name"]
            # Skip input/output
            if t["index"] == input_details[0]["index"]:
                continue
            if t["index"] == output_details[0]["index"]:
                continue
            
            # Check if any keyword matches
            for kw in keywords:
                if kw in name:
                    try:
                        tensor_data = interp.get_tensor(t["index"])
                        # Flatten and convert to list
                        intermediates[layer_name] = tensor_data.flatten().astype(np.int32).tolist()
                        print(f"Found {layer_name}: {tensor_data.shape} from {name}")
                    except (ValueError, TypeError) as e:
                        print(f"Skip {name}: {e}")
                    break
    
    # Save results
    result = {
        "input": input_data.flatten().astype(np.int32).tolist(),
        "output": output_data.flatten().astype(np.int32).tolist(),
        "intermediates": intermediates
    }
    
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\nSaved {len(intermediates)} intermediate tensors to {args.output}")
    for name, data in intermediates.items():
        print(f"  {name}: {len(data)} values")


if __name__ == "__main__":
    main()

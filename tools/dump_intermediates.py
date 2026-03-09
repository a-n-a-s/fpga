#!/usr/bin/env python3
"""
Dump intermediate tensors from TFLite model for RTL parity checking.
Outputs: conv1, pool, conv2, gap, fc (before argmax)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def run_inference_with_intermediates(tflite_path, input_data):
    """
    Run TFLite inference and capture intermediate layer outputs.
    
    Args:
        tflite_path: Path to .tflite model
        input_data: numpy array of shape (12,) or (1, 12, 1)
    
    Returns:
        dict with intermediate tensor outputs
    """
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    
    # Get tensor details
    details = interp.get_tensor_details()
    
    # Find input tensor
    input_tensor = None
    for t in details:
        if t["index"] == interp.get_input_details()[0]["index"]:
            input_tensor = t
            break
    
    # Find output tensor
    output_tensor = None
    for t in details:
        if t["index"] == interp.get_output_details()[0]["index"]:
            output_tensor = t
            break
    
    # Reshape input if needed
    if input_data.ndim == 1:
        input_data = input_data.reshape(1, 12, 1)
    
    # Set input
    interp.set_tensor(input_tensor["index"], input_data.astype(np.int8))
    
    # Invoke
    interp.invoke()
    
    # Get final output
    final_output = interp.get_tensor(output_tensor["index"])
    
    # Extract intermediate tensors by name pattern matching
    intermediates = {}

    for t in details:
        name = t["name"]
        try:
            tensor_data = interp.get_tensor(t["index"])
        except ValueError:
            # Skip tensors that can't be accessed
            continue

        # Conv1 output (before or after ReLU depending on model structure)
        if "conv1d" in name.lower() and "biasadd" in name.lower():
            if "1" in name and "2" not in name:  # First conv
                intermediates["conv1"] = tensor_data.copy()

        # MaxPool output
        if "max_pool" in name.lower() or "maxpool" in name.lower():
            intermediates["pool"] = tensor_data.copy()

        # Conv2 output
        if "conv1d" in name.lower() and "biasadd" in name.lower():
            if "2" in name or "1_1" in name or "second" in name.lower():
                intermediates["conv2"] = tensor_data.copy()

        # Global Average Pooling output
        if "mean" in name.lower() or "global_avg" in name.lower() or "gap" in name.lower():
            intermediates["gap"] = tensor_data.copy()

        # Dense/FC output (logits, before softmax/argmax)
        if "dense" in name.lower() or "fc" in name.lower() or "logits" in name.lower():
            if tensor_data.shape[-1] == 2:  # Output dimension
                intermediates["fc"] = tensor_data.copy()
    
    # If we couldn't find by name, try by shape
    if "conv1" not in intermediates:
        for t in details:
            shape = list(t["shape"])
            # Conv1: (1, 12, 8) or (1, 8, 12) depending on layout
            if len(shape) == 3 and 8 in shape and t["index"] != input_tensor["index"]:
                if shape[2] == 8 or shape[1] == 8:
                    intermediates["conv1"] = interp.get_tensor(t["index"]).copy()
    
    if "pool" not in intermediates:
        for t in details:
            shape = list(t["shape"])
            # Pool: (1, 6, 8) after pooling 12->6
            if len(shape) == 3 and 6 in shape and 8 in shape:
                intermediates["pool"] = interp.get_tensor(t["index"]).copy()
    
    if "conv2" not in intermediates:
        for t in details:
            shape = list(t["shape"])
            # Conv2: (1, 6, 16)
            if len(shape) == 3 and 16 in shape:
                intermediates["conv2"] = interp.get_tensor(t["index"]).copy()
    
    if "gap" not in intermediates:
        for t in details:
            shape = list(t["shape"])
            # GAP: (1, 16)
            if len(shape) == 2 and 16 in shape:
                intermediates["gap"] = interp.get_tensor(t["index"]).copy()
    
    if "fc" not in intermediates:
        for t in details:
            shape = list(t["shape"])
            # FC output: (1, 2)
            if len(shape) == 2 and shape[-1] == 2 and t["index"] != output_tensor["index"]:
                intermediates["fc"] = interp.get_tensor(t["index"]).copy()
    
    return {
        "input": input_data,
        "output": final_output,
        "intermediates": intermediates,
        "tensor_names": {k: list(v.shape) for k, v in intermediates.items()}
    }


def save_intermediates(results, output_dir):
    """Save intermediates to JSON for comparison."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to JSON-serializable format
    serializable = {
        "input": results["input"].tolist(),
        "output": results["output"].tolist(),
        "intermediates": {},
        "tensor_shapes": results["tensor_names"]
    }
    
    for name, tensor in results["intermediates"].items():
        serializable["intermediates"][name] = tensor.tolist()
    
    output_file = output_dir / "tflite_intermediates.json"
    with open(output_file, "w") as f:
        json.dump(serializable, f, indent=2)
    
    print(f"Saved TFLite intermediates to {output_file}")
    print(f"Found layers: {list(results['intermediates'].keys())}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Dump TFLite intermediate tensors")
    parser.add_argument("--tflite", default="data/model_int8.tflite",
                        help="Path to TFLite model")
    parser.add_argument("--input", default=None,
                        help="Input file (npy or mem format)")
    parser.add_argument("--output-dir", default="data",
                        help="Output directory")
    parser.add_argument("--sample-index", type=int, default=0,
                        help="If using X_test.npy, which sample to use")
    
    args = parser.parse_args()
    
    # Load input data
    if args.input:
        input_path = Path(args.input)
        if input_path.suffix == ".npy":
            input_data = np.load(input_path)
            if input_data.ndim == 2:  # (N, 12)
                input_data = input_data[args.sample_index]
        elif input_path.suffix == ".mem":
            # Load decimal mem file
            with open(input_path) as f:
                values = [int(line.strip()) for line in f if line.strip()]
            input_data = np.array(values, dtype=np.int8)
        else:
            raise ValueError(f"Unknown input format: {input_path.suffix}")
    else:
        # Generate test input
        input_data = np.arange(12, dtype=np.int8)
    
    print(f"Input shape: {input_data.shape}")
    print(f"Input values: {input_data}")
    
    # Run inference
    results = run_inference_with_intermediates(args.tflite, input_data)
    
    # Save results
    save_intermediates(results, args.output_dir)
    
    # Print summary
    print("\n=== Tensor Shapes ===")
    for name, shape in results["tensor_names"].items():
        print(f"  {name}: {shape}")


if __name__ == "__main__":
    main()

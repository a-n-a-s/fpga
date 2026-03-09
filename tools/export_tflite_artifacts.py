import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def to_hex_lines(arr: np.ndarray, bits: int):
    mask = (1 << bits) - 1
    width = bits // 4
    flat = arr.reshape(-1)
    return [f"{(int(v) & mask):0{width}X}" for v in flat]


def write_hex(path: Path, arr: np.ndarray, bits: int):
    path.write_text("\n".join(to_hex_lines(arr, bits)) + "\n", encoding="ascii")


def find_tensor(details, shape, dtype):
    matches = [t for t in details if list(t["shape"]) == list(shape) and t["dtype"] == dtype]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one tensor shape={shape} dtype={dtype}, got {len(matches)}")
    return matches[0]


def main():
    ap = argparse.ArgumentParser(description="Export TFLite tensors/quant params for RTL")
    ap.add_argument("--tflite", default="data/model_int8.tflite")
    ap.add_argument("--outdir", default="data")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    itp = tf.lite.Interpreter(model_path=str(args.tflite))
    itp.allocate_tensors()
    details = itp.get_tensor_details()

    # Model-specific tensor lookup by shape/dtype (not by fragile index).
    t_in = find_tensor(details, [1, 12, 1], np.int8)
    t_out = find_tensor(details, [1, 2], np.int8)
    t_c1_w = find_tensor(details, [8, 1, 3, 1], np.int8)
    t_c1_b = find_tensor(details, [8], np.int32)
    t_c2_w = find_tensor(details, [16, 1, 3, 8], np.int8)
    t_c2_b = find_tensor(details, [16], np.int32)
    t_d_w = find_tensor(details, [2, 16], np.int8)
    t_d_b = find_tensor(details, [2], np.int32)

    tensors = {
        "conv1_weights": itp.get_tensor(t_c1_w["index"]),
        "conv1_bias": itp.get_tensor(t_c1_b["index"]),
        "conv2_weights": itp.get_tensor(t_c2_w["index"]),
        "conv2_bias": itp.get_tensor(t_c2_b["index"]),
        "dense_weights": itp.get_tensor(t_d_w["index"]),
        "dense_bias": itp.get_tensor(t_d_b["index"]),
    }

    # Export hex memories consumed by RTL.
    write_hex(outdir / "conv1_weights_hex.mem", tensors["conv1_weights"], 8)
    write_hex(outdir / "conv1_bias_hex.mem", tensors["conv1_bias"], 32)
    write_hex(outdir / "conv2_weights_hex.mem", tensors["conv2_weights"], 8)
    write_hex(outdir / "conv2_bias_hex.mem", tensors["conv2_bias"], 32)
    write_hex(outdir / "dense_weights_hex.mem", tensors["dense_weights"], 8)
    write_hex(outdir / "dense_bias_hex.mem", tensors["dense_bias"], 32)

    def qinfo(t):
        qp = t.get("quantization_parameters", {})
        return {
            "name": t["name"],
            "shape": [int(x) for x in t["shape"]],
            "dtype": str(t["dtype"]),
            "scale": [float(x) for x in qp.get("scales", [])],
            "zero_point": [int(x) for x in qp.get("zero_points", [])],
        }

    # Pull key activation tensors for quantization traceability.
    def find_by_name(substr):
        for t in details:
            if substr in t["name"]:
                return t
        return None

    activation_keys = [
        "serving_default_keras_tensor",
        "conv1d_1/BiasAdd",
        "max_pooling1d_1/MaxPool1d/Squeeze",
        "conv1d_1_2/BiasAdd",
        "global_average_pooling1d_1/Mean",
        "StatefulPartitionedCall",
    ]

    activations = {}
    for k in activation_keys:
        t = find_by_name(k)
        if t is not None:
            activations[k] = qinfo(t)

    payload = {
        "input": qinfo(t_in),
        "output": qinfo(t_out),
        "weights": {
            "conv1": qinfo(t_c1_w),
            "conv2": qinfo(t_c2_w),
            "dense": qinfo(t_d_w),
        },
        "bias": {
            "conv1": qinfo(t_c1_b),
            "conv2": qinfo(t_c2_b),
            "dense": qinfo(t_d_b),
        },
        "activations": activations,
    }

    (outdir / "quant_params.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Exported hex mem files + quant_params.json")


if __name__ == "__main__":
    main()

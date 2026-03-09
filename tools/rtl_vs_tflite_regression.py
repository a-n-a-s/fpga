import argparse
import random
import re
import subprocess
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix


def run(cmd, cwd):
    p = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    return p.returncode, p.stdout, p.stderr


def parse_rtl_class(sim_out: str):
    m = re.search(r"Predicted Class:\s*(\d+)", sim_out)
    if not m:
        return None
    return int(m.group(1))


def tflite_class(interpreter, window_int8: np.ndarray):
    in_d = interpreter.get_input_details()[0]
    out_d = interpreter.get_output_details()[0]
    x = window_int8.astype(np.int8).reshape(1, 12, 1)
    interpreter.set_tensor(in_d["index"], x)
    interpreter.invoke()
    out_q = interpreter.get_tensor(out_d["index"]).astype(np.int16)
    # output tensor is int8 logits; argmax is class.
    return int(np.argmax(out_q, axis=1)[0])

def quantize_from_raw(window_raw: np.ndarray, input_scale: float, input_zero: int):
    # Match notebook preprocessing: X = X / 400.0, then int8 quantization.
    x_norm = window_raw.astype(np.float32) / 400.0
    q = np.round(x_norm / input_scale + input_zero).astype(np.int32)
    q = np.clip(q, -128, 127).astype(np.int8)
    return q


def quantize_from_normalized(window_norm: np.ndarray, input_scale: float, input_zero: int):
    # X_test.npy is already normalized (divided by 400), just quantize
    q = np.round(window_norm.astype(np.float32) / input_scale + input_zero).astype(np.int32)
    q = np.clip(q, -128, 127).astype(np.int8)
    return q


def main():
    ap = argparse.ArgumentParser(description="RTL vs TFLite class regression")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--tflite", default="data/model_int8.tflite")
    ap.add_argument("--input", default="input_data.mem")
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--raw-mgdl", action="store_true", help="Treat input file as raw glucose values and quantize via X/400.")
    ap.add_argument("--x-test-npy", default=None, help="Optional path to X_test.npy (N,12).")
    ap.add_argument("--y-test-npy", default=None, help="Optional path to y_test.npy (N,).")
    ap.add_argument("--rtl-max", type=int, default=None, help="If set, cap number of RTL windows for runtime.")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    y_arr = None
    if args.x_test_npy:
        x = np.load(repo / args.x_test_npy, allow_pickle=False)
        if x.ndim == 3 and x.shape[-1] == 1:
            x = x[:, :, 0]
        if x.ndim != 2 or x.shape[1] != 12:
            raise RuntimeError(f"Unexpected X shape: {x.shape}, expected (N,12) or (N,12,1)")
        windows_raw = x.astype(np.float32)
        starts = list(range(len(windows_raw)))
        random.Random(args.seed).shuffle(starts)
        starts = starts[: min(args.windows, len(starts))]
        if args.y_test_npy:
            y_arr = np.load(repo / args.y_test_npy, allow_pickle=False)
    else:
        vals = []
        for ln in (repo / args.input).read_text(encoding="utf-8").splitlines():
            s = ln.strip()
            if not s or s.startswith("//"):
                continue
            vals.append(int(s, 16) if re.fullmatch(r"[0-9A-Fa-f]+", s) else int(s, 10))
        vals = np.array(vals, dtype=np.int16)
        if len(vals) < 12:
            raise RuntimeError("Need at least 12 input samples")
        starts = list(range(0, len(vals) - 12 + 1))
        random.Random(args.seed).shuffle(starts)
        starts = starts[: min(args.windows, len(starts))]

    # Compile once.
    files = [
        "cnn_tb.v", "cnn_top.v", "conv1d_layer.v", "sliding_window_1d.v",
        "mac_unit.v", "maxpool1d.v", "global_avg_pool.v", "fc_layer.v", "argmax.v"
    ]
    rc, out, err = run(["iverilog", "-g2012", "-o", "simv"] + files, cwd=repo)
    if rc != 0:
        print(out)
        print(err)
        raise RuntimeError("iverilog compile failed")

    itp = tf.lite.Interpreter(model_path=str(repo / args.tflite))
    itp.allocate_tensors()
    in_d = itp.get_input_details()[0]
    in_scale, in_zero = in_d["quantization"]

    mismatches = 0
    temp_input = repo / "temp_window.mem"
    tfl_preds = []
    y_true = []
    rtl_tests = starts if args.rtl_max is None else starts[: min(args.rtl_max, len(starts))]
    rtl_test_set = set(rtl_tests)

    for idx, st in enumerate(starts):
        if args.x_test_npy:
            w_raw = windows_raw[st]
            # X_test.npy is already normalized, use quantize_from_normalized
            w = quantize_from_normalized(w_raw, in_scale, in_zero)
        else:
            w_raw = vals[st:st + 12]
            w = quantize_from_raw(w_raw, in_scale, in_zero) if args.raw_mgdl else w_raw.astype(np.int8)
        tfl_cls = tflite_class(itp, w)
        tfl_preds.append(tfl_cls)
        if y_arr is not None:
            y_true.append(int(y_arr[st]))

        if st not in rtl_test_set:
            continue

        # Write in hex format ($readmemh expects hex, not decimal)
        temp_input.write_text("\n".join(format(int(x) & 0xFF, '02X') for x in w) + "\n", encoding="ascii")
        rc, sim_out, sim_err = run(["vvp", "simv", f"+INPUT_FILE={temp_input.name}"], cwd=repo)
        if rc != 0:
            print(sim_out)
            print(sim_err)
            raise RuntimeError(f"Simulation failed on window {idx}")

        rtl_cls = parse_rtl_class(sim_out)
        ok = rtl_cls == tfl_cls
        mismatches += 0 if ok else 1
        print(f"window_start={st:3d} rtl={rtl_cls} tflite={tfl_cls} match={ok}")

    if args.y_test_npy:
        acc = accuracy_score(y_true, tfl_preds)
        cm = confusion_matrix(y_true, tfl_preds)
        print(f"tflite_vs_labels: samples={len(y_true)} accuracy={acc:.6f}")
        print(f"confusion_matrix=\n{cm}")

    print(f"rtl_vs_tflite: tested={len(rtl_tests)} mismatches={mismatches}")
    if mismatches and len(rtl_tests) > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate a simple test input file for parity checking (hex format)."""

import numpy as np

# Generate a simple test input (12 samples, INT8 range)
test_input = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110], dtype=np.int8)

# Save as .mem file (HEX format for $readmemh, two's complement)
with open("input_data.mem", "w") as f:
    for val in test_input:
        # Convert to unsigned hex representation
        if val < 0:
            hex_val = val + 256
        else:
            hex_val = val
        f.write(f"{hex_val:02X}\n")

# Also save as .npy for TFLite script (decimal)
np.save("test_input.npy", test_input)

print("Generated test input:")
print(f"  Values: {test_input}")
print(f"  Hex values: {[f'{v:02X}' for v in test_input.astype(np.uint8)]}")
print(f"  Saved: input_data.mem (hex), test_input.npy (decimal)")

#!/usr/bin/env python3
"""Convert .mem weight files to hex format for RTL simulation."""

from pathlib import Path

ROOT = Path(__file__).parent.parent

def convert_to_hex(input_path, output_path, bit_width=32):
    """Convert decimal mem file to hex format.
    
    Args:
        input_path: Path to input .mem file
        output_path: Path to output _hex.mem file
        bit_width: Bit width for two's complement (8 for weights, 32 for biases)
    """
    with open(input_path, 'r') as f:
        values = f.read().split()

    with open(output_path, 'w') as f:
        for val in values:
            try:
                num = int(val)
                # Handle signed values with appropriate bit width
                if num < 0:
                    num = (num + (1 << bit_width)) % (1 << bit_width)
                
                # Format hex based on bit width
                if bit_width == 8:
                    f.write(f"{num:02X}\n")
                elif bit_width == 16:
                    f.write(f"{num:04X}\n")
                else:  # 32-bit
                    f.write(f"{num:08X}\n")
            except ValueError:
                # Already hex or skip
                f.write(f"{val}\n")

def main():
    latest_data = ROOT / '1_1data'  # Updated to 1:1 balanced data
    data_dir = ROOT / 'data'

    # Weights are INT8, biases are INT32
    files_config = [
        ('conv1_weights.mem', 8),
        ('conv1_bias.mem', 32),
        ('conv2_weights.mem', 8),
        ('conv2_bias.mem', 32),
        ('dense_weights.mem', 8),
        ('dense_bias.mem', 32),
    ]

    for filename, bit_width in files_config:
        input_path = latest_data / filename
        if input_path.exists():
            output_name = filename.replace('.mem', '_hex.mem')
            output_path = data_dir / output_name

            # Convert and save to latest_data
            hex_temp = latest_data / output_name
            convert_to_hex(input_path, hex_temp, bit_width)

            # Create symlink in data/
            if output_path.exists() or output_path.is_symlink():
                output_path.unlink()
            output_path.symlink_to(hex_temp)

            print(f"Created: {output_name} (bit_width={bit_width})")

if __name__ == "__main__":
    main()

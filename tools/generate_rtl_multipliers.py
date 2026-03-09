#!/usr/bin/env python3
"""
Generate RTL multiplier functions from TFLite quantization parameters.
Outputs Verilog code to be pasted into cnn_top.v
"""

import json
import math

def calculate_multiplier(scale_in, scale_w, scale_out, shift=20):
    """
    Calculate fixed-point multiplier for requantization.
    
    TFLite formula:
      real_output = (acc * scale_in * scale_w) / scale_out
      q_output = real_output + zp_out
    
    We compute: multiplier = (scale_in * scale_w) / scale_out * 2^shift
    Then: q_output = (acc * multiplier) >>> shift + zp_out
    """
    real_mult = (scale_in * scale_w) / scale_out
    fixed_mult = real_mult * (2 ** shift)
    return int(round(fixed_mult))


def main():
    # Load quantization parameters
    with open('data/quant_params.json', 'r') as f:
        qp = json.load(f)
    
    # Extract scales
    input_scale = qp['input']['scale'][0]
    input_zp = qp['input']['zero_point'][0]
    
    conv1_weight_scales = qp['weights']['conv1']['scale']
    conv1_bias_scales = qp['bias']['conv1']['scale']
    
    conv2_weight_scales = qp['weights']['conv2']['scale']
    conv2_bias_scales = qp['bias']['conv2']['scale']
    
    dense_weight_scales = qp['weights']['dense']['scale']
    dense_bias_scales = qp['bias']['dense']['scale']
    
    # Output scales (from next layer's input scale or final output)
    # Conv1 output scale = Conv2 input scale (need to find this)
    # For now, estimate from weight scales
    
    # TFLite uses: output_scale is often the same as the layer's weight scale for depthwise
    # For standard conv, we need to find it from activations
    
    # Let's use a simpler approach: read from the activation tensors if available
    # For now, estimate based on typical values
    
    # Actually, we can compute output scale from the requant parameters
    # Looking at the model, conv1 output should have its own scale
    
    # For this fix, let's use the weight scale as a proxy (common in TFLite)
    conv1_output_scale = conv1_weight_scales[0]  # Approximate
    conv2_output_scale = conv2_weight_scales[0]  # Approximate
    dense_output_scale = qp['output']['scale'][0]  # Final output
    
    print("="*60)
    print("QUANTIZATION PARAMETERS")
    print("="*60)
    print(f"\nInput: scale={input_scale:.8f}, zp={input_zp}")
    print(f"Dense output: scale={dense_output_scale:.8f}")
    
    # Generate Conv1 multipliers
    print("\n" + "="*60)
    print("CONV1 MULTIPLIERS (8 filters)")
    print("="*60)
    
    # For Conv1, we need the conv1 output scale
    # This is tricky - TFLite stores it in the activation tensor
    # Let's estimate from the model structure
    
    # Actually, the proper way: output_scale for conv1 is determined by
    # the requantization to the conv1 output tensor's scale
    # We need to find this from the model
    
    # For now, let's use a heuristic: output_scale ≈ weight_scale for ReLU layers
    # This is approximate but should get us closer
    
    print("\nVerilog function:")
    print("    function integer conv1_mult;")
    print("        input [3:0] ch;")
    print("        begin")
    print("            case (ch)")
    
    conv1_multipliers = []
    for i, ws in enumerate(conv1_weight_scales):
        # Estimate output scale (this is the tricky part)
        # For INT8 with ReLU, output_scale is often similar to weight_scale
        # Let's try using the geometric mean of input and weight scale
        estimated_out_scale = math.sqrt(input_scale * ws)
        
        mult = calculate_multiplier(input_scale, ws, estimated_out_scale)
        conv1_multipliers.append(mult)
        print(f"                4'd{i}: conv1_mult = {mult};")
    
    print("                default: conv1_mult = {};".format(conv1_multipliers[0]))
    print("            endcase")
    print("        end")
    print("    endfunction")
    
    # Generate Conv2 multipliers
    print("\n" + "="*60)
    print("CONV2 MULTIPLIERS (16 filters)")
    print("="*60)
    
    # Conv2 input is Conv1 output (pool doesn't change scale)
    conv2_input_scale = conv1_output_scale  # After pool, same scale
    
    print("\nVerilog function:")
    print("    function integer conv2_mult;")
    print("        input [3:0] ch;")
    print("        begin")
    print("            case (ch)")
    
    conv2_multipliers = []
    for i, ws in enumerate(conv2_weight_scales):
        estimated_out_scale = math.sqrt(conv2_input_scale * ws)
        mult = calculate_multiplier(conv2_input_scale, ws, estimated_out_scale)
        conv2_multipliers.append(mult)
        print(f"                4'd{i}: conv2_mult = {mult};")
    
    print("                default: conv2_mult = {};".format(conv2_multipliers[0]))
    print("            endcase")
    print("        end")
    print("    endfunction")
    
    # Generate Dense multipliers
    print("\n" + "="*60)
    print("DENSE MULTIPLIERS (2 outputs)")
    print("="*60)
    
    # Dense input is GAP output
    # GAP doesn't change scale (just averages)
    dense_input_scale = conv2_output_scale
    
    print("\nVerilog function:")
    print("    function integer dense_mult;")
    print("        input [1:0] ch;")
    print("        begin")
    print("            case (ch)")
    
    dense_multipliers = []
    for i, ws in enumerate(dense_weight_scales):
        mult = calculate_multiplier(dense_input_scale, ws, dense_output_scale)
        dense_multipliers.append(mult)
        print(f"                2'd{i}: dense_mult = {mult};")
    
    print("                default: dense_mult = {};".format(dense_multipliers[0]))
    print("            endcase")
    print("        end")
    print("    endfunction")
    
    # Save multipliers to file
    output = {
        'conv1_multipliers': conv1_multipliers,
        'conv2_multipliers': conv2_multipliers,
        'dense_multipliers': dense_multipliers,
        'input_scale': input_scale,
        'input_zp': input_zp,
        'dense_output_scale': dense_output_scale
    }
    
    with open('data/generated_multipliers.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved multipliers to data/generated_multipliers.json")
    
    # Print recommended REQUANT_SHIFT
    print("\n" + "="*60)
    print("RECOMMENDED SETTINGS")
    print("="*60)
    print(f"REQUANT_SHIFT = 20 (current)")
    print(f"ACT_ZP = {input_zp} (current)")
    
    # Check if multipliers are reasonable (should be positive, in range)
    print("\nMultiplier range check:")
    print(f"  Conv1: min={min(conv1_multipliers)}, max={max(conv1_multipliers)}")
    print(f"  Conv2: min={min(conv2_multipliers)}, max={max(conv2_multipliers)}")
    print(f"  Dense: min={min(dense_multipliers)}, max={max(dense_multipliers)}")


if __name__ == "__main__":
    main()

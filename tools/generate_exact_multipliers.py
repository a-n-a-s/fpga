#!/usr/bin/env python3
"""
Generate EXACT RTL multiplier functions from known TFLite activation scales.
"""

def calculate_multiplier(scale_in, scale_w, scale_out, shift=20):
    """Calculate fixed-point multiplier."""
    real_mult = (scale_in * scale_w) / scale_out
    fixed_mult = real_mult * (2 ** shift)
    return int(round(fixed_mult))

def main():
    import json
    
    # Exact scales from TFLite model
    input_scale = 0.0039215689
    input_zp = -128
    
    conv1_out_scale = 0.0046529151
    conv2_out_scale = 0.0169535801
    gap_out_scale = 0.0069386028
    output_scale = 0.0562192239
    
    # Load weight scales from quant_params.json
    with open('data/quant_params.json', 'r') as f:
        qp = json.load(f)
    
    conv1_weight_scales = qp['weights']['conv1']['scale']
    conv2_weight_scales = qp['weights']['conv2']['scale']
    dense_weight_scales = qp['weights']['dense']['scale']
    
    print("="*60)
    print("EXACT MULTIPLIERS (from TFLite activation scales)")
    print("="*60)
    
    # Generate Conv1 multipliers
    print("\n" + "="*60)
    print("CONV1 MULTIPLIERS - REPLACE IN cnn_top.v")
    print("="*60)
    print("    function integer conv1_mult;")
    print("        input [3:0] ch;")
    print("        begin")
    print("            case (ch)")
    
    conv1_multipliers = []
    for i, ws in enumerate(conv1_weight_scales):
        mult = calculate_multiplier(input_scale, ws, conv1_out_scale)
        conv1_multipliers.append(mult)
        print(f"                4'd{i}: conv1_mult = {mult};")
    
    print(f"                default: conv1_mult = {conv1_multipliers[0]};")
    print("            endcase")
    print("        end")
    print("    endfunction")
    
    # Generate Conv2 multipliers
    print("\n" + "="*60)
    print("CONV2 MULTIPLIERS - REPLACE IN cnn_top.v")
    print("="*60)
    print("    function integer conv2_mult;")
    print("        input [3:0] ch;")
    print("        begin")
    print("            case (ch)")
    
    conv2_input_scale = conv1_out_scale  # Pool doesn't change scale
    
    conv2_multipliers = []
    for i, ws in enumerate(conv2_weight_scales):
        mult = calculate_multiplier(conv2_input_scale, ws, conv2_out_scale)
        conv2_multipliers.append(mult)
        print(f"                4'd{i}: conv2_mult = {mult};")
    
    print(f"                default: conv2_mult = {conv2_multipliers[0]};")
    print("            endcase")
    print("        end")
    print("    endfunction")
    
    # Generate Dense multipliers
    print("\n" + "="*60)
    print("DENSE MULTIPLIERS - REPLACE IN cnn_top.v")
    print("="*60)
    print("    function integer dense_mult;")
    print("        input [1:0] ch;")
    print("        begin")
    print("            case (ch)")
    
    dense_input_scale = gap_out_scale
    
    dense_multipliers = []
    for i, ws in enumerate(dense_weight_scales):
        mult = calculate_multiplier(dense_input_scale, ws, output_scale)
        dense_multipliers.append(mult)
        print(f"                2'd{i}: dense_mult = {mult};")
    
    print(f"                default: dense_mult = {dense_multipliers[0]};")
    print("            endcase")
    print("        end")
    print("    endfunction")
    
    # Summary
    print("\n" + "="*60)
    print("SCALE SUMMARY")
    print("="*60)
    print(f"Input:        {input_scale:.10f} (zp={input_zp})")
    print(f"Conv1 out:    {conv1_out_scale:.10f}")
    print(f"Conv2 out:    {conv2_out_scale:.10f}")
    print(f"GAP out:      {gap_out_scale:.10f}")
    print(f"Output:       {output_scale:.10f}")
    
    print("\n" + "="*60)
    print("MULTIPLIER STATISTICS")
    print("="*60)
    print(f"Conv1: min={min(conv1_multipliers)}, max={max(conv1_multipliers)}")
    print(f"Conv2: min={min(conv2_multipliers)}, max={max(conv2_multipliers)}")
    print(f"Dense: min={min(dense_multipliers)}, max={max(dense_multipliers)}")

if __name__ == "__main__":
    main()

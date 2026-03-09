import json

with open('data/quant_params.json') as f:
    qp = json.load(f)

# GAP takes conv2 output (scale=0.01695) and produces GAP output (scale=0.00694)
# RTL does: acc = sum(conv2_buf - zp), then acc / 6, then requant with GAP_MULT

conv2_scale = qp['activations']['conv1d_1_2/BiasAdd']['scale'][0]  # 0.01695358008146286
gap_scale = qp['activations']['global_average_pooling1d_1/Mean']['scale'][0]  # 0.006938602775335312

# In RTL:
# acc = sum of (conv2_val - (-128)) for 6 positions
# acc is in "conv2 scale domain" but with zero-point removed
# After dividing by 6: mean in conv2 scale domain
# Then requant: mean * GAP_MULT / 2^20 + gap_zp

# The requantization formula should be:
# gap_val = (mean_in_conv2_domain) * (conv2_scale / gap_scale) * 2^20 / 2^20 + gap_zp
# But we want: gap_val = mean_in_gap_domain + gap_zp
# mean_in_gap_domain = mean_in_conv2_domain * (conv2_scale / gap_scale)

# So GAP_MULT should be: (conv2_scale / gap_scale) * 2^20
REQUANT_SHIFT = 20
gap_mult_correct = (conv2_scale / gap_scale) * (2 ** REQUANT_SHIFT)

print(f'conv2_scale: {conv2_scale}')
print(f'gap_scale: {gap_scale}')
print(f'conv2_scale / gap_scale: {conv2_scale / gap_scale:.6f}')
print(f'Correct GAP_MULT: {gap_mult_correct:.0f}')
print(f'Current GAP_MULT: 2562060')

# Also check: what does acc/6 do?
# acc = sum of 6 values, each in range [-128-(-128), 127-(-128)] = [0, 255]
# So acc is in [0, 1530]
# acc/6 is in [0, 255] - this is the mean in the "de-quantized" domain
# Then we need to requantize to gap_scale

# Actually, the issue is:
# conv2_buf values are INT8 with conv2_scale and zp=-128
# When we do (conv2_buf - (-128)), we get values in [0, 255]
# acc/6 gives us the mean in this [0, 255] domain
# But to convert to gap domain with gap_scale, we need:
# gap_quantized = (mean / 255) * (conv2_scale / gap_scale) * 255 + gap_zp
# Simplifying: gap_quantized = mean * (conv2_scale / gap_scale) + gap_zp

# So GAP_MULT = (conv2_scale / gap_scale) * 2^20
print()
print(f'Ratio conv2/gap: {conv2_scale / gap_scale:.6f}')
print(f'This means GAP output values should be ~2.44x SMALLER than conv2 output')

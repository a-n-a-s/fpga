import json

with open('data/quant_params.json') as f:
    qp = json.load(f)

input_scale = qp['input']['scale'][0]
conv1_out_scale = qp['activations']['conv1d_1/BiasAdd']['scale'][0]

print('=== CONV1 Multipliers ===')
for i, ws in enumerate(qp['weights']['conv1']['scale']):
    acc_scale = input_scale * ws
    mult = (acc_scale / conv1_out_scale) * (2**20)
    print(f'Filter {i}: mult={mult:.0f}')

print()
print('=== CONV2 Multipliers ===')
conv2_out_scale = qp['activations']['conv1d_1_2/BiasAdd']['scale'][0]
conv1_act_scale = qp['activations']['max_pooling1d_1/MaxPool1d/Squeeze']['scale'][0]

for i, ws in enumerate(qp['weights']['conv2']['scale']):
    acc_scale = conv1_act_scale * ws
    mult = (acc_scale / conv2_out_scale) * (2**20)
    print(f'Filter {i}: mult={mult:.0f}')

print()
print('=== DENSE Multipliers ===')
dense_out_scale = qp['output']['scale'][0]
gap_scale = qp['activations']['global_average_pooling1d_1/Mean']['scale'][0]

for i, ws in enumerate(qp['weights']['dense']['scale']):
    acc_scale = gap_scale * ws
    mult = (acc_scale / dense_out_scale) * (2**20)
    print(f'Output {i}: mult={mult:.0f}')

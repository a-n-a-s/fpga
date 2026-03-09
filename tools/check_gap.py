# Check GAP computation
conv2_scale = 0.0169535801
gap_scale = 0.0069386028
ratio = conv2_scale / gap_scale

with open('debug_gap.txt', 'w') as f:
    f.write(f'conv2_scale: {conv2_scale}\n')
    f.write(f'gap_scale: {gap_scale}\n')
    f.write(f'ratio: {ratio}\n')
    f.write(f'ratio / 6: {ratio / 6}\n')
    f.write(f'GAP_MULT / 2^20: {2562060 / (2**20)}\n')

print("Saved to debug_gap.txt")

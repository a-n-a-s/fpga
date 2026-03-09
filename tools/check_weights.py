# Check weight file format
with open('data/conv1_weights_hex.mem') as f:
    weights = []
    for line in f:
        line = line.strip()
        if line:
            val = int(line, 16)
            if val > 127:
                val -= 256
            weights.append(val)

with open('debug_weights.txt', 'w') as f:
    f.write(f'RTL Conv1 weights (first 10): {weights[:10]}\n')
    f.write(f'Filter 0: {weights[0:3]}\n')
    f.write(f'Filter 1: {weights[3:6]}\n')
    f.write(f'Total weights: {len(weights)}\n')

print("Saved to debug_weights.txt")

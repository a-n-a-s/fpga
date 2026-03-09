import numpy as np

# Load RTL conv1 output
rtl = []
with open('rtl_intermediates.dump') as f:
    in_conv1 = False
    for line in f:
        line = line.strip()
        if line == '=== conv1 ===':
            in_conv1 = True
            continue
        if in_conv1:
            if line.startswith('==='):
                break
            rtl.append(int(line))

rtl = np.array(rtl, dtype=np.int32).reshape(12, 8)

with open('rtl_conv1_debug.txt', 'w') as f:
    f.write('RTL Conv1 output (12 positions x 8 filters):\n')
    f.write(str(rtl))
    f.write('\n\nFilter 0 all positions:\n')
    f.write(str(rtl[:, 0]))
    f.write('\n\nUnique values:\n')
    f.write(str(np.unique(rtl)))

print("Saved to rtl_conv1_debug.txt")

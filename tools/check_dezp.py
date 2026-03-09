import numpy as np

input_q = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110], dtype=np.int16)
ACT_ZP = -128

input_dezp = input_q - ACT_ZP

with open('debug_dezp.txt', 'w') as f:
    f.write(f'Input quantized: {input_q}\n')
    f.write(f'ACT_ZP: {ACT_ZP}\n')
    f.write(f'Input de-zero-pointed (q - zp): {input_dezp}\n')
    f.write(f'\nFor input[0]=0: 0 - (-128) = {0 - ACT_ZP}\n')
    f.write(f'For input[1]=10: 10 - (-128) = {10 - ACT_ZP}\n')

print("Saved to debug_dezp.txt")

import numpy as np

x = np.load('data/X_test.npy', allow_pickle=False)
print(f'Shape: {x.shape}')
print(f'Dtype: {x.dtype}')
print(f'Min: {x.min()}, Max: {x.max()}')
print(f'First 3 windows:\n{x[:3]}')

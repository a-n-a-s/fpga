import tensorflow as tf
import numpy as np

interp = tf.lite.Interpreter('latest_data/model_int8.tflite')
interp.allocate_tensors()
inp = np.zeros((1,12,1), dtype=np.int8)
interp.set_tensor(interp.get_input_details()[0]['index'], inp)
interp.invoke()

# Check specific bias tensors
for idx in [6, 8, 10]:
    try:
        data = interp.get_tensor(idx)
        print(f'Index {idx}: shape={data.shape}, dtype={data.dtype}')
        print(f'  Values: {data.flatten()}')
        print(f'  Min/Max: {data.min()} / {data.max()}')
    except Exception as e:
        print(f'Index {idx}: Error - {e}')

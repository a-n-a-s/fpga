import numpy as np
import tensorflow as tf

interp = tf.lite.Interpreter('data/model_int8.tflite', experimental_preserve_all_tensors=True)
interp.allocate_tensors()
inp = np.array([[0,10,20,30,40,50,60,70,80,90,100,110]], dtype=np.int8).reshape(1,12,1)
interp.set_tensor(interp.get_input_details()[0]['index'], inp)
interp.invoke()

with open('debug_output.txt', 'w') as f:
    for t in interp.get_tensor_details():
        if 're_lu_1/Relu' in t['name'] and '1_2' not in t['name']:
            out = interp.get_tensor(t['index'])
            f.write(f'Shape: {out.shape}\n')
            f.write(f'All values:\n{out}\n')
            break
    
    # Also get final output
    out = interp.get_tensor(interp.get_output_details()[0]['index'])
    f.write(f'\nFinal output: {out}\n')

print("Saved to debug_output.txt")

import tensorflow as tf
import numpy as np

# TFLite output
interp = tf.lite.Interpreter('data/model_int8.tflite')
interp.allocate_tensors()
inp = np.array([[0,10,20,30,40,50,60,70,80,90,100,110]], dtype=np.int8).reshape(1,12,1)
interp.set_tensor(interp.get_input_details()[0]['index'], inp)
interp.invoke()
out = interp.get_tensor(interp.get_output_details()[0]['index'])

with open('debug_final.txt', 'w') as f:
    f.write(f'TFLite output: {out[0]}\n')
    f.write(f'TFLite class: {np.argmax(out[0])}\n')
    f.write(f'\n')
    f.write(f'RTL output: logit0=117, logit1=-120\n')
    f.write(f'RTL class: 0 (since 117 > -120)\n')
    f.write(f'\n')
    f.write(f'Match: {np.argmax(out[0]) == 0}\n')

print("Saved to debug_final.txt")

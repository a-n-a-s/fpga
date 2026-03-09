import tensorflow as tf
import numpy as np

interp = tf.lite.Interpreter('data/model_int8.tflite')
interp.allocate_tensors()

X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')

print('TFLite inference on first 10 test samples:')
print('='*50)

correct = 0
for i in range(10):
    inp = X_test[i:i+1].reshape(1, 12, 1).astype(np.int8)
    interp.set_tensor(interp.get_input_details()[0]['index'], inp)
    interp.invoke()
    out = interp.get_tensor(interp.get_output_details()[0]['index'])[0]
    pred = np.argmax(out)
    true = y_test[i]
    match = pred == true
    correct += match
    print(f'  Sample {i}: TFLite={pred}, True={true}, Match={match}, logits={out}')

print(f'\nTFLite accuracy: {correct}/10 ({100*correct/10:.0f}%)')

with open('tflite_test_results.txt', 'w') as f:
    f.write(f'TFLite accuracy: {correct}/10 ({100*correct/10:.0f}%)\n')
    for i in range(10):
        inp = X_test[i:i+1].reshape(1, 12, 1).astype(np.int8)
        interp.set_tensor(interp.get_input_details()[0]['index'], inp)
        interp.invoke()
        out = interp.get_tensor(interp.get_output_details()[0]['index'])[0]
        pred = np.argmax(out)
        f.write(f'Sample {i}: pred={pred}, true={y_test[i]}, logits={out}\n')

print('Saved to tflite_test_results.txt')

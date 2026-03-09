import numpy as np
import tensorflow as tf

# Load test data
X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')

# Find Class 1 samples
class1_idx = np.where(y_test == 1)[0]

print(f'Class 1 samples: {len(class1_idx)}')
print(f'Testing on first 20 Class 1 samples...')

# Load TFLite model
interp = tf.lite.Interpreter('data/model_int8.tflite')
interp.allocate_tensors()

# Test on Class 1 samples
correct = 0
print('\nClass 1 predictions:')
for i, idx in enumerate(class1_idx[:20]):
    inp = X_test[idx:idx+1].reshape(1, 12, 1).astype(np.int8)
    interp.set_tensor(interp.get_input_details()[0]['index'], inp)
    interp.invoke()
    out = interp.get_tensor(interp.get_output_details()[0]['index'])[0]
    pred = np.argmax(out)
    
    if pred == 1:
        correct += 1
        status = 'OK'
    else:
        status = 'MISS'

    print(f'  Sample {idx}: logits={out}, pred={pred} {status}')

print(f'\nClass 1 Recall: {correct}/20 ({100*correct/20:.0f}%)')

# Also test some Class 0
class0_idx = np.where(y_test == 0)[0]
print(f'\nClass 0 samples: {len(class0_idx)}')
print(f'Testing on first 20 Class 0 samples...')

correct_0 = 0
print('\nClass 0 predictions:')
for i, idx in enumerate(class0_idx[:20]):
    inp = X_test[idx:idx+1].reshape(1, 12, 1).astype(np.int8)
    interp.set_tensor(interp.get_input_details()[0]['index'], inp)
    interp.invoke()
    out = interp.get_tensor(interp.get_output_details()[0]['index'])[0]
    pred = np.argmax(out)
    
    if pred == 0:
        correct_0 += 1
        status = 'OK'
    else:
        status = 'MISS'

    print(f'  Sample {idx}: logits={out}, pred={pred} {status}')

print(f'\nClass 0 Accuracy: {correct_0}/20 ({100*correct_0/20:.0f}%)')

# Save results
with open('class_accuracy.txt', 'w') as f:
    f.write(f'Class 1 Recall: {correct}/20 ({100*correct/20:.0f}%)\n')
    f.write(f'Class 0 Accuracy: {correct_0}/20 ({100*correct_0/20:.0f}%)\n')
    f.write(f'\nNote: Model appears to always predict Class 0\n')

print('\nSaved to class_accuracy.txt')

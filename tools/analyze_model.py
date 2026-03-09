import numpy as np
import tensorflow as tf

# Load test data
X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')

with open('model_analysis.txt', 'w') as f:
    f.write('Test set class distribution:\n')
    f.write(f'  Class 0: {np.sum(y_test == 0)} samples\n')
    f.write(f'  Class 1: {np.sum(y_test == 1)} samples\n')
    
    # Load TFLite model
    interp = tf.lite.Interpreter('data/model_int8.tflite')
    interp.allocate_tensors()
    
    # Test on first 100 samples
    correct = 0
    pred_0 = 0
    pred_1 = 0
    
    for i in range(100):
        inp = X_test[i:i+1].reshape(1, 12, 1).astype(np.int8)
        interp.set_tensor(interp.get_input_details()[0]['index'], inp)
        interp.invoke()
        out = interp.get_tensor(interp.get_output_details()[0]['index'])[0]
        pred = np.argmax(out)
        
        if pred == 0:
            pred_0 += 1
        else:
            pred_1 += 1
        
        if pred == y_test[i]:
            correct += 1
    
    f.write(f'\nTFLite predictions (first 100 samples):\n')
    f.write(f'  Predicted Class 0: {pred_0}\n')
    f.write(f'  Predicted Class 1: {pred_1}\n')
    f.write(f'  Accuracy: {correct}/100 ({correct}%)\n')
    
    # Check logits for a few samples
    f.write('\nSample logits:\n')
    for i in [0, 10, 50, 90]:
        inp = X_test[i:i+1].reshape(1, 12, 1).astype(np.int8)
        interp.set_tensor(interp.get_input_details()[0]['index'], inp)
        interp.invoke()
        out = interp.get_tensor(interp.get_output_details()[0]['index'])[0]
        f.write(f'  Sample {i} (true={y_test[i]}): logits={out}, pred={np.argmax(out)}\n')

print('Saved to model_analysis.txt')

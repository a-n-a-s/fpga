import subprocess
import numpy as np

# Load test data
X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')

print('RTL inference on first 5 test samples:')
print('='*50)

correct = 0
agreement = 0

# TFLite results (from previous run)
tflite_results = [0, 0, 0, 0, 0]  # All class 0 for first 10
tflite_logits = [[40, -45]] * 10

for i in range(5):
    # Convert input to hex format
    input_data = X_test[i].astype(np.int16)
    input_hex = [int(v) & 0xFF for v in input_data]
    
    # Write input file
    with open('input_data.mem', 'w') as f:
        for val in input_hex:
            f.write(f'{val:02X}\n')
    
    # Run simulation
    result = subprocess.run(['vvp', 'simv'], capture_output=True, text=True)
    
    # Parse output
    rtl_class = None
    rtl_logits = None
    for line in result.stdout.split('\n'):
        if 'DEBUG_ARGMAX' in line:
            # DEBUG_ARGMAX: logit0=117, logit1=-119, class=0
            try:
                # Extract logit0
                logit0_str = line.split('logit0=')[1].split(',')[0]
                logit0 = int(logit0_str)
                # Extract logit1
                logit1_str = line.split('logit1=')[1].split(',')[0]
                logit1 = int(logit1_str)
                # Extract class
                class_str = line.split('class=')[1].strip()
                rtl_class = int(class_str)
                rtl_logits = [logit0, logit1]
            except (IndexError, ValueError) as e:
                print(f'Parse error: {e}')
                print(f'Line: {line}')
    
    true = int(y_test[i])
    tflite_class = tflite_results[i]
    
    match = rtl_class == true
    agree = rtl_class == tflite_class
    
    if match:
        correct += 1
    if agree:
        agreement += 1
    
    print(f'  Sample {i}: RTL={rtl_class}, TFLite={tflite_class}, True={true}')
    print(f'           RTL logits={rtl_logits}, TFLite logits={tflite_logits[i]}')
    print(f'           Match={match}, Agreement={agree}')

print(f'\nRTL accuracy: {correct}/5 ({100*correct/5:.0f}%)')
print(f'RTL-TFLite agreement: {agreement}/5 ({100*agreement/5:.0f}%)')

# Save results
with open('rtl_test_results.txt', 'w') as f:
    f.write(f'RTL accuracy: {correct}/5 ({100*correct/5:.0f}%)\n')
    f.write(f'RTL-TFLite agreement: {agreement}/5 ({100*agreement/5:.0f}%)\n')

print('\nSaved to rtl_test_results.txt')

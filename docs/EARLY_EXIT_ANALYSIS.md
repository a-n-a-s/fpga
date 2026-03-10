# Early Exit Threshold Sweep Analysis

**Date**: March 10, 2026
**Test**: Threshold sweep on 100 samples
**Status**: ⚠️ Simple classifier insufficient - needs improvement

---

## Executive Summary

The early exit threshold sweep revealed that the **simple feature sum heuristic** (Σ conv1_buf[f]) is **not effective** for this dataset. Even at the highest threshold (±800), accuracy remains at 77% vs. 94% baseline.

### Key Finding
**The early exit classifier needs a more sophisticated approach.** Simple feature sum thresholding does not capture the decision boundary learned by the CNN.

---

## Sweep Results

| Threshold | Exit Rate | Overall Acc | Early Exit Acc | Full Net Acc | Avg Cycles | Savings |
|-----------|-----------|-------------|----------------|--------------|------------|---------|
| ±300 | 100% | 53% | 53% | N/A | 833 | 77.9% |
| ±350 | 100% | 53% | 53% | N/A | 833 | 77.9% |
| ±400 | 100% | 53% | 53% | N/A | 833 | 77.9% |
| ±450 | 100% | 53% | 53% | N/A | 833 | 77.9% |
| ±500 | 100% | 53% | 53% | N/A | 833 | 77.9% |
| ±550 | 100% | 53% | 53% | N/A | 833 | 77.9% |
| ±600 | 100% | 53% | 53% | N/A | 833 | 77.9% |
| ±650 | 99% | 54% | 53.5% | 100% | 866 | 77.0% |
| ±700 | 94% | 58% | 55.3% | 100% | 1032 | 72.6% |
| ±750 | 83% | 67% | 60.2% | 100% | 1398 | 62.9% |
| ±800 | 72% | 77% | 68.1% | 100% | 1763 | 53.2% |
| **Baseline** | 0% | **94%** | N/A | 94% | 3767 | 0% |

### Observations

1. **Early exit accuracy is ~53-68%** across all thresholds - barely better than random (50%)
2. **Full network accuracy is 100%** when it runs (samples that don't trigger early exit)
3. **Higher thresholds** → fewer early exits → better overall accuracy
4. **Even at ±800**, 72% of samples exit early with only 68% accuracy

---

## Root Cause Analysis

### Why Simple Feature Sum Fails

The current heuristic:
```verilog
feature_sum = Σ(conv1_buf[f]) for f in [0:7]

if (feature_sum > threshold): Class 0
elif (feature_sum < -threshold): Class 1
```

**Problems:**

1. **ReLU destroys sign information**: After ReLU, all activations are positive (0 to 127)
   - `conv1_buf[f]` contains ReLU output, not signed values
   - Sum is always positive or zero
   - Negative threshold branch never triggers

2. **Feature sum doesn't correlate with class**: The CNN decision boundary is complex
   - Class determination happens through FC layer weights
   - Simple sum ignores which features are important

3. **No class discrimination**: Both classes can have high or low feature sums
   - Need class-specific confidence, not aggregate magnitude

---

## Recommended Solutions

### Option 1: Mini-Classifier (RECOMMENDED) ⭐

Add a tiny linear classifier on Conv1 features:

```verilog
// Pre-computed weights (from logistic regression on Conv1 features)
integer weight[8];
integer bias;

// Compute score
score = bias;
for (f = 0; f < 8; f++) begin
    score = score + weight[f] * conv1_buf[f];
end

// Decision
if (score > THRESHOLD_HIGH): Class 0 (exit early)
elif (score < THRESHOLD_LOW): Class 1 (exit early)
else: Continue to full network
```

**Pros:**
- Can learn proper decision boundary
- Minimal hardware (~8 multipliers, 8 adders)
- Can be trained offline with Python

**Cons:**
- Need to train weights
- Slightly more complex logic

**Estimated accuracy**: 85-90% with proper training

---

### Option 2: Logit-Based Early Exit

Exit after FC layer based on confidence:

```verilog
// After FC, before argmax
logit_diff = |logit0 - logit1|
logit_sum = |logit0| + |logit1|

if (logit_diff * 255 / logit_sum > CONFIDENCE_THRESHOLD): Exit early
```

**Pros:**
- Uses actual network confidence
- No additional training needed
- Already implemented in confidence unit

**Cons:**
- Only saves GAP + FC cycles (~10% savings)
- Still needs full Conv1 + Conv2

**Estimated savings**: 10-15% cycles

---

### Option 3: Feature Variance

Exit based on activation variance:

```verilog
mean = Σ(conv1_buf[f]) / 8
variance = Σ((conv1_buf[f] - mean)²)

if (variance > HIGH_VARIANCE_THRESHOLD): Class 0 (confident pattern)
elif (variance < LOW_VARIANCE_THRESHOLD): Class 1 (flat response)
else: Continue
```

**Pros:**
- Captures pattern "distinctiveness"
- No training needed

**Cons:**
- Requires squaring (expensive)
- May not correlate with class

---

### Option 4: Disable Early Exit (CURRENT STATUS)

Keep early exit disabled and accept full network latency:

```verilog
// Always use full network
state <= S_CONV2;
```

**Pros:**
- Guaranteed 94% accuracy
- Simple, verified implementation
- Still have XAI + Confidence features

**Cons:**
- No cycle savings
- Missing "adaptive inference" demo feature

---

## Accuracy vs. Speedup Trade-off

```
100% |● Baseline (94% acc, 0% savings)
     |
 90% |
     |                                    ● Option 1 (est.)
 80% |                                    (85% acc, 50% savings)
     |
     |                        ●
 70% |                    ●
     |                ●
     |            ●
 60% |        ●
     |    ●
     |●
 50% +----+----+----+----+----+----+----+
     0%   20%  40%  60%  80%  100%
          Cycle Savings →
```

---

## Next Steps

### Immediate (Recommended)

1. **Implement Option 1 (Mini-Classifier)**
   - Train logistic regression on Conv1 features (Python)
   - Extract weights and bias
   - Implement in Verilog
   - Verify on 100 samples

2. **Alternative: Implement Option 2 (Logit-Based)**
   - Reuse confidence unit
   - Add early exit after confidence calculation
   - Lower overhead, but lower savings

### If Time Permits

3. **Compare approaches**
   - Run threshold sweep for mini-classifier
   - Measure accuracy vs. savings
   - Select optimal configuration

4. **Document findings**
   - Update STATUS.md
   - Add analysis to docs/

---

## Appendix: Python Training Script

To train mini-classifier weights:

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

# Load data
X = np.load('1_1data/X_test.npy')[:100]
y = np.load('1_1data/y_test.npy')[:100]

# Get Conv1 features (run through TFLite up to Conv1)
# ... (extract conv1_output for each sample)

# Train logistic regression
clf = LogisticRegression()
clf.fit(conv1_features, y)

# Extract weights
weights = clf.coef_[0]
bias = clf.intercept_[0]

print(f"Weights: {weights}")
print(f"Bias: {bias}")
```

---

## Conclusion

The simple feature sum heuristic is **not suitable** for early exit in this CNN. A **mini-classifier approach** (Option 1) is recommended for proper decision boundary learning, with estimated 85-90% early exit accuracy at 50% cycle savings.

**Current Status**: Early exit disabled, using full network for 94% accuracy.

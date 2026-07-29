# Quick Start: Label Configuration

## Current Mode: 5-Label Mode ✓

The model is configured to predict 5 pathologies:
- Atelectasis
- Cardiomegaly
- Consolidation
- Pleural Effusion
- Pneumothorax

**No changes needed to run the existing pipeline.**

---

## Switch to All-Labels Mode (13 Pathologies)

To use all 13 CheXpert pathologies (automatically excluding "No Finding"):

### Step 1: Edit config.yaml

Find this line:
```yaml
use_all_labels: false
```

Change it to:
```yaml
use_all_labels: true
```

### Step 2: Understand the Effects

When you set `use_all_labels: true`:

| Aspect | Before (5-label) | After (13-label) |
|--------|------------------|-----------------|
| Model output shape | (batch_size, 5) | (batch_size, 13) |
| Predictions | 5 pathologies | All 13 pathologies |
| No Finding preprocessing | None | Enabled |
| Task weights | Used (5 values) | Skipped (5 ≠ 13), unweighted loss |

### Step 3: The 13 Pathologies

```
1. Enlarged Cardiomediastinum
2. Cardiomegaly
3. Lung Opacity
4. Lung Lesion
5. Edema
6. Consolidation
7. Pneumonia
8. Atelectasis
9. Pneumothorax
10. Pleural Effusion
11. Pleural Other
12. Fracture
13. Support Devices
```

### Step 4: No Finding Preprocessing

When a sample has `No Finding = 1.0` (patient is healthy):
- All pathology labels are zeroed
- **EXCEPT** "Support Devices" (which is equipment, not a finding)

Example:
```
Original labels:  [0.0, 1.0, 0.0, 0.5, 0.0, ...]  (Support Devices at index 12)
After preprocessing: [0.0, 0.0, 0.0, 0.0, 0.0, ..., 0.5]  (all zero except Support Devices)
```

### Step 5: Run Training

No code changes needed. Just run as normal:

```bash
python train.py
# or via runai
runai-bgu submit python -n train-13labels ... -- "python train.py"
```

The model will automatically:
- Predict 13 outputs instead of 5
- Apply No Finding preprocessing to training data
- Fall back to unweighted loss (because task_weights only has 5 values)

---

## Back to 5-Label Mode

Simply revert in config.yaml:
```yaml
use_all_labels: false
```

Everything goes back to the original behavior.

---

## Key Behaviors

### Flexible Label Selection

In 5-label mode, you can choose which 5 (or any number) of labels to use by editing:

```yaml
use_labels:
  - Atelectasis
  - Cardiomegaly
  - Consolidation
  - Pleural Effusion
  - Pneumothorax
```

### Task Weights Mismatch

If you switch to all-labels mode and still have 5 task_weights in config:
- You'll see a warning: `WARNING: task_weights length (5) != num_classes (13)`
- Loss function will automatically fall back to unweighted BCEWithLogitsLoss
- Training will continue normally

To use weighted loss in 13-label mode, calculate and add 13 task_weights to config.yaml.

### Model Output Consistency

The model's output size **always** matches `len(pathologies)`:
- 5-label mode: output shape = (batch_size, 5)
- 13-label mode: output shape = (batch_size, 13)
- Custom mode: output shape = (batch_size, N)

---

## Testing Locally (No Compute Resources)

Check that dimensions work correctly:

```bash
python test_model_dims.py
```

Output:
```
✓ Output shape: (2, 5) (correct)
✓ Output shape: (2, 13) (correct)
All dimension checks PASSED ✓
```

---

## Questions?

Refer to `IMPLEMENTATION_SUMMARY.md` for detailed technical information.

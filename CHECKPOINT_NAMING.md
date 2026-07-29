# Checkpoint Naming Convention

## Overview

Checkpoints are now automatically named with a label-mode prefix to clearly differentiate between models trained with **5 labels** vs. **all 13 labels**.

## Naming Format

### Template Structure

```
densenet-{LABEL_MODE}-{epoch:02d}-{metric_name}={metric_value:.3f}.ckpt
```

### Label Mode Prefix

- **5-label mode** (`use_all_labels: false`):
  ```
  densenet-5labels-epoch=XX-val_auroc_mean=X.XXX.ckpt
  ```

- **All-labels mode** (`use_all_labels: true`):
  ```
  densenet-alllabels-epoch=XX-val_auroc_mean=X.XXX.ckpt
  ```

## Examples

### 5-Label Checkpoints
```
densenet-5labels-epoch=30-val_loss=0.165.ckpt           (Early checkpoint)
densenet-5labels-epoch=39-val_auroc_mean=0.819.ckpt     (Best validation)
densenet-5labels-epoch=42-val_auroc_mean=0.805.ckpt     (Later training)
```

### All-Label Checkpoints (Future)
```
densenet-alllabels-epoch=10-val_auroc_mean=0.755.ckpt
densenet-alllabels-epoch=35-val_auroc_mean=0.801.ckpt
densenet-alllabels-epoch=48-val_auroc_mean=0.818.ckpt
```

## How It Works

### Automatic Assignment

When you run training, the naming prefix is automatically determined:

```python
# In train.py, during ModelCheckpoint initialization:
use_all_labels = config.get('use_all_labels', False)
label_suffix = 'alllabels' if use_all_labels else '5labels'
checkpoint_filename = f"densenet-{label_suffix}-{{epoch:02d}}-{{val_auroc_mean:.3f}}.ckpt"
```

### Implementation Details

1. **Config determines naming**: Set `use_all_labels` in `config.yaml`
2. **Runtime injection**: The label suffix is injected into the filename template at training startup
3. **Automatic in new training runs**: Every new checkpoint saved will have the appropriate prefix
4. **Backward compatible**: Existing checkpoints were renamed to include the 5labels prefix

## Existing Checkpoints

All existing checkpoints have been renamed to clearly indicate they were trained with **5 labels**:

| Old Name | New Name |
|----------|----------|
| `densenet-epoch=30-val_loss=0.165.ckpt` | `densenet-5labels-epoch=30-val_loss=0.165.ckpt` |
| `densenet-epoch=39-val_auroc_mean=0.819.ckpt` | `densenet-5labels-epoch=39-val_auroc_mean=0.819.ckpt` |
| `densenet-epoch=42-val_auroc_mean=0.805.ckpt` | `densenet-5labels-epoch=42-val_auroc_mean=0.805.ckpt` |

**Note:** The `config.yaml` checkpoint path has been updated to reference the renamed file.

## Workflow

### Training with 5 Labels (Default)

```yaml
# config.yaml
use_all_labels: false
```

**Saved checkpoints will be named:**
```
densenet-5labels-epoch=XX-val_auroc_mean=X.XXX.ckpt
```

### Training with All 13 Labels

```yaml
# config.yaml
use_all_labels: true
```

**Saved checkpoints will be named:**
```
densenet-alllabels-epoch=XX-val_auroc_mean=X.XXX.ckpt
```

### Loading Checkpoints

When loading a checkpoint for inference or resuming training, simply use the full checkpoint path:

```python
# In config.yaml (optional, can use latest)
checkpoint_path: /path/to/densenet-5labels-epoch=39-val_auroc_mean=0.819.ckpt

# Or programmatically
from pytorch_lightning import Trainer
trainer = Trainer(...)
trainer.fit(model, ckpt_path="/path/to/checkpoint.ckpt")
```

## Benefits

✓ **Clear differentiation**: Instantly see which label count a checkpoint was trained with
✓ **Prevents model-data mismatch**: Avoid loading 5-label checkpoint with 13-label data
✓ **Organization**: Easy to organize and archive checkpoints by mode
✓ **Reproducibility**: Clear naming aids in tracking experiment lineage
✓ **Automatic**: No manual naming needed - handled at runtime

## Configuration

### Default Behavior

The checkpoint naming is automatically configured based on `use_all_labels` in `config.yaml`. No additional setup needed.

### Customization

To change the base filename template (not recommended), edit `config.yaml`:

```yaml
chkpt_callback:
  filename: "densenet-{epoch:02d}-{val_auroc_mean:.3f}"  # Base template
  # Label suffix is injected automatically before this template
```

The final filename will be:
```
densenet-{LABEL_SUFFIX}-{epoch:02d}-{val_auroc_mean:.3f}.ckpt
```

## Notes

- The label suffix is injected at training startup, not stored in config
- Both 5-label and all-label checkpoints can coexist in the same directory
- Checkpoint loading doesn't care about the name format - PyTorch Lightning uses the file path
- For clarity, organize checkpoints or use separate directories if preferred (optional)

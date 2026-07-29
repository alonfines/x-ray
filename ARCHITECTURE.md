# Architecture: Configurable Label System

## High-Level Flow

```
config.yaml
├── use_all_labels: false/true  ◄─── MAIN SWITCH
├── use_labels: [5 pathologies]
└── task_weights: [5 weights]

                    ↓
        ┌─────────────────────────┐
        │  Label Config Inference │
        ├─────────────────────────┤
        │ If use_all_labels=true: │
        │   pathologies ← 13 labels│
        │   preprocessing ← ON    │
        │                          │
        │ If use_all_labels=false:│
        │   pathologies ← use_labels│
        │   preprocessing ← OFF   │
        └─────────────────────────┘
                    ↓
        ┌─────────────────────────────────┐
        │    Module Initialization        │
        ├─────────────────────────────────┤
        │ CXRDataModule:                  │
        │   self.pathologies = [...]      │
        │   self.no_finding_preprocessing │
        │                                  │
        │ CXRClassifier:                  │
        │   self.num_classes = len(...)   │
        │   self.pathologies = [...]      │
        │                                  │
        │ CXRDenseNet:                    │
        │   self.num_classes = len(...)   │
        └─────────────────────────────────┘
                    ↓
        ┌───────────────────────────────────┐
        │      Data Loading Pipeline        │
        ├───────────────────────────────────┤
        │ CheXpertDataset.__getitem__:      │
        │   1. Load image, extract labels   │
        │   2. If no_finding_preprocessing: │
        │        If No Finding = 1.0:       │
        │          Zero all labels except   │
        │          Support Devices         │
        │   3. Return (image, labels)       │
        └───────────────────────────────────┘
                    ↓
        ┌─────────────────────────────────┐
        │      Model Forward Pass         │
        ├─────────────────────────────────┤
        │ Input: (B, 1, 224, 224)        │
        │ Output: (B, num_classes)       │
        │   where num_classes = 5 or 13  │
        └─────────────────────────────────┘
```

## Module Responsibilities

### config.yaml

**Single source of truth:**
- `use_all_labels`: Boolean switch (false=5-label, true=13-label)
- `use_labels`: List of pathologies (used only when `use_all_labels=false`)
- `task_weights`: Loss weights (validated at runtime)

**Never hardcoded:**
- `num_classes` (derived dynamically)

### data.py

**Constants:**
```python
ALL_CHEXPERT_LABELS = [13 pathologies]  # Authoritative list
NO_FINDING_COL = "No Finding"
SUPPORT_DEVICES_COL = "Support Devices"
```

**CheXpertDataset:**
- Parameter: `no_finding_preprocessing: bool`
- Responsibility: Apply preprocessing logic during sample loading

**CXRDataModule:**
- Reads `use_all_labels` from config
- Sets `self.pathologies` and `self.no_finding_preprocessing`
- Passes both to all dataset instances

### densenet_model.py

**CXRDenseNet:**
- Parameter: `num_classes: int` (optional)
- If not provided, reads from config:
  - If `use_all_labels=true`: 13
  - Else: `len(use_labels)`
- Never hardcodes 5 or 14

### train.py

**CXRClassifier:**
- Reads `use_all_labels` from config
- Sets `self.pathologies` accordingly
- Sets `self.num_classes = len(self.pathologies)`
- Uses `self.pathologies` for per-class AUROC logging

**setup():**
- Validates: `len(task_weights) == self.num_classes`
- On mismatch: prints warning, uses unweighted loss
- On match: uses weighted loss

## Key Design Decisions

### 1. Boolean Flag Over List Switching

**Choice: `use_all_labels: bool`**

Pros:
- Clear intent ("all" vs. "selected")
- Automatic preprocessing toggle
- Less error-prone than listing all 13 manually

Cons:
- Less flexible (can't pick custom subset from all 13)
- But `use_labels` still allows custom 5-label mode

### 2. Dynamic num_classes Over Config

**Choice: Derive from `len(pathologies)` at runtime**

Pros:
- Always consistent
- No sync bugs (e.g., config says 5, model loads 13)
- Works for custom label counts

Cons:
- Slightly slower (one array length call)
- Negligible impact

### 3. No Finding Preprocessing Modular

**Choice: Boolean flag in CheXpertDataset**

Pros:
- Easy to enable/disable per dataset
- Gated by `use_all_labels` (not separate config)
- Clear single responsibility

Cons:
- Adds parameter to dataset constructor
- But minimal complexity

### 4. Graceful task_weights Mismatch

**Choice: Runtime validation + fallback**

Pros:
- Doesn't break pipeline if weights don't match
- Explicit warning to user
- Safer than crashing

Cons:
- Silent fallback could mask config errors
- But clear warning message mitigates this

## Extension Points

### Add Custom Labels

To use a different subset of pathologies:

```yaml
use_all_labels: false
use_labels:
  - Cardiomegaly
  - Pneumothorax
  - Pneumonia
```

### Add All-Labels with Custom task_weights

When switching to 13-label mode, calculate and add 13 weights:

```yaml
use_all_labels: true
task_weights: [1.0, 2.0, ..., 1.5]  # 13 values
```

### Support New Preprocessing Rules

Add parameters to `CheXpertDataset.__init__`:

```python
def __init__(self, ..., no_finding_preprocessing=False, custom_preprocessing=False):
    ...
    if custom_preprocessing:
        # Add new preprocessing logic here
```

## Data Flow Example

### 5-Label Mode (Default)

```
config.yaml
  use_all_labels: false
  use_labels: [Atelectasis, Cardiomegaly, ...]

CXRDataModule:
  pathologies = [Atelectasis, Cardiomegaly, ...]
  no_finding_preprocessing = False

CheXpertDataset:
  Extract labels for 5 pathologies only
  No preprocessing applied

Model:
  Input: (batch, 1, 224, 224)
  Output: (batch, 5)  ◄─── 5 predictions
```

### 13-Label Mode (All Pathologies)

```
config.yaml
  use_all_labels: true
  use_labels: [...]  (ignored)

CXRDataModule:
  pathologies = [13 ALL_CHEXPERT_LABELS]
  no_finding_preprocessing = True

CheXpertDataset:
  Extract labels for 13 pathologies
  Apply preprocessing if No Finding = 1.0:
    Zero all except Support Devices

Model:
  Input: (batch, 1, 224, 224)
  Output: (batch, 13)  ◄─── 13 predictions
```

## Backward Compatibility

**100% backward compatible** with existing 5-label setup:

1. Default `use_all_labels: false` preserves existing behavior
2. Existing configs continue to work unchanged
3. No breaking changes to API
4. Data loading logic identical when in 5-label mode

## Testing Strategy

### Local Tests (No Compute)
- Model dimension smoke test
- Config parsing validation
- Import and constant verification
- task_weights validation logic

### Integration Tests (Via runai)
- Full data loading pipeline
- Model forward pass with real data
- Training loop startup
- Loss function computation

### Edge Cases Covered
- task_weights length mismatch (warning + fallback)
- No Finding preprocessing when disabled (no-op)
- Support Devices not in pathologies (full zero instead)
- Empty pathologies (default fallback to 13)

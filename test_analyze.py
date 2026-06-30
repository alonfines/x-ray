import os
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple


def load_analysis_data(config_path: str = "config.yaml") -> Tuple[pd.DataFrame, Dict, List, Dict]:
    """Load CSV results and threshold data for analysis."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    data_config = config.get('data', {})
    conformal_config = config.get('conformal', {})
    output_config = config.get('output', {})

    working_dir = data_config.get('working_dir', os.getcwd())
    calibration_dir = conformal_config.get('calibration_dir', './conformal_calibration')
    results_csv = output_config.get('results_csv', 'results.csv')

    # Load CSV
    csv_path = os.path.join(working_dir, results_csv)
    df = pd.read_csv(csv_path)

    # Load thresholds and metadata
    thresholds_file = os.path.join(calibration_dir, 'bcops_thresholds.pt')
    saved_data = torch.load(thresholds_file)

    pathologies = config.get('use_labels', [])

    return df, saved_data, pathologies, config


def compute_bcops_scores(probs: np.ndarray, presence_thresholds: np.ndarray, absence_thresholds: np.ndarray) -> np.ndarray:
    """
    Compute BCOPS prediction regions for each sample/class.

    Args:
        probs: (N, num_classes) probability array
        presence_thresholds: (num_classes,) array of presence thresholds
        absence_thresholds: (num_classes,) array of absence thresholds

    Returns:
        prediction_regions: (N, num_classes) array with values {0, 1, 2, 3}
                           0={0}, 1={1}, 2={0,1}, 3=∅
    """
    N, num_classes = probs.shape

    s_pre = 1.0 - probs  # Nonconformity for presence
    s_abs = probs        # Nonconformity for absence

    prediction_regions = np.zeros_like(probs, dtype=int)

    for i in range(N):
        for j in range(num_classes):
            s_pre_val = s_pre[i, j]
            s_abs_val = s_abs[i, j]
            q_pre = presence_thresholds[j]
            q_abs = absence_thresholds[j]

            if s_pre_val > q_pre and s_abs_val <= q_abs:
                prediction_regions[i, j] = 0  # {0}: Negative
            elif s_pre_val <= q_pre and s_abs_val > q_abs:
                prediction_regions[i, j] = 1  # {1}: Positive
            elif s_pre_val <= q_pre and s_abs_val <= q_abs:
                prediction_regions[i, j] = 2  # {0,1}: Both
            else:  # s_pre_val > q_pre and s_abs_val > q_abs
                prediction_regions[i, j] = 3  # ∅: OOD

    return prediction_regions


def compute_metrics(df: pd.DataFrame, pathologies: List[str], saved_data: Dict) -> Dict:
    """Compute all metrics from CSV data."""
    metrics = {}

    # Extract thresholds
    presence_thresholds = saved_data['presence_thresholds'].cpu().numpy()
    absence_thresholds = saved_data['absence_thresholds'].cpu().numpy()

    for pathology in pathologies:
        # Extract columns for this pathology
        true_label_col = f'{pathology}_true_label'
        prob_col = f'{pathology}_prob'

        # Convert to arrays
        true_labels = df[true_label_col].values.astype(int)
        probs = df[prob_col].values.astype(float)

        # Compute BCOPS regions for this pathology
        probs_2d = probs.reshape(-1, 1)
        q_pre = presence_thresholds[pathologies.index(pathology)]
        q_abs = absence_thresholds[pathologies.index(pathology)]

        s_pre = 1.0 - probs
        s_abs = probs

        bcops_regions = np.zeros_like(probs, dtype=int)
        for i in range(len(probs)):
            if s_pre[i] > q_pre and s_abs[i] <= q_abs:
                bcops_regions[i] = 0  # {0}
            elif s_pre[i] <= q_pre and s_abs[i] > q_abs:
                bcops_regions[i] = 1  # {1}
            elif s_pre[i] <= q_pre and s_abs[i] <= q_abs:
                bcops_regions[i] = 2  # {0,1}
            else:
                bcops_regions[i] = 3  # ∅

        # Baseline predictions (prob > 0.5)
        baseline_preds = (probs > 0.5).astype(int)

        # Split by certainty
        definite_mask = (true_labels == 0) | (true_labels == 1)
        uncertain_mask = true_labels == -1

        definite_labels = true_labels[definite_mask]
        definite_regions = bcops_regions[definite_mask]
        definite_baseline = baseline_preds[definite_mask]

        n_definite_pos = (definite_labels == 1).sum()
        n_definite_neg = (definite_labels == 0).sum()
        n_total_definite = len(definite_labels)
        n_uncertain = uncertain_mask.sum()

        # ===== BCOPS METRICS =====
        tp = ((definite_regions == 1) & (definite_labels == 1)).sum()
        tn = ((definite_regions == 0) & (definite_labels == 0)).sum()
        fp = ((definite_regions == 1) & (definite_labels == 0)).sum()
        fn = ((definite_regions == 0) & (definite_labels == 1)).sum()

        bcops_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        bcops_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        bcops_ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        n_abstained = ((definite_regions == 2) | (definite_regions == 3)).sum()
        bcops_abstention_rate = n_abstained / n_total_definite if n_total_definite > 0 else 0.0

        if (tp + tn + fp + fn) > 0:
            bcops_selective_accuracy = (tp + tn) / (tp + tn + fp + fn)
        else:
            bcops_selective_accuracy = 0.0

        # ===== BASELINE METRICS =====
        baseline_tp = ((definite_baseline == 1) & (definite_labels == 1)).sum()
        baseline_tn = ((definite_baseline == 0) & (definite_labels == 0)).sum()
        baseline_fp = ((definite_baseline == 1) & (definite_labels == 0)).sum()
        baseline_fn = ((definite_baseline == 0) & (definite_labels == 1)).sum()

        baseline_sensitivity = baseline_tp / (baseline_tp + baseline_fn) if (baseline_tp + baseline_fn) > 0 else 0.0
        baseline_specificity = baseline_tn / (baseline_tn + baseline_fp) if (baseline_tn + baseline_fp) > 0 else 0.0
        baseline_ppv = baseline_tp / (baseline_tp + baseline_fp) if (baseline_tp + baseline_fp) > 0 else 0.0

        # ===== UNCERTAIN LABELS =====
        uncertain_regions = bcops_regions[uncertain_mask]
        n_uncertain_both = (uncertain_regions == 2).sum()
        uncertain_abstain = n_uncertain_both / n_uncertain if n_uncertain > 0 else 0.0

        metrics[pathology] = {
            'Pos': n_definite_pos,
            'Neg': n_definite_neg,
            'Unc': n_uncertain,
            # BCOPS
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'BCOPS_Sens': bcops_sensitivity,
            'BCOPS_Spec': bcops_specificity,
            'BCOPS_PPV': bcops_ppv,
            'BCOPS_Abstain': bcops_abstention_rate,
            'BCOPS_Selective_Acc': bcops_selective_accuracy,
            # Baseline
            'Baseline_TP': baseline_tp,
            'Baseline_TN': baseline_tn,
            'Baseline_FP': baseline_fp,
            'Baseline_FN': baseline_fn,
            'Baseline_Sens': baseline_sensitivity,
            'Baseline_Spec': baseline_specificity,
            'Baseline_PPV': baseline_ppv,
            # Uncertain
            'Uncertain_Abstain': uncertain_abstain,
            'Unc_Both': n_uncertain_both,
            # Store for plotting
            'true_labels': true_labels,
            'bcops_regions': bcops_regions,
        }

    return metrics


from typing import Dict, List

from typing import Dict, List

from typing import Dict, List

def print_metrics(metrics: Dict, pathologies: List[str]):
    """Print a unified metrics table for Baseline and Conformal performance."""
    print('=' * 125)
    print(f"{'':<35} | {'BASELINE (Definite Only)':^30} | {'CONFORMAL':^50}")
    print(f"{'Pathology':<35} | {'TPR':>9} {'TNR':>9} {'Accuracy':>10} | {'Pos Cov':>10} {'Neg Cov':>10} {'Decisv Acc':>12} {'Uncert Acc':>12}")
    print("-" * 125)

    for pathology in pathologies:
        row = metrics[pathology]
        
        # ---------------------------------------------------------
        # BASELINE CALCULATIONS (Definite labels only)
        # ---------------------------------------------------------
        tp = row['Baseline_TP']
        tn = row['Baseline_TN']
        fp = row['Baseline_FP']
        fn = row['Baseline_FN']
        
        total = tp + tn + fp + fn
        actual_positives = tp + fn
        actual_negatives = tn + fp
        
        tpr = tp / actual_positives if actual_positives > 0 else 0.0
        tnr = tn / actual_negatives if actual_negatives > 0 else 0.0
        acc = (tp + tn) / total if total > 0 else 0.0
        
        # ---------------------------------------------------------
        # CONFORMAL CALCULATIONS
        # ---------------------------------------------------------
        true_labels = row['true_labels']
        bcops_regions = row['bcops_regions']
        
        # Masks for Ground Truth
        pos_mask = true_labels == 1
        neg_mask = true_labels == 0
        def_mask = pos_mask | neg_mask
        unc_mask = true_labels == -1  # Ground Truth is Uncertain
        
        # Coverage Rates (Evaluated on Definite labels)
        pos_covered = ((bcops_regions == 1) | (bcops_regions == 2))[pos_mask].sum()
        neg_covered = ((bcops_regions == 0) | (bcops_regions == 2))[neg_mask].sum()
        
        pos_coverage = pos_covered / pos_mask.sum() if pos_mask.sum() > 0 else 0.0
        neg_coverage = neg_covered / neg_mask.sum() if neg_mask.sum() > 0 else 0.0
        
        # ---------------------------------------------------------
        # ACCURACY CALCULATIONS 
        # ---------------------------------------------------------
        # Decisive Accuracy: When the model makes a strict call (0 or 1) on a definite sample, is it right?
        decisive_mask = ((bcops_regions == 0) | (bcops_regions == 1)) & def_mask
        n_decisive = decisive_mask.sum()
        
        if n_decisive > 0:
            correct_decisive = (bcops_regions[decisive_mask] == true_labels[decisive_mask]).sum()
            decisive_acc = correct_decisive / n_decisive
        else:
            decisive_acc = 0.0
            
        # Uncertain Accuracy: When the GT sample is Uncertain (-1), does the model properly Abstain (2)?
        n_unc = unc_mask.sum()
        
        if n_unc > 0:
            correct_uncertain = (bcops_regions[unc_mask] == 2).sum()
            uncertain_acc = correct_uncertain / n_unc
        else:
            uncertain_acc = 0.0
        
        # ---------------------------------------------------------
        # PRINT ROW
        # ---------------------------------------------------------
        print(f"{pathology:<35} | {tpr:>9.1%} {tnr:>9.1%} {acc:>10.1%} | {pos_coverage:>10.1%} {neg_coverage:>10.1%} {decisive_acc:>12.1%} {uncertain_acc:>12.1%}")

    print('=' * 125)
def compute_and_save_confusion_matrices(metrics: Dict, pathologies: List[str], calibration_dir: str) -> None:
    """
    Compute and visualize confusion matrices for each pathology.
    """
    region_names = {0: 'Absent', 1: 'Present', 2: 'Abstain'}
    label_names = {1: 'Positive', 0: 'Negative', -1: 'Uncertain'}
    label_values = [1, 0, -1]

    print("\n" + "=" * 70)
    print("CONFUSION MATRICES (per pathology)")
    print("=" * 70)

    plt.ioff()  # Disable interactive mode for faster batch processing

    for pathology in pathologies:
        true_labels = metrics[pathology]['true_labels']
        regions = metrics[pathology]['bcops_regions']

        # Create 3x3 confusion matrix
        matrix = np.zeros((3, 3))  # 3 predicted regions × 3 true labels

        for true_idx, true_val in enumerate(label_values):
            true_mask = true_labels == true_val
            n_true = true_mask.sum()

            if n_true > 0:
                regions_for_label = regions[true_mask]
                for pred_idx, region_id in enumerate([1, 0, 2]):  # present, absent, abstain
                    count = (regions_for_label == region_id).sum()
                    percentage = 100.0 * count / n_true
                    matrix[pred_idx, true_idx] = percentage

        # Create and save heatmap
        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            matrix,
            annot=True,
            fmt='.1f',
            cmap='YlOrRd',
            xticklabels=[label_names[val] for val in label_values],
            yticklabels=[region_names[rid] for rid in [1, 0, 2]],
            cbar_kws={'label': 'Percentage (%)'},
            ax=ax,
            vmin=0,
            vmax=100
        )

        ax.set_xlabel('True Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted Region', fontsize=12, fontweight='bold')
        ax.set_title(f'Confusion Matrix: {pathology}', fontsize=14, fontweight='bold')

        # Add percentage sign to annotations
        for text in ax.texts:
            text.set_text(text.get_text() + '%')

        plt.tight_layout()

        # Save figure
        filename = f'{pathology.replace(" ", "_").lower()}_confusion_matrix.png'
        filepath = os.path.join(calibration_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved confusion matrix for {pathology}: {filename}")

    print(f"\n✓ All confusion matrices saved to: {calibration_dir}")


if __name__ == "__main__":
    print('=' * 80)
    print('ANALYZING TEST RESULTS')
    print('=' * 80)

    df, saved_data, pathologies, config = load_analysis_data(config_path="/gpfs0/tamyr/users/alonfi/XRay/config.yaml")

    print(f"\nLoaded {len(df)} samples from results.csv")
    print(f"Analyzing {len(pathologies)} pathologies")

    metrics = compute_metrics(df, pathologies, saved_data)
    print_metrics(metrics, pathologies)

    # Generate confusion matrices
    conformal_config = config.get('conformal', {})
    calibration_dir = conformal_config.get('calibration_dir', './conformal_calibration')
    Path(calibration_dir).mkdir(parents=True, exist_ok=True)
    compute_and_save_confusion_matrices(metrics, pathologies, calibration_dir)

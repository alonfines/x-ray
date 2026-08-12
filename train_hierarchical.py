"""Two-phase hierarchical conditional training (Pham et al. 2020).

Phase 1 (Conditional): Train on subset where all parent labels are positive.
Phase 2 (Fine-tune):   Freeze backbone, retrain classifier on full dataset.

Usage:
    python train_hierarchical.py

Config (config.yaml):
    conditional_training:
      enabled: true
      phase1_epochs: 5
      phase2_epochs: 5
      phase1_lr: 0.001
      phase2_lr: 0.0001
      phase1_checkpoint_dir: "checkpoints/ct_phase1"
"""

import os
import yaml
import torch
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader

from data import get_data_module, parse_labels_config
from data.conditional_dataset import ConditionalSubset
from train import CXRClassifier


def train_hierarchical():
    config_path = "/gpfs0/tamyr/users/alonfi/XRay/config.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    ct_config = config.get('conditional_training', {})
    trainer_config = config.get('trainer', {})
    logger_config = trainer_config.get('logger', {}).get('init_args', {})
    chkpt_config = config.get('chkpt_callback', {})

    phase1_epochs = ct_config.get('phase1_epochs', 5)
    phase2_epochs = ct_config.get('phase2_epochs', 5)
    phase1_lr = ct_config.get('phase1_lr', 0.001)
    phase2_lr = ct_config.get('phase2_lr', 0.0001)
    phase1_checkpoint_dir = ct_config.get('phase1_checkpoint_dir', 'checkpoints/ct_phase1')

    data_config = config.get('data', {})
    working_dir = data_config.get('working_dir', os.getcwd())
    phase1_checkpoint_dir = os.path.join(working_dir, phase1_checkpoint_dir)
    os.makedirs(phase1_checkpoint_dir, exist_ok=True)

    pathologies, use_all = parse_labels_config(config)
    label_suffix = 'alllabels' if use_all else '5labels'
    loss_type = config.get('loss', {}).get('type', 'bce')
    uncertain_strategy = config.get('training', {}).get('uncertain_strategy', 'u_zeros')

    # Shared WandB logger for both phases
    wandb_logger = WandbLogger(
        project=logger_config.get('project', 'cxr_uda'),
        name=logger_config.get('name', 'hierarchical_run') + '_CT',
        tags=logger_config.get('tags', []) + ['conditional_training'],
        config=config
    )

    # =========================================================================
    # PHASE 1: Conditional Training
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: Conditional Training (parent-positive subset)")
    print("=" * 70)

    data_module = get_data_module(config_path=config_path)
    data_module.setup(stage='fit')

    # Save reference to full train dataset for Phase 2
    full_train_dataset = data_module.train_dataset

    # Wrap with conditional filter
    conditional_dataset = ConditionalSubset(full_train_dataset, pathologies)

    # Override train_dataloader to use conditional subset
    # (can't just swap train_dataset because Lightning's setup() would overwrite it)
    _original_train_dataloader = data_module.train_dataloader
    def _conditional_train_dataloader():
        return DataLoader(conditional_dataset, batch_size=data_module.batch_size,
                          num_workers=data_module.num_workers, shuffle=True, pin_memory=True)
    data_module.train_dataloader = _conditional_train_dataloader

    # Create model, optionally loading from pretrained checkpoint
    pretrained_ckpt = ct_config.get('pretrained_checkpoint')
    if pretrained_ckpt:
        if not os.path.isabs(pretrained_ckpt):
            pretrained_ckpt = os.path.join(working_dir, pretrained_ckpt)
        print(f"  Loading pretrained checkpoint: {pretrained_ckpt}")
        model = CXRClassifier.load_from_checkpoint(pretrained_ckpt, strict=False, config_path=config_path)
    else:
        model = CXRClassifier(config_path=config_path)
    model.learning_rate = phase1_lr

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    phase1_checkpoint = ModelCheckpoint(
        dirpath=phase1_checkpoint_dir,
        filename=f'densenet-{label_suffix}-{loss_type}-ct_phase1-{uncertain_strategy}-' + '{epoch:02d}-{val_auroc_mean:.3f}',
        monitor='val_auroc_mean',
        mode='max',
        save_top_k=1
    )

    phase1_trainer = pl.Trainer(
        max_epochs=phase1_epochs,
        precision=trainer_config.get('precision', 'bf16-mixed'),
        accelerator=trainer_config.get('accelerator', 'gpu'),
        devices=trainer_config.get('devices', 1),
        logger=wandb_logger,
        callbacks=[phase1_checkpoint, lr_monitor],
        log_every_n_steps=10,
    )

    phase1_trainer.fit(model, datamodule=data_module)

    best_phase1_path = phase1_checkpoint.best_model_path
    print(f"\n  Phase 1 complete. Best checkpoint: {best_phase1_path}")
    print(f"  Best val_auroc_mean: {phase1_checkpoint.best_model_score:.4f}")

    # =========================================================================
    # PHASE 2: Fine-tune on full data (frozen backbone)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: Fine-tuning on full dataset (frozen backbone)")
    print("=" * 70)

    # Restore full training dataloader
    data_module.train_dataloader = _original_train_dataloader
    print(f"  Restored full training set: {len(full_train_dataset)} samples")

    # Load best Phase 1 checkpoint
    model = CXRClassifier.load_from_checkpoint(best_phase1_path, strict=False, config_path=config_path)

    # Freeze backbone, only train classifier head
    model.model.freeze_backbone()
    frozen_count = sum(1 for p in model.parameters() if not p.requires_grad)
    trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"  Frozen parameters: {frozen_count}, Trainable: {trainable_count}")

    # Override learning rate for Phase 2
    model.learning_rate = phase2_lr
    # Reset criterion so it gets re-created with full dataset pos_weights
    model.criterion = None

    # Early stopping config
    early_stop_config = {}
    for callback in trainer_config.get('callbacks', []):
        if 'EarlyStopping' in callback.get('class_path', ''):
            early_stop_config = callback.get('init_args', {})
            break

    early_stop = EarlyStopping(
        monitor=early_stop_config.get('monitor', 'val_auroc_mean'),
        patience=early_stop_config.get('patience', 30),
        verbose=early_stop_config.get('verbose', True),
        mode=early_stop_config.get('mode', 'max')
    )

    lr_monitor_p2 = LearningRateMonitor(logging_interval='epoch')

    base_filename = chkpt_config.get('filename', 'densenet-{epoch:02d}-{val_loss:.3f}')
    checkpoint_filename = base_filename.replace('densenet-', f'densenet-{label_suffix}-{loss_type}-ct-{uncertain_strategy}-', 1)

    phase2_checkpoint = ModelCheckpoint(
        dirpath=chkpt_config.get('dirpath', './checkpoints'),
        filename=checkpoint_filename,
        monitor=chkpt_config.get('monitor', 'val_auroc_mean'),
        mode=chkpt_config.get('mode', 'max'),
        save_top_k=chkpt_config.get('save_top_k', 1)
    )

    phase2_trainer = pl.Trainer(
        max_epochs=phase2_epochs,
        precision=trainer_config.get('precision', 'bf16-mixed'),
        accelerator=trainer_config.get('accelerator', 'gpu'),
        devices=trainer_config.get('devices', 1),
        logger=wandb_logger,
        callbacks=[early_stop, phase2_checkpoint, lr_monitor_p2],
        log_every_n_steps=10,
    )

    phase2_trainer.fit(model, datamodule=data_module)

    print(f"\n  Phase 2 complete. Best checkpoint: {phase2_checkpoint.best_model_path}")
    print(f"  Best val_auroc_mean: {phase2_checkpoint.best_model_score:.4f}")

    # Final validation
    if not trainer_config.get('fast_dev_run', False):
        phase2_trainer.validate(model, datamodule=data_module)

    print("\n" + "=" * 70)
    print("Hierarchical conditional training complete!")
    print("=" * 70)


if __name__ == '__main__':
    train_hierarchical()

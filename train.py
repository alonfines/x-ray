import os
import yaml
import torch
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import Accuracy, AUROC
from typing import Optional

from data import get_data_module, parse_labels_config
from densenet_model import CXRDenseNet
from losses import get_loss_function, AUCMarginLoss, calculate_pos_weights


class CXRClassifier(pl.LightningModule):
    """PyTorch Lightning wrapper for multi-label chest X-ray classification."""

    def __init__(self, config_path: str = "config.yaml"):
        super().__init__()
        self.save_hyperparameters()

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.config = config

        # Determine which labels to use
        self.pathologies, _ = parse_labels_config(config)
        self.num_classes = len(self.pathologies)

        # Hyperparameters from config
        optimizer_config = config.get('optimizer', {})
        self.learning_rate = optimizer_config.get('lr', 0.001)
        self.min_lr = float(optimizer_config.get('min_lr', 1e-6))
        self.w_decay = optimizer_config.get('w_decay', 0.05)
        self.task_weights = config.get('training', {}).get('task_weights', None)

        # Loss configuration
        loss_config = config.get('loss', {})
        self.loss_type = loss_config.get('type', 'bce')

        self.model = CXRDenseNet(config_path=config_path, num_classes=self.num_classes)
        self.criterion = None

        # Load pretrained backbone weights (supports any loss type)
        loss_config = config.get('loss', {})
        pretrained_ckpt = (loss_config.get('pretrained_checkpoint')
                           or loss_config.get('auc_margin', {}).get('pretrained_checkpoint'))
        if pretrained_ckpt:
                data_config = config.get('data', {})
                working_dir = data_config.get('working_dir', os.getcwd())
                if not os.path.isabs(pretrained_ckpt):
                    pretrained_ckpt = os.path.join(working_dir, pretrained_ckpt)
                print(f"[Two-stage] Loading BCE-pretrained weights from: {pretrained_ckpt}")
                ckpt = torch.load(pretrained_ckpt, map_location='cpu', weights_only=False)
                # Extract model weights from Lightning checkpoint, skipping
                # keys with shape mismatches (e.g. classifier head when num_classes differs)
                current_sd = self.model.state_dict()
                state_dict = {}
                for k, v in ckpt['state_dict'].items():
                    if not k.startswith('model.'):
                        continue
                    key = k.replace('model.', '', 1)
                    if key in current_sd and current_sd[key].shape != v.shape:
                        print(f"[Two-stage] Skipping {key}: shape {v.shape} != {current_sd[key].shape}")
                        continue
                    state_dict[key] = v
                self.model.load_state_dict(state_dict, strict=False)
                print(f"[Two-stage] Loaded {len(state_dict)} weight tensors")

                # Reinitialize classifier head with random weights (paper recommendation)
                reinit = (loss_config.get('reinit_classifier')
                          or loss_config.get('auc_margin', {}).get('reinit_classifier', True))
                if reinit:
                    import torch.nn as nn
                    nn.init.xavier_uniform_(self.model.model.classifier.weight)
                    nn.init.zeros_(self.model.model.classifier.bias)
                    print(f"[Two-stage] Reinitialized classifier head ({self.model.model.classifier})")

        # Metrics
        self._binary_mode = self.num_classes == 1
        if self._binary_mode:
            self.train_acc = Accuracy(task='binary')
            self.val_acc = Accuracy(task='binary')
            self.train_auroc = AUROC(task='binary')
            self.val_auroc = AUROC(task='binary')
        else:
            self.train_acc = Accuracy(task='multilabel', num_labels=self.num_classes)
            self.val_acc = Accuracy(task='multilabel', num_labels=self.num_classes)
            self.train_auroc = AUROC(task='multilabel', num_labels=self.num_classes, average=None)
            self.val_auroc = AUROC(task='multilabel', num_labels=self.num_classes, average=None)

        self.val_preds = []
        self.val_labels = []

        self.example_input_array = torch.randn(1, 1, 224, 224)

    def setup(self, stage: Optional[str] = None):
        """Setup loss function based on config: BCE or AUC Margin Loss."""
        if self.criterion is not None:
            return

        if self.loss_type == 'auc_margin':
            auc_config = self.config.get('loss', {}).get('auc_margin', {})
            margin = auc_config.get('margin', 1.0)
            print(f"Using AUC Margin Loss v2 (margin={margin})")

            self.criterion = get_loss_function(
                loss_type='auc_margin', num_classes=self.num_classes,
                margin=margin
            )
            # Negate alpha's gradient so standard descent becomes ascent (maximization)
            self.criterion.alpha.register_hook(lambda grad: -grad)
        else:
            # BCE loss (default)
            if self.task_weights and len(self.task_weights) == self.num_classes:
                print(f"Using task weights from config: {self.task_weights}")
                pos_weight = torch.tensor(self.task_weights, dtype=torch.float32)
                self.criterion = get_loss_function(loss_type='bce', weighted=True, pos_weight=pos_weight)
            else:
                # Auto-compute pos_weights from training data
                if self.task_weights and len(self.task_weights) != self.num_classes:
                    print(f"WARNING: task_weights length ({len(self.task_weights)}) != num_classes ({self.num_classes})")
                print("Computing pos_weights from training data...")
                train_labels = torch.tensor(
                    self.trainer.datamodule.train_dataset.df[self.trainer.datamodule.pathologies].fillna(0).values,
                    dtype=torch.float32
                )
                pos_weight = calculate_pos_weights(train_labels)
                print(f"Computed pos_weights: {pos_weight}")
                self.criterion = get_loss_function(loss_type='bce', weighted=True, pos_weight=pos_weight)

    def configure_optimizers(self):
        """Configure optimizer and scheduler. Includes loss params for AUC Margin Loss."""
        if isinstance(self.criterion, AUCMarginLoss):
            # AUC loss: model params get weight decay, loss params (a, b, alpha) do not
            param_groups = [
                {'params': self.model.parameters(), 'weight_decay': self.w_decay},
                {'params': [self.criterion.a, self.criterion.b, self.criterion.alpha],
                 'weight_decay': 0.0},
            ]
        else:
            param_groups = [{'params': self.model.parameters(), 'weight_decay': self.w_decay}]

        optimizer = torch.optim.AdamW(param_groups, lr=self.learning_rate)

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.min_lr
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _common_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        return logits, labels, loss

    def training_step(self, batch, batch_idx):
        logits, labels, loss = self._common_step(batch, batch_idx)
        probs = torch.sigmoid(logits)

        # Labels are already perfectly clean from data.py
        if self._binary_mode:
            self.train_acc(probs.squeeze(-1), labels.int().squeeze(-1))
            self.train_auroc(probs.squeeze(-1), labels.int().squeeze(-1))
        else:
            self.train_acc(probs, labels.int())
            self.train_auroc(probs, labels.int())

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        logits, labels, loss = self._common_step(batch, batch_idx)
        probs = torch.sigmoid(logits)

        if self._binary_mode:
            self.val_acc(probs.squeeze(-1), labels.int().squeeze(-1))
            self.val_auroc(probs.squeeze(-1), labels.int().squeeze(-1))
        else:
            self.val_acc(probs, labels.int())
            self.val_auroc(probs, labels.int())

        self.val_preds.append(probs.detach().cpu())
        self.val_labels.append(labels.detach().cpu())

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Project alpha >= 0 after each optimizer step (AUC Margin Loss constraint)."""
        if isinstance(self.criterion, AUCMarginLoss):
            with torch.no_grad():
                self.criterion.alpha.clamp_(min=0)

    def on_train_epoch_end(self):
        auroc_scores = self.train_auroc.compute()
        if auroc_scores.dim() == 0:
            auroc_scores = auroc_scores.unsqueeze(0)
        for pathology, score in zip(self.pathologies, auroc_scores):
            self.log(f'train_auroc_{pathology.replace(" ", "_")}', score, prog_bar=False)
        self.train_auroc.reset()

    def on_validation_epoch_end(self):
        auroc_scores = self.val_auroc.compute()
        if auroc_scores.dim() == 0:
            auroc_scores = auroc_scores.unsqueeze(0)
        for pathology, score in zip(self.pathologies, auroc_scores):
            self.log(f'val_auroc_{pathology.replace(" ", "_")}', score, prog_bar=False)

        mean_auroc = auroc_scores.mean()
        self.log('val_auroc_mean', mean_auroc, prog_bar=True)

        self.val_auroc.reset()
        self.val_preds.clear()
        self.val_labels.clear()


def train(config_path: str = "/gpfs0/tamyr/users/alonfi/XRay/config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    trainer_config = config.get('trainer', {})
    logger_config = trainer_config.get('logger', {}).get('init_args', {})
    chkpt_config = config.get('chkpt_callback', {})

    # Data module & Model
    data_module = get_data_module(config_path=config_path)
    model = CXRClassifier(config_path=config_path)

    # Auto wandb name for single-label runs
    pathologies, _ = parse_labels_config(config)
    if len(pathologies) == 1 and 'name' not in logger_config:
        default_name = f'binary_{pathologies[0].replace(" ", "_").lower()}'
    else:
        default_name = 'baseline_run'

    # Setup Loggers
    wandb_logger = WandbLogger(
        project=logger_config.get('project', 'cxr_uda'),
        name=logger_config.get('name', default_name),
        tags=logger_config.get('tags', ['baseline']),
        config=config
    )

    # Get EarlyStopping config
    early_stop_config = {}
    for callback in trainer_config.get('callbacks', []):
        if 'EarlyStopping' in callback.get('class_path', ''):
            early_stop_config = callback.get('init_args', {})
            break

    # Callbacks
    early_stop = EarlyStopping(
        monitor=early_stop_config.get('monitor', 'val_loss'),
        patience=early_stop_config.get('patience', 30),
        verbose=early_stop_config.get('verbose', True),
        mode=early_stop_config.get('mode', 'min')
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Dynamically set checkpoint filename based on label mode and loss type
    pathologies, use_all = parse_labels_config(config)
    if use_all:
        label_suffix = 'alllabels'
    elif len(pathologies) == 1:
        label_suffix = pathologies[0].replace(' ', '_').lower()
    else:
        label_suffix = f'{len(pathologies)}labels'
    loss_type = config.get('loss', {}).get('type', 'bce')
    hier_suffix = '-hier' if config.get('hierarchy_correction', {}).get('enabled', False) else ''
    base_filename = chkpt_config.get('filename', 'densenet-{epoch:02d}-{val_loss:.3f}')
    checkpoint_filename = base_filename.replace('densenet-', f'densenet-{label_suffix}-{loss_type}{hier_suffix}-', 1)

    checkpoint = ModelCheckpoint(
        dirpath=chkpt_config.get('dirpath', './checkpoints'),
        filename=checkpoint_filename,
        monitor=chkpt_config.get('monitor', 'val_loss'),
        mode=chkpt_config.get('mode', 'min'),
        save_top_k=chkpt_config.get('save_top_k', 1)
    )

    # Initialize Trainer with parsed config
    trainer = pl.Trainer(
        fast_dev_run=trainer_config.get('fast_dev_run', False),
        max_epochs=trainer_config.get('max_epochs', 150),
        precision=trainer_config.get('precision', 'bf16-mixed'),
        accelerator=trainer_config.get('accelerator', 'gpu'),
        devices=trainer_config.get('devices', 1),
        logger=wandb_logger,
        callbacks=[early_stop, checkpoint, lr_monitor],
        log_every_n_steps=10,
    )

    # Train & Validate
    trainer.fit(model, datamodule=data_module)
    
    if not trainer_config.get('fast_dev_run', False):
        trainer.validate(model, datamodule=data_module)

    print("Training complete!")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/gpfs0/tamyr/users/alonfi/XRay/config.yaml')
    args = parser.parse_args()
    train(config_path=args.config)
import os
import yaml
import torch
import pandas as pd
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import Accuracy, AUROC
from typing import Optional

from data import CXRDataModule, ALL_CHEXPERT_LABELS
from densenet_model import CXRDenseNet, get_loss_function, AUCMarginLoss


class CXRClassifier(pl.LightningModule):
    """PyTorch Lightning wrapper for multi-label chest X-ray classification."""

    def __init__(self, config_path: str = "config.yaml"):
        super().__init__()
        self.save_hyperparameters()

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.config = config

        # Parse the nested LightningCLI-style config
        model_args = config.get('model', {}).get('init_args', {})
        densenet_args = model_args.get('model', {}).get('init_args', {})

        # Determine which labels to use based on use_all_labels config
        use_all_labels = config.get('use_all_labels', False)
        if use_all_labels:
            self.pathologies = ALL_CHEXPERT_LABELS
        else:
            self.pathologies = config.get('use_labels', [])

        # Always derive num_classes from len(pathologies), never from config
        self.num_classes = len(self.pathologies)
        
        # Hyperparameters from config
        self.learning_rate = model_args.get('lr', 0.001)
        self.min_lr = float(model_args.get('min_lr', 1e-6))
        self.w_decay = model_args.get('w_decay', 0.05)
        self.task_weights = model_args.get('task_weights', None)

        # Loss configuration
        loss_config = config.get('loss', {})
        self.loss_type = loss_config.get('type', 'bce')

        self.model = CXRDenseNet(config_path=config_path, num_classes=self.num_classes)
        self.criterion = None

        # Metrics
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

            # Compute imratio (prior probability of positive per label) from training CSV
            data_config = self.config.get('data', {})
            working_dir = data_config.get('working_dir', os.getcwd())
            train_csv = os.path.join(working_dir, data_config.get('train_split_csv', 'train_split.csv'))
            df = pd.read_csv(train_csv)
            imratio = []
            for pathology in self.pathologies:
                col = df[pathology].fillna(0).replace(-1.0, 0.0)
                imratio.append(float(col.mean()))
            print(f"Using AUC Margin Loss (margin={margin})")
            print(f"  Imratio per label: {[f'{r:.3f}' for r in imratio]}")

            self.criterion = get_loss_function(
                loss_type='auc_margin', num_classes=self.num_classes,
                margin=margin, imratio=imratio
            )
            # Negate alpha's gradient so standard descent becomes ascent (maximization)
            self.criterion.alpha.register_hook(lambda grad: -grad)
        else:
            # BCE loss (default)
            if self.task_weights:
                if len(self.task_weights) != self.num_classes:
                    print(f"WARNING: task_weights length ({len(self.task_weights)}) != num_classes ({self.num_classes})")
                    print(f"Falling back to unweighted BCEWithLogitsLoss")
                    self.criterion = get_loss_function(loss_type='bce')
                else:
                    print(f"Using task weights from config: {self.task_weights}")
                    pos_weight = torch.tensor(self.task_weights, dtype=torch.float32)
                    self.criterion = get_loss_function(loss_type='bce', weighted=True, pos_weight=pos_weight)
            else:
                self.criterion = get_loss_function(loss_type='bce')

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
        self.train_acc(probs, labels.int())
        self.train_auroc(probs, labels.int())

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        logits, labels, loss = self._common_step(batch, batch_idx)
        probs = torch.sigmoid(logits)

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
        for i, (pathology, score) in enumerate(zip(self.pathologies, auroc_scores)):
            self.log(f'train_auroc_{pathology.replace(" ", "_")}', score, prog_bar=False)
        self.train_auroc.reset()

    def on_validation_epoch_end(self):
        auroc_scores = self.val_auroc.compute()
        for i, (pathology, score) in enumerate(zip(self.pathologies, auroc_scores)):
            self.log(f'val_auroc_{pathology.replace(" ", "_")}', score, prog_bar=False)

        mean_auroc = auroc_scores.mean()
        self.log('val_auroc_mean', mean_auroc, prog_bar=True)

        self.val_auroc.reset()
        self.val_preds.clear()
        self.val_labels.clear()


def train():
    config_path = "/gpfs0/tamyr/users/alonfi/XRay/config.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    trainer_config = config.get('trainer', {})
    logger_config = trainer_config.get('logger', {}).get('init_args', {})
    chkpt_config = config.get('chkpt_callback', {})

    # Data module & Model
    data_module = CXRDataModule(config_path=config_path)
    model = CXRClassifier(config_path=config_path)

    # Setup Loggers
    wandb_logger = WandbLogger(
        project=logger_config.get('project', 'cxr_uda'),
        name=logger_config.get('name', 'baseline_run'),
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
    use_all_labels = config.get('use_all_labels', False)
    label_suffix = 'alllabels' if use_all_labels else '5labels'
    loss_type = config.get('loss', {}).get('type', 'bce')
    base_filename = chkpt_config.get('filename', 'densenet-{epoch:02d}-{val_loss:.3f}')
    checkpoint_filename = base_filename.replace('densenet-', f'densenet-{label_suffix}-{loss_type}-', 1)

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
    train()
import yaml
import torch
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import Accuracy, AUROC
from typing import Optional

from data import CXRDataModule
from densenet_model import CXRDenseNet, get_loss_function


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

        self.pathologies = config.get('use_labels', [])

        self.num_classes = densenet_args.get('num_classes', len(self.pathologies))
        
        # Hyperparameters from config
        self.learning_rate = model_args.get('lr', 0.001)
        self.min_lr = float(model_args.get('min_lr', 1e-6))
        self.w_decay = model_args.get('w_decay', 0.05)
        self.task_weights = model_args.get('task_weights', None)

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
        """Setup loss function using hardcoded config weights or dynamic calculation."""
        if self.criterion is None:
            if self.task_weights:
                # Use the hardcoded weights from the config YAML
                print(f"Using task weights from config: {self.task_weights}")
                pos_weight = torch.tensor(self.task_weights, dtype=torch.float32)
                self.criterion = get_loss_function(weighted=True, pos_weight=pos_weight)
            else:
                # Fallback to unweighted if not provided
                self.criterion = get_loss_function(weighted=False)

    def configure_optimizers(self):
        """Configure AdamW optimizer and CosineAnnealingLR scheduler."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.w_decay
        )
        
        # Use a scheduler to hit the min_lr specified in the config
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

    checkpoint = ModelCheckpoint(
        dirpath=chkpt_config.get('dirpath', './checkpoints'),
        filename=chkpt_config.get('filename', 'densenet-{epoch:02d}-{val_loss:.3f}'),
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
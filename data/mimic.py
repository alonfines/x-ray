"""MIMIC-CXR-JPG dataset and DataModule."""

import os
import yaml
import random
import pandas as pd
import numpy as np
import torch
import lightning as pl
import torchxrayvision as xrv
import torchvision.transforms as tforms
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader

from data import parse_labels_config, NO_FINDING_COL, SUPPORT_DEVICES_COL
from data.hierarchy import apply_hierarchy_correction_to_df

# Allow loading truncated images (common with ongoing downloads)
ImageFile.LOAD_TRUNCATED_IMAGES = True


class MIMICDataset(Dataset):
    """PyTorch Dataset for MIMIC-CXR-JPG chest X-ray images."""

    def __init__(self,
                 csv_path: str,
                 files_dir: str,
                 pathologies: list[str],
                 transform: tforms.Compose | None = None,
                 data_aug: tforms.Compose | None = None,
                 uncertain_strategy: str = "u_zeros",
                 no_finding_preprocessing: bool = False):
        """`uncertain_strategy` maps -1.0 labels: u_zeros, u_ones, u_zeros_lsr, u_ones_lsr, or keep (-1.0 stays)."""
        self.files_dir = files_dir
        self.transform = transform
        self.data_aug = data_aug
        self.pathologies = pathologies
        self.no_finding_preprocessing = no_finding_preprocessing
        self.uncertain_strategy = uncertain_strategy
        self.df = pd.read_csv(csv_path)

    def __len__(self) -> int:
        return len(self.df)

    def _build_image_path(self, row) -> str:
        """Build the full image path from MIMIC-CXR directory structure.

        Structure: files/p{XX}/p{subject_id}/s{study_id}/{dicom_id}.jpg
        """
        subject_id = str(int(row['subject_id']))
        study_id = str(int(row['study_id']))
        dicom_id = row['dicom_id']
        prefix = f"p{subject_id[:2]}"
        return os.path.join(self.files_dir, prefix, f"p{subject_id}", f"s{study_id}", f"{dicom_id}.jpg")

    def _label_value(self, value) -> float:
        """NaN -> 0.0; uncertain (-1.0) -> per `uncertain_strategy`; otherwise the value itself."""
        if pd.isna(value):
            return 0.0
        value = float(value)
        if value != -1.0:
            return value
        strategy = self.uncertain_strategy
        if strategy == "u_ones":
            return 1.0
        if strategy == "u_zeros_lsr":
            return random.uniform(0.0, 0.3)
        if strategy == "u_ones_lsr":
            return random.uniform(0.55, 0.85)
        if strategy == "keep":
            return -1.0
        return 0.0  # u_zeros (and any unknown strategy)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        full_path = self._build_image_path(row)

        img = Image.open(full_path).convert('L')
        img = np.array(img)
        img = np.expand_dims(img, axis=0)

        if self.transform:
            img = self.transform(img)

        if self.data_aug:
            img = self.data_aug(img)

        labels = [self._label_value(row[p]) for p in self.pathologies]
        labels = torch.tensor(labels, dtype=torch.float32)

        # No Finding preprocessing: if No Finding is positive, zero all labels except Support Devices
        if self.no_finding_preprocessing:
            no_finding_val = row.get(NO_FINDING_COL, 0.0)
            if not pd.isna(no_finding_val) and float(no_finding_val) == 1.0:
                if SUPPORT_DEVICES_COL in self.pathologies:
                    support_idx = self.pathologies.index(SUPPORT_DEVICES_COL)
                    support_val = labels[support_idx].item()
                    labels = torch.zeros_like(labels)
                    labels[support_idx] = support_val
                else:
                    labels = torch.zeros_like(labels)

        return img, labels


class MIMICDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for MIMIC-CXR-JPG dataset."""

    def __init__(self, config_path: str = "config.yaml", **kwargs):
        super().__init__()

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.pathologies, use_all_labels = parse_labels_config(config)
        self.no_finding_preprocessing = use_all_labels

        mimic_config = config.get('mimic', {})
        data_config = config.get('data', {})
        self.working_dir = data_config.get('working_dir', os.getcwd())
        self.files_dir = mimic_config.get('files_dir')

        self.train_csv = os.path.join(self.working_dir, mimic_config.get('train_split_csv', 'csv_files/mimic_train_split.csv'))
        self.valid_csv = os.path.join(self.working_dir, mimic_config.get('valid_split_csv', 'csv_files/mimic_valid_split.csv'))
        self.conformal_csv = os.path.join(self.working_dir, mimic_config.get('conformal_split_csv', 'csv_files/mimic_conformal_split.csv'))
        self.test_csv = os.path.join(self.working_dir, mimic_config.get('test_split_csv', 'csv_files/mimic_test_split.csv'))

        training_config = config.get('training', {})
        self.batch_size = kwargs.get('batch_size', training_config.get('batch_size', 32))
        self.num_workers = kwargs.get('num_workers', 4)
        self.uncertain_strategy = training_config.get('uncertain_strategy', 'u_zeros')
        self.hierarchy_correction = config.get('hierarchy_correction', {}).get('enabled', False)

        self.train_dataset = None
        self.val_dataset = None
        self.conformal_dataset = None
        self.test_dataset = None

    def setup(self, stage: str | None = None):
        transform = tforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224)
        ])

        augment = tforms.Compose([
            xrv.datasets.ToPILImage(),
            tforms.RandomHorizontalFlip(p=0.5),
            tforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.95, 1.05)),
            tforms.ToTensor()
        ])

        def make_dataset(csv_path, data_aug=None, **kwargs):
            return MIMICDataset(
                csv_path=csv_path, files_dir=self.files_dir, pathologies=self.pathologies,
                transform=transform, data_aug=data_aug,
                no_finding_preprocessing=self.no_finding_preprocessing, **kwargs)

        if stage in [None, 'fit']:
            self.train_dataset = make_dataset(self.train_csv, augment, uncertain_strategy=self.uncertain_strategy)
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = make_dataset(self.valid_csv)
        if stage in [None, 'fit', 'validate', 'test']:
            self.conformal_dataset = make_dataset(self.conformal_csv)
        if stage in [None, 'test']:
            self.test_dataset = make_dataset(self.test_csv, uncertain_strategy="keep")

        if self.hierarchy_correction:
            for ds in [self.train_dataset, self.val_dataset, self.conformal_dataset, self.test_dataset]:
                if ds is not None:
                    apply_hierarchy_correction_to_df(ds.df)

    def _dataloader(self, dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=shuffle, pin_memory=True)

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset)

    def conformal_dataloader(self) -> DataLoader:
        return self._dataloader(self.conformal_dataset)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self.test_dataset)

import os
import yaml
import pandas as pd
import numpy as np
import torch
import lightning as pl
import torchxrayvision as xrv
import torchvision.transforms as tforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List


class CheXpertDataset(Dataset):
    """Custom PyTorch Dataset for CheXpert chest X-ray images."""

    def __init__(self,
                 csv_path: str,
                 img_dir: str,
                 pathologies: Optional[List[str]] = None,
                 transform: Optional[tforms.Compose] = None,
                 data_aug: Optional[tforms.Compose] = None):
        
        self.csv_path = csv_path
        self.img_dir = img_dir
        self.transform = transform
        self.data_aug = data_aug

        self.df = pd.read_csv(csv_path)

        if pathologies is None:
            self.pathologies = [
                'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
                'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
            ]
        else:
            self.pathologies = pathologies

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = row['Path']
        
        if not os.path.isabs(img_path):
            img_path_clean = img_path.replace('CheXpert-v1.0/train/', '').replace('CheXpert-v1.0-small/valid/', '')
            full_path = os.path.join(self.img_dir, img_path_clean)

            if not os.path.exists(full_path):
                parent_dir = os.path.dirname(self.img_dir)
                alt_dir = os.path.join(parent_dir, 'train' if 'valid' in self.img_dir else 'valid')
                full_path_alt = os.path.join(alt_dir, img_path_clean)
                if os.path.exists(full_path_alt):
                    full_path = full_path_alt
        else:
            full_path = img_path

        img = Image.open(full_path).convert('L') 
        img = np.array(img)
        img = np.expand_dims(img, axis=0) 

        if self.transform:
            img = self.transform(img)

        if self.data_aug:
            img = self.data_aug(img)

        labels = []
        for pathology in self.pathologies:
            value = row[pathology]
            
            # ==========================================
            # FIX: Clean BOTH NaN and -1.0 right here!
            # ==========================================
            if pd.isna(value) or float(value) == -1.0:
                labels.append(0.0) 
            else:
                labels.append(float(value))

        labels = torch.tensor(labels, dtype=torch.float32)

        return img, labels


class CXRDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for CheXpert dataset."""

    def __init__(self, config_path: str = "config.yaml", **kwargs):
        super().__init__()

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.pathologies = config.get('use_labels', None)
        data_config = config.get('data', {})
        self.working_dir = data_config.get('working_dir', os.getcwd())
        self.chexpert_dir = data_config.get('chexpert_dir')
        
        # Define all four CSV paths
        self.train_csv = os.path.join(self.working_dir, data_config.get('train_split_csv', 'train_split.csv'))
        self.valid_csv = os.path.join(self.working_dir, data_config.get('valid_split_csv', 'valid_split.csv'))
        self.conformal_csv = os.path.join(self.working_dir, data_config.get('conformal_split_csv', 'conformal_split.csv'))
        self.test_csv = os.path.join(self.working_dir, data_config.get('test_split_csv', 'test_split.csv'))
        
        self.train_img_dir = os.path.join(self.chexpert_dir, data_config.get('train_images_dir', 'train'))
        self.valid_img_dir = os.path.join(self.chexpert_dir, data_config.get('valid_images_dir', 'valid'))

        training_config = config.get('training', {})
        self.batch_size = kwargs.get('batch_size', training_config.get('batch_size', 32))
        self.num_workers = kwargs.get('num_workers', 4)

        self.train_dataset = None
        self.val_dataset = None
        self.conformal_dataset = None
        self.test_dataset = None # Added test dataset placeholder

    def setup(self, stage: Optional[str] = None):
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

        if stage in [None, 'fit']:
            self.train_dataset = CheXpertDataset(
                csv_path=self.train_csv,
                img_dir=self.train_img_dir,
                pathologies=self.pathologies,
                transform=transform,
                data_aug=augment
            )

        if stage in [None, 'fit', 'validate']:
            self.val_dataset = CheXpertDataset(
                csv_path=self.valid_csv,
                img_dir=self.train_img_dir, # Valid was split from train.csv, so it uses train_img_dir!
                pathologies=self.pathologies,
                transform=transform,
                data_aug=None
            )

        if stage in [None, 'fit', 'validate', 'test']:
            self.conformal_dataset = CheXpertDataset(
                csv_path=self.conformal_csv,
                img_dir=self.train_img_dir, # Conformal was split from train.csv, so it uses train_img_dir!
                pathologies=self.pathologies,
                transform=transform,
                data_aug=None
            )

        # Added setup for the purely isolated Test set
        if stage in [None, 'test']:
            self.test_dataset = CheXpertDataset(
                csv_path=self.test_csv,
                img_dir=self.valid_img_dir, # Test came from the raw valid.csv, so it uses valid_img_dir!
                pathologies=self.pathologies,
                transform=transform,
                data_aug=None
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True)

    def conformal_dataloader(self) -> DataLoader:
        return DataLoader(self.conformal_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True)

    # Added the missing test_dataloader
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True)
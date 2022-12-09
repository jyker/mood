import torch
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Tuple, Any

torch.multiprocessing.set_sharing_strategy('file_system')


def plog(data):
    return json.dumps(data, indent=2)


def read_bytes(file_path: str, first_n_byte: int):
    with open(file_path, 'rb') as f:
        # index 0 will be special padding index
        data = [i + 1 for i in f.read()[:first_n_byte]]
        data = data + [0] * (first_n_byte - len(data))
    return data


class MalConvDataset(Dataset):
    '''MalConv Feature Styple Dataset'''

    def __init__(self,
                 data_dir: str,
                 csv_file: str,
                 label_field: str = 'family',
                 train_val_test: str = 'train',
                 first_n_byte: int = 2_000_000):
        self.data_dir = data_dir
        self.csv_file = csv_file
        self.label_field = label_field
        self.train_val_test = train_val_test
        self.first_n_byte = first_n_byte

        self.prepare_data()

    def prepare_data(self):
        expset = self.train_val_test
        log.info(f'Prepare {expset} data')
        df = pd.read_csv(self.csv_file)
        df = df[df['train_val_test'] == expset]
        log.info(f'{expset} length = {len(df)}')
        label_list = df[self.label_field]
        log.info(f'{expset} counter = {plog(Counter(label_list))}')
        # class_to_idx
        self.class_to_idx = {
            name: idx
            for idx, name in enumerate(sorted(pd.unique(label_list)))
        }
        log.info(f'{expset} class_to_idx = {plog(self.class_to_idx)}')
        # targets
        self.targets = [self.class_to_idx[i] for i in label_list]
        # file_path
        self.file_path = [Path(self.data_dir) / i for i in df['file_path']]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        # data = read_bytes(self.file_list[index], self.first_n_byte)
        data = np.load(str(self.file_path[index]) + '.npy').astype(np.int32)
        target = self.targets[index]
        return data, target


class MalConvDataModule(LightningDataModule):
    """LightningDataModule for MalConv Feature dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: str,
        csv_file: str,
        label_field: str = 'family',
        first_n_byte: int = 2_000_000,
        batch_size: int = 32,
        num_workers: int = 2,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = MalConvDataset(self.hparams.data_dir,
                                             self.hparams.csv_file,
                                             self.hparams.label_field, 'train',
                                             self.hparams.first_n_byte)
            self.data_val = MalConvDataset(self.hparams.data_dir,
                                           self.hparams.csv_file,
                                           self.hparams.label_field, 'val',
                                           self.hparams.first_n_byte)
            self.data_test = MalConvDataset(self.hparams.data_dir,
                                            self.hparams.csv_file,
                                            self.hparams.label_field, 'test',
                                            self.hparams.first_n_byte)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
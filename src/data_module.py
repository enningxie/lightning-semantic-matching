# Created by xieenning at 2020/10/19
from argparse import ArgumentParser, Namespace
from typing import Optional, Union, List
from pytorch_lightning import LightningDataModule
from transformers import BertTokenizer
from transformers import ElectraTokenizer
from transformers.utils import logging
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.data_processor import SemanticMatchingProcessor, convert_examples_to_features

logger = logging.get_logger(__name__)


class SemanticMatchingDataModule(LightningDataModule):
    def __init__(self, hparams: Namespace):
        super().__init__()
        self.data_path = hparams.data_path
        self.model_name_or_path = hparams.model_name_or_path
        self.max_length = hparams.max_length
        self.train_batch_size = hparams.train_batch_size
        self.val_batch_size = hparams.val_batch_size
        self.loader_workers = hparams.loader_workers
        self.tokenizer = ElectraTokenizer.from_pretrained(hparams.model_name_or_path)
        self.processor = SemanticMatchingProcessor()

        self.train_features = None
        self.val_features = None

        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self, *args, **kwargs):
        train_examples = self.processor.get_train_examples(self.data_path)
        self.train_features = convert_examples_to_features(train_examples,
                                                           self.tokenizer,
                                                           label_list=self.processor.get_labels(),
                                                           max_length=self.max_length)

        val_examples = self.processor.get_dev_examples(self.data_path)
        self.val_features = convert_examples_to_features(val_examples,
                                                         self.tokenizer,
                                                         label_list=self.processor.get_labels(),
                                                         max_length=self.max_length)
        logger.info("`prepare_data` finished!")

    @staticmethod
    def generate_dataset(features):
        return TensorDataset(
            torch.tensor([f.input_ids for f in features], dtype=torch.long),
            torch.tensor([f.attention_mask for f in features], dtype=torch.long),
            torch.tensor([f.token_type_ids for f in features], dtype=torch.long),
            torch.tensor([f.label for f in features], dtype=torch.long)
        )

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = self.generate_dataset(self.train_features)
        self.val_dataset = self.generate_dataset(self.val_features)
        logger.info("`setup` finished!")

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.train_batch_size,
                          num_workers=self.loader_workers)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, num_workers=self.loader_workers)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, num_workers=self.loader_workers)

    @classmethod
    def add_data_specific_args(
            cls, parser: ArgumentParser
    ) -> ArgumentParser:
        """ Parser for Estimator specific arguments/hyperparameters.
        :param parser: argparse.ArgumentParser
        Returns:
            - updated parser
        """
        parser.add_argument(
            "--data_path",
            default="/Data/enningxie/Codes/lightning-semantic-matching/data",
            type=str
        )
        parser.add_argument(
            "--max_length",
            default=64,
            type=int
        )
        parser.add_argument(
            "--train_batch_size",
            default=64,
            type=int
        )
        parser.add_argument(
            "--val_batch_size",
            default=64,
            type=int
        )
        parser.add_argument(
            "--loader_workers",
            default=64,
            type=int,
            help="How many subprocesses to use for data loading. 0 means that \
                        the data will be loaded in the main process.",
        )
        return parser


if __name__ == '__main__':
    tmp_parser = ArgumentParser()
    tmp_parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="/Data/public/pretrained_models/pytorch/chinese-bert-wwm-ext"
    )
    tmp_parser = SemanticMatchingDataModule.add_data_specific_args(tmp_parser)
    hparams = tmp_parser.parse_args()

    tmp_data_module = SemanticMatchingDataModule(hparams)
    tmp_data_module.prepare_data()
    tmp_data_module.setup()

    train_dataloader = tmp_data_module.val_dataloader()
    for batch in train_dataloader:
        print(type(batch))
        print(batch)
        print('break point.')
    print('break point.')

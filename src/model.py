# Created by xieenning at 2020/10/19
import logging as log
import pandas as pd
from argparse import ArgumentParser, Namespace
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
import numpy as np
from typing import Union, List, Optional
from collections import OrderedDict
from torchnlp.encoders import LabelEncoder
from torchnlp.utils import collate_tensors, lengths_to_mask
from pytorch_lightning import LightningModule, LightningDataModule
from transformers import BertForSequenceClassification, BertConfig, BertModel, BertTokenizer
from transformers import AdamW


class SemanticMatchingClassifier(LightningModule):
    """
    Sample model to show how to use a Transformer model to classify sentence.

    :param hparams: ArgumentParser containing the hyperparameters.
    """

    def __init__(self, hparams: Namespace) -> None:
        super().__init__()

        self.save_hyperparameters()
        self.hparams = hparams
        self.model_name_or_path = hparams.model_name_or_path
        self.num_labels = hparams.num_labels

        # build model
        self.model = self.__build_model()

        # loss criterion initialization
        self.__build_loss()

    def __build_model(self):
        """Init BERT model + tokenizer + classification head."""
        config = BertConfig.from_pretrained(self.model_name_or_path, num_labels=self.num_labels)
        return BertForSequenceClassification.from_pretrained(self.model_name_or_path, config=config)

    def __build_loss(self) -> None:
        """Initializes the loss functions."""
        self._loss = nn.modules.CrossEntropyLoss()

    def predict(self, sample: dict) -> dict:
        """ Predict function.
        :param sample: dictionary with the text we want to classify.

        :return: Dictionary with the input text and the predicted label.
        """
        self.eval()
        with torch.no_grad():
            model_input, _ = self.prepare_sample([sample], prepare_target=False)
            model_out = self.forward(**model_input)
            logits = model_out["logits"].numpy()
            predicted_labels = [
                self.data.label_encoder.index_to_token[prediction] for prediction in np.argmax(logits, axis=1)
            ]
            sample["predicted_label"] = predicted_labels[0]
        return sample

    def prepare_sample(self, sample: list, prepare_target: bool = True) -> (dict, dict):
        """
        Function that prepares a sample to input the model.
        :param sample: list of dictionaries.
        :param prepare_target:
        :return:
            - dictionary with the expected model inputs.
            - dictionary with the expected target labels.
        """
        sample = collate_tensors(sample)
        tokens, lengths = self.tokenizer(
            sample['text'],
            return_tensors='pt',
            padding=True,
            return_length=True,
            return_token_type_ids=False,
            return_attention_mask=False,
            truncation='only_first',
            max_length=512
        )

        inputs = {"tokens": tokens, "lengths": lengths}

        if not prepare_target:
            return inputs, {}

        # Prepare target:
        try:
            targets = {'labels': self.data.label_encoder.batch_encode(sample["label"])}
            return inputs, targets
        except RuntimeError:
            raise Exception("Label encoder found an unknown label.")

    def forward(self, input_ids, attention_mask, token_type_ids, label):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=label,
        )

    def loss(self, predictions: dict, targets: dict) -> torch.tensor:
        """
        Computes Loss value according to a loss function.
        :param predictions: model specific output. Must contain a key 'logits' with
            a tensor [batch_size x 1] with model predictions
        :param targets: Label values [batch_size]
        Returns:
            torch.tensor with loss value.
        """
        return self._loss(predictions["logits"], targets["labels"])

    def training_step(self, batch: tuple, batch_idx: int, *args, **kwargs) -> dict:
        input_ids, attention_mask, token_type_ids, label = batch
        model_out = self.forward(input_ids, attention_mask, token_type_ids, label)
        loss_train = model_out[0]
        # TODO: calculate loss by myself.
        self.log('train_loss', loss_train)
        return loss_train

    def validation_step(self, batch: tuple, batch_idx: int, *args, **kwargs) -> dict:
        """ Similar to the training step but with the model in eval mode.
        Returns:
            - dictionary passed to the validation_end function.
        """
        input_ids, attention_mask, token_type_ids, label = batch
        model_out = self.forward(input_ids, attention_mask, token_type_ids, label)
        loss_val = model_out[0]
        y_hat = model_out[1]
        y = label
        if self.num_labels >= 1:
            y_hat = torch.argmax(y_hat, dim=1)
        elif self.num_labels == 1:
            y_hat = y_hat.squeeze()

        val_acc = torch.sum(y == y_hat).item() / (len(y) * 1.0)
        val_acc = torch.tensor(val_acc)

        output = OrderedDict({"val_loss": loss_val, "val_acc": val_acc})
        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_end(self, outputs: list) -> dict:
        """ Function that takes as input a list of dictionaries returned by the validation_step
        function and measures the model performance accross the entire validation set.

        Returns:
            - Dictionary with metrics to be added to the lightning logger.
        """
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc_mean = torch.stack([x['val_acc'] for x in outputs]).mean()

        self.log("val_loss", val_loss_mean, prog_bar=True)
        self.log("val_acc", val_acc_mean, prog_bar=True)
        result = {
            "val_loss": val_loss_mean,
        }
        return result

    def configure_optimizers(self):
        """ Sets different Learning rates for different parameter groups. """
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        return [optimizer], []

    @classmethod
    def add_model_specific_args(
            cls, parser: ArgumentParser
    ) -> ArgumentParser:
        """ Parser for Estimator specific arguments/hyperparameters.
        :param parser: argparse.ArgumentParser
        Returns:
            - updated parser
        """
        parser.add_argument(
            "--model_name_or_path",
            default="/Data/enningxie/Pretrained_models/transformers_test/bert-base-uncased",
            type=str,
            help="Encoder model to be used.",
        )
        parser.add_argument(
            "--learning_rate",
            default=2e-05,
            type=float,
            help="Classification head learning rate.",
        )
        parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        parser.add_argument("--warmup_steps", default=0, type=int)
        parser.add_argument("--weight_decay", default=0.0, type=float)
        parser.add_argument("--num_labels", default=2, type=int)
        return parser

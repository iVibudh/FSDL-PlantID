"""Basic convolutional model building blocks."""
import argparse
from typing import Any, Dict

import torch
from torch import nn
import torch.nn.functional as F
import timm

import plant_id.metadata.inat as metadata


FC_DIM = 128
FC_DROPOUT = 0.25
MODEL_NAME = 'resnet50'


class Pretrained_CNN(nn.Module):
    """
    Loads a pretrained CNN from timm
    """

    def __init__(self, model_name: str) -> None:
        super().__init__()
        # TODO: is this getting to the GPU via args passed to the Lightning Trainer?
        try:
            self.pretrained = timm.create_model(model_name, pretrained=True, num_classes = FC_DIM)#.to(device)
        except Exception as e:
            print(e)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the ConvBlock to x.

        Parameters
        ----------
        x
            (B, C, H, W) tensor

        Returns
        -------
        torch.Tensor
            (B, FC_DIM) tensor
        """
        return self.pretrained(x)


class Finetuning_CNN(nn.Module):
    """Load a pretrained CNN and stick on a classifier for finetuning."""

    def __init__(self, data_config: Dict[str, Any], args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.data_config = data_config

        input_channels, input_height, input_width = self.data_config["input_dims"]
        assert (
            input_height == input_width
        ), f"input height and width should be equal, but was {input_height}, {input_width}"
        self.input_height, self.input_width = input_height, input_width

        num_classes = metadata.NUM_PLANT_CLASSES

        fc_dim = self.args.get("fc_dim", FC_DIM)
        fc_dropout = self.args.get("fc_dropout", FC_DROPOUT)
        model_name = self.args.get("model_name", MODEL_NAME)
        
        self.pretrained_model = Pretrained_CNN(model_name)
        self.dropout = nn.Dropout(fc_dropout)
        self.classifier = nn.Linear(fc_dim, num_classes)
        self._init_weights(self.classifier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the CNN to x.

        Parameters
        ----------
        x
            (B, Ch, H, W) tensor, where H and W must equal input height and width from data_config.

        Returns
        -------
        torch.Tensor
            (B, Cl) tensor
        """
        _B, _Ch, H, W = x.shape
        assert H == self.input_height and W == self.input_width, f"bad inputs to CNN with shape {x.shape}"
        x = self.pretrained_model(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def _init_weights(self, m):
        """
        Initialize weights in a better way than default.
        See https://github.com/pytorch/pytorch/issues/18182
        """
        nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            _fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
            bound = 1 / math.sqrt(fan_out)
            nn.init.normal_(m.bias, -bound, bound)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--fc_dim", type=int, default=FC_DIM)
        parser.add_argument("--fc_dropout", type=float, default=FC_DROPOUT)
        parser.add_argument("--model_name", type=str, default=MODEL_NAME)
        return parser


#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import pytest

from rbfn_head.norms import euclidian
from rbfn_head.radials import gaussian
from rbfn_head.head import RBFNProjectionHead, RBFNPredictionHead


class TestRBFNHead:
    @classmethod
    def setup_class(cls) -> None:
        cls.input_dim = 2048
        cls.hidden_dim = 2048
        cls.output_dim = 128
        cls.num_kernels = 2048
        cls.normalization = True
        cls.num_layers = 2
        cls.batch_norm = False
        cls.batch_norm_output = False
        cls.batch_size = 512

    def test_rbfn_projection_head(self) -> None:
        rbfn_projection_head = RBFNProjectionHead(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            num_kernels=self.hidden_dim,
            radial_function=gaussian,
            norm_function=euclidian,
            normalization=self.normalization,
            num_layers=self.num_layers,
            batch_norm=self.batch_norm,
        )
        print(rbfn_projection_head)

        backbone_output = torch.rand((self.batch_size, self.input_dim))
        projector_output = rbfn_projection_head(backbone_output)
        print(projector_output)

        assert torch.all(
            projector_output == projector_output[0, :]
        ), "Not all columns have the same values"

    def test_rbfn_prediction_head(self) -> None:
        rbfn_prediction_head = RBFNPredictionHead(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            num_kernels=self.hidden_dim,
            radial_function=gaussian,
            norm_function=euclidian,
            normalization=self.normalization,
        )
        print(rbfn_prediction_head)

        projector_output = torch.rand((self.batch_size, self.input_dim))
        predictor_output = rbfn_prediction_head(projector_output)
        print(predictor_output)

        assert torch.all(
            predictor_output == predictor_output[0, :]
        ), "Not all columns have the same values"


if __name__ == "__main__":
    pytest.main()

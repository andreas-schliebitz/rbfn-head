from torch import nn

from typing import Callable
from rbfn_head.layer import RBFLayer
from rbfn_head.norms import euclidian
from rbfn_head.radials import gaussian
from lightly.models.modules.heads import ProjectionHead


class RBFNProjectionHead(ProjectionHead):
    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 2048,
        output_dim: int = 128,
        num_kernels: int = 2048,
        radial_function: Callable = gaussian,
        norm_function: Callable = euclidian,
        normalization: bool = True,
        num_layers: int = 2,
        batch_norm: bool = False,
    ) -> None:
        layers: list[tuple[int, int, nn.Module | None, nn.Module | None]] = [
            (
                input_dim,
                hidden_dim,
                nn.BatchNorm1d(hidden_dim) if batch_norm else None,
                RBFLayer(
                    in_features_dim=hidden_dim,
                    num_kernels=num_kernels,
                    out_features_dim=hidden_dim,
                    radial_function=radial_function,
                    norm_function=norm_function,
                    normalization=normalization,
                ),
            )
        ]

        for _ in range(2, num_layers):
            layers.append(
                (
                    hidden_dim,
                    hidden_dim,
                    nn.BatchNorm1d(hidden_dim) if batch_norm else None,
                    RBFLayer(
                        in_features_dim=hidden_dim,
                        num_kernels=num_kernels,
                        out_features_dim=hidden_dim,
                        radial_function=radial_function,
                        norm_function=norm_function,
                        normalization=normalization,
                    ),
                )
            )

        layers.append(
            (
                hidden_dim,
                output_dim,
                nn.BatchNorm1d(output_dim) if batch_norm else None,
                None,
            )
        )
        super().__init__(layers)


class RBFNPredictionHead(ProjectionHead):
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 2048,
        output_dim: int = 128,
        num_kernels: int = 2048,
        radial_function: Callable = gaussian,
        norm_function: Callable = euclidian,
        normalization: bool = True,
    ) -> None:
        super(RBFNPredictionHead, self).__init__(
            [
                (
                    input_dim,
                    hidden_dim,
                    nn.BatchNorm1d(hidden_dim),
                    RBFLayer(
                        in_features_dim=hidden_dim,
                        num_kernels=num_kernels,
                        out_features_dim=hidden_dim,
                        radial_function=radial_function,
                        norm_function=norm_function,
                        normalization=normalization,
                    ),
                ),
                (hidden_dim, output_dim, None, None),
            ]
        )

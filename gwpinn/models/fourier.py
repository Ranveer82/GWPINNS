"""Random Fourier feature embeddings.

Plain coordinate MLPs are strongly biased towards low frequencies (the "spectral
bias" of Rahaman et al., 2019), so they converge to a smooth field and refuse to
resolve the short-scale structure of a heterogeneous aquifer. Mapping the input
through random Fourier features first removes that bias (Tancik et al., 2020),
and using *several* frequency bands rather than one avoids having to guess a
single correct bandwidth (Wang et al., 2021, "On the eigenvector bias of Fourier
feature networks").
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class RandomFourierFeatures(nn.Module):
    """Multi-scale embedding ``x -> [sin(2 pi B x), cos(2 pi B x)]``.

    The frequency matrix ``B`` is drawn once from ``N(0, sigma^2)`` per band and
    then held fixed - training the frequencies as well tends to destabilise the
    PDE residual early on.

    Parameters
    ----------
    in_dim:
        Dimension of the (normalised) input coordinates.
    n_features:
        Total number of frequencies, split evenly across ``sigmas``.
    sigmas:
        Standard deviations of the frequency bands. With coordinates scaled to
        ``[-1, 1]``, a band of ``sigma`` resolves features of roughly
        ``1/sigma`` in normalised units.
    """

    def __init__(
        self,
        in_dim: int,
        n_features: int = 64,
        sigmas: Sequence[float] = (1.0, 3.0, 6.0),
        trainable: bool = False,
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        sigmas = [float(s) for s in sigmas if s > 0] or [1.0]
        per_band = max(1, n_features // len(sigmas))

        blocks = []
        for s in sigmas:
            b = torch.randn(per_band, in_dim, dtype=dtype, generator=generator) * s
            blocks.append(b)
        B = torch.cat(blocks, dim=0)

        if trainable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("B", B)

        self.in_dim = in_dim
        self.sigmas = tuple(sigmas)
        self.out_dim = 2 * B.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * torch.pi * (x @ self.B.T)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    def extra_repr(self) -> str:
        return (
            f"in_dim={self.in_dim}, out_dim={self.out_dim}, "
            f"sigmas={self.sigmas}"
        )


__all__ = ["RandomFourierFeatures"]

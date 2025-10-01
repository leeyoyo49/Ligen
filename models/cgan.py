import torch, torch.nn as nn
from typing import Optional, Sequence


def _get_activation(name: str):
    name = (name or "leakyrelu").lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name in ("gelu",):
        return nn.GELU()
    # default
    return nn.LeakyReLU(0.2, inplace=True)


class CGANGenerator(nn.Module):
    """MLP Generator with optional BatchNorm and configurable hidden sizes/activation.

    Defaults kept close to original unless cfg overrides them.
    """

    def __init__(
        self,
        noise_dim: int,
        coord_dim: int,
        fingerprint_dim: int,
        cfg: Optional[object] = None,
    ):
        super().__init__()
        cfg = cfg or object()

        # configuration with safe defaults
        hidden: Sequence[int] = getattr(cfg, "gen_hidden", [128, 256])
        use_bn: bool = bool(getattr(cfg, "gen_use_bn", True))
        dropout: float = float(getattr(cfg, "gen_dropout", 0.0) or 0.0)
        act = _get_activation(getattr(cfg, "gen_activation", "leakyrelu"))

        layers: list[nn.Module] = []
        in_dim = noise_dim + coord_dim
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act)
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, fingerprint_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, coords, noise):
        x = torch.cat([coords, noise], dim=1)
        return self.net(x)


class CGANDiscriminator(nn.Module):
    """MLP Discriminator with optional SpectralNorm/Dropout; outputs logits (no Sigmoid).
    Use BCEWithLogitsLoss on the outputs.
    """

    def __init__(
        self,
        coord_dim: int,
        fingerprint_dim: int,
        cfg: Optional[object] = None,
    ):
        super().__init__()
        cfg = cfg or object()

        hidden: Sequence[int] = getattr(cfg, "disc_hidden", [128, 256])
        use_sn: bool = bool(getattr(cfg, "d_spectral_norm", False))
        dropout: float = float(getattr(cfg, "disc_dropout", 0.0) or 0.0)
        act = _get_activation(getattr(cfg, "disc_activation", "leakyrelu"))

        def maybe_sn(linear: nn.Linear) -> nn.Module:
            return nn.utils.spectral_norm(linear) if use_sn else linear

        layers: list[nn.Module] = []
        in_dim = coord_dim + fingerprint_dim
        for h in hidden:
            layers.append(maybe_sn(nn.Linear(in_dim, h)))
            layers.append(act)
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(maybe_sn(nn.Linear(in_dim, 1)))  # logits, no Sigmoid

        self.net = nn.Sequential(*layers)

    def forward(self, coords, fps):
        x = torch.cat([coords, fps], dim=1)
        return self.net(x)

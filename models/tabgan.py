import torch.nn as nn
from typing import Optional, Sequence


def _get_activation(name: str):
    name = (name or "leakyrelu").lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name in ("gelu",):
        return nn.GELU()
    return nn.LeakyReLU(0.2, inplace=True)


class Generator(nn.Module):
    """Configurable MLP Generator for tabular data.

    Args:
        z_dim: latent input dimension
        hidden_size: kept for backward compat if hidden list not provided
        output_size: number of features to generate
        cfg: optional object with attributes:
             - gen_hidden: list of hidden sizes (default [hidden_size, hidden_size])
             - gen_use_bn: bool BatchNorm (default True)
             - gen_dropout: float dropout (default 0.0)
             - gen_activation: str activation name (relu|leakyrelu|gelu)
    """

    def __init__(self, z_dim: int, hidden_size: int, output_size: int = 6, cfg: Optional[object] = None):
        super().__init__()
        cfg = cfg or object()
        hidden: Sequence[int] = getattr(cfg, "gen_hidden", None) or [hidden_size, hidden_size]
        use_bn: bool = bool(getattr(cfg, "gen_use_bn", True))
        dropout: float = float(getattr(cfg, "gen_dropout", 0.0) or 0.0)
        act = _get_activation(getattr(cfg, "gen_activation", "leakyrelu"))

        layers = []
        in_dim = z_dim
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act)
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    """Configurable MLP Discriminator; outputs logits (no Sigmoid)."""

    def __init__(self, input_size: int = 6, hidden_size: int = 256, cfg: Optional[object] = None):
        super().__init__()
        cfg = cfg or object()
        hidden: Sequence[int] = getattr(cfg, "disc_hidden", None) or [hidden_size, hidden_size]
        dropout: float = float(getattr(cfg, "disc_dropout", 0.0) or 0.0)
        use_sn: bool = bool(getattr(cfg, "d_spectral_norm", False))
        act = _get_activation(getattr(cfg, "disc_activation", "leakyrelu"))

        def maybe_sn(linear: nn.Linear) -> nn.Module:
            return nn.utils.spectral_norm(linear) if use_sn else linear

        layers = []
        in_dim = input_size
        for h in hidden:
            layers.append(maybe_sn(nn.Linear(in_dim, h)))
            layers.append(act)
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(maybe_sn(nn.Linear(in_dim, 1)))  # logits
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# Aliases to keep trainer imports stable
TabGenerator = Generator
TabDiscriminator = Discriminator

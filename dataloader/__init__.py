# runtime-select the dataset module: data.wifi or data.spectral
import importlib

def api_for_kind(kind: str):
    """
    Returns the dataset module for a given kind ('wifi' or 'spectral').

    Each module should implement:
      - load_df(path)
      - make_feature_loader(df, x_cols, batch_size)
      - make_pair_loader(df, y_cols, x_cols, batch_size)
      - split_by_frac(df, frac, seed, y_cols=None)
      - make_mlp_dataloaders(df, x_cols, y_cols, batch_size, seed)
    """
    return importlib.import_module(f"dataloader.{kind}")

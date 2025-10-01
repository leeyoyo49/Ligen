import pandas as pd, numpy as np, torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.utils.data import Dataset

def split_by_frac(df: pd.DataFrame, frac: float, seed: int, y_cols=None):
    """Split by target columns when provided; returns (train_df, test_df)."""
    if y_cols is not None:
        y_cols = [c for c in y_cols if c in df.columns]
    if y_cols and len(df) > 0:
        uniq_y = df[y_cols].drop_duplicates()
        tr_y = uniq_y.sample(frac=frac, random_state=seed) if len(uniq_y) > 0 else uniq_y
        tr_set = set(map(tuple, tr_y.values))
        mask = df[y_cols].apply(tuple, axis=1).isin(tr_set)
        train_df = df[mask]
        test_df = df[~mask]
    else:
        train_df = df.sample(frac=frac, random_state=seed)
        test_df = df.drop(train_df.index)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

def make_mlp_dataloaders(df: pd.DataFrame, x_cols, y_cols, batch_size: int, seed: int):
    """Returns (all_loader, train_loader, test_loader) for MLP (X->coords)."""
    X = torch.from_numpy(df[x_cols].values.astype("float32"))
    y = torch.from_numpy(df[y_cols].values.astype("float32"))
    ds = TensorDataset(X, y)

    torch.manual_seed(seed)
    tr_size = int(0.9 * len(ds))
    te_size = len(ds) - tr_size
    tr_ds, te_ds = random_split(ds, [tr_size, te_size])

    all_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    tr_loader  = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    te_loader  = DataLoader(te_ds, batch_size=batch_size, shuffle=False)
    return all_loader, tr_loader, te_loader

def load_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

class FeatureDataset(Dataset):
    """For TabGAN: returns only features X."""
    def __init__(self, df: pd.DataFrame, x_cols):
        self.X = df[x_cols].values.astype("float32")
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i])

class PairDataset(Dataset):
    """For CGAN: returns (coords, features)."""
    def __init__(self, df: pd.DataFrame, y_cols, x_cols):
        self.C = df[y_cols].values.astype("float32")
        self.X = df[x_cols].values.astype("float32")
    def __len__(self): return len(self.C)
    def __getitem__(self, i):
        return torch.from_numpy(self.C[i]), torch.from_numpy(self.X[i])

def make_feature_loader(df: pd.DataFrame, x_cols, batch_size: int):
    ds = FeatureDataset(df, x_cols=x_cols)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)

def make_pair_loader(df: pd.DataFrame, y_cols, x_cols, batch_size: int):
    ds = PairDataset(df, y_cols=y_cols, x_cols=x_cols)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)

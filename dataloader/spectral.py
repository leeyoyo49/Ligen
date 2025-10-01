import pandas as pd, numpy as np, torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.utils.data import Dataset



def split_by_frac(df: pd.DataFrame, frac: float, seed: int, y_cols=None):
    """Split by target columns when provided; returns (train_df, test_df).

    Behavior:
    - If y_cols is provided and all exist in df, sample unique y tuples and
      keep all rows of selected tuples together in the train split.
    - Otherwise, fallback to simple row-wise random split.
    """

    # 檢查是否被額外條件（例如 ap/fold/mode）過濾光
    # print(test_df['ap'].value_counts(), ...) 視你的欄位而定

    # Prefer grouping by target columns to avoid leaking targets across splits
    if y_cols is not None:
        # filter to only columns present to validate
        y_cols = [c for c in y_cols if c in df.columns]

    if y_cols and len(df) > 0:
        # sample unique target tuples
        uniq_pairs = df[y_cols].drop_duplicates()
        tr_pairs = uniq_pairs.sample(frac=frac, random_state=seed) if len(uniq_pairs) > 0 else uniq_pairs

        # build membership mask: rows whose y tuple is in sampled pairs
        tr_set = set(map(tuple, tr_pairs.values))
        mask = df[y_cols].apply(tuple, axis=1).isin(tr_set)
        train_df = df[mask]
        test_df = df[~mask]
    else:
        # fallback: row-wise split
        train_df = df.sample(frac=frac, random_state=seed)
        test_df = df.drop(train_df.index)
    
    print("len(train_df) =", len(train_df))
    print("len(test_df)  =", len(test_df))
    print("y_cols =", y_cols)

    if len(test_df) == 0:
        # 觀察為什麼為 0
        print("test_df head (before dropna):")
        print(test_df.head())

        # 檢查是否因為 NaN 被濾掉
        if y_cols:
            print("NaNs in test y:", test_df[y_cols].isna().sum().sum())


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
    df = pd.read_csv(path)
    if "sample_id" in df.columns: df = df.drop(columns=["sample_id"])
    return df

class FeatureDataset(Dataset):
    def __init__(self, df: pd.DataFrame, x_cols):
        self.X = df[x_cols].values.astype("float32")
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        import torch
        return torch.from_numpy(self.X[i])

class PairDataset(Dataset):
    def __init__(self, df: pd.DataFrame, y_cols, x_cols):
        self.C = df[y_cols].values.astype("float32")
        self.X = df[x_cols].values.astype("float32")
    def __len__(self): return len(self.C)
    def __getitem__(self, i):
        import torch
        return torch.from_numpy(self.C[i]), torch.from_numpy(self.X[i])

def make_feature_loader(df: pd.DataFrame, x_cols, batch_size: int):
    return DataLoader(FeatureDataset(df, x_cols=x_cols), batch_size=batch_size, shuffle=True)

def make_pair_loader(df: pd.DataFrame, y_cols, x_cols, batch_size: int):
    x = df[x_cols].values.astype("float32")
    y = df[y_cols].values.astype("float32")
    if len(x) != len(y):
        raise ValueError(f"make_pair_loader: x/y length mismatch {len(x)} != {len(y)}")
    ds = PairDataset(df, y_cols=y_cols, x_cols=x_cols)
    
    return DataLoader(ds, batch_size=batch_size, shuffle=True)

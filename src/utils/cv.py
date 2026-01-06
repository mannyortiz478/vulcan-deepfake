from typing import List, Tuple
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def stratified_kfold_df(df: pd.DataFrame, label_col: str = "label", n_splits: int = 5, random_state: int = 42) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Create stratified K-fold splits from a DataFrame preserving label distribution.

    Returns a list of (train_df, val_df) tuples.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")

    labels = df[label_col].astype(str).values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    folds = []
    for train_idx, val_idx in skf.split(df, labels):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        folds.append((train_df, val_df))

    return folds

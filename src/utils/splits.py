from typing import Tuple
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def stratified_split_df(df: pd.DataFrame, label_col: str = "label", val_size: float = 0.1, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split a DataFrame into train and validation DataFrames preserving label distribution.

    Args:
        df: DataFrame containing at least the label_col column
        label_col: name of the column with class labels
        val_size: fraction reserved for validation
        random_state: RNG seed

    Returns:
        (train_df, val_df)
    """
    if val_size <= 0.0:
        return df.copy().reset_index(drop=True), pd.DataFrame(columns=df.columns)

    labels = df[label_col].astype(str).values
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    train_idx, val_idx = next(splitter.split(X=df.index.values.reshape(-1, 1), y=labels))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    return train_df, val_df

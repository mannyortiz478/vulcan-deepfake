import pandas as pd
from src.utils.cv import stratified_kfold_df


def test_stratified_kfold_preserves_ratio():
    # create imbalanced df
    n_b = 80
    n_s = 20
    df = pd.DataFrame({
        "path": [f"/fake/path/{i}.wav" for i in range(n_b + n_s)],
        "label": ["bonafide"] * n_b + ["spoof"] * n_s,
        "attack_type": ["N/A"] * (n_b + n_s),
    })

    folds = stratified_kfold_df(df, label_col="label", n_splits=5, random_state=123)
    assert len(folds) == 5

    ratios = []
    for train_df, val_df in folds:
        # compute spoof fraction in val
        spoof_frac = (val_df['label'] == 'spoof').mean()
        ratios.append(spoof_frac)

    # all ratios should be close to original 0.2
    for r in ratios:
        assert abs(r - 0.2) < 0.05

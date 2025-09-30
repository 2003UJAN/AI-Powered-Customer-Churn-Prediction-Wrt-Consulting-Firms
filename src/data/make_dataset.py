# src/data/make_dataset.py
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Colab-safe sys.path fix ---
ROOT_DIR = "/content/churn-prediction-genai"
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from data.generate_synthetic import generate_synthetic


def get_dataset(path="data/synthetic_churn.csv", force_generate=False):
    """Load dataset or generate if missing."""
    abs_path = os.path.join(ROOT_DIR, path)
    if force_generate or not os.path.exists(abs_path):
        print("Dataset not found. Generating synthetic dataset...")
        generate_synthetic(outpath=abs_path)
    return pd.read_csv(abs_path)


def split_dataset(
    df, test_size=0.2, val_size=0.1, random_state=42, outdir="data/processed"
):
    """Split dataset into train/val/test and save CSVs."""
    outdir = os.path.join(ROOT_DIR, outdir)
    os.makedirs(outdir, exist_ok=True)

    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["churn"], random_state=random_state
    )
    train_df, val_df = train_test_split(
        train_df, test_size=val_size, stratify=train_df["churn"], random_state=random_state
    )

    train_df.to_csv(os.path.join(outdir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(outdir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(outdir, "test.csv"), index=False)

    print(f"✅ Saved train/val/test splits in {outdir}/")
    return train_df, val_df, test_df


if __name__ == "__main__":
    df = get_dataset()
    split_dataset(df)

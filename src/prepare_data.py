from pathlib import Path
import argparse
from .data import load_dataset

INITIAL_CROPS = ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane"]

def main():
    parser = argparse.ArgumentParser(description="Prepare the public India district crop yield panel")
    parser.add_argument("--input", default="data/raw/crop_yield_dataset.csv")
    parser.add_argument("--output", default="data/processed/crop_yield.csv")
    args = parser.parse_args()
    df = load_dataset(args.input)
    df = df[df.crop.isin(INITIAL_CROPS)].copy()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df):,} rows for {df.crop.nunique()} crops, {df.state.nunique()} states, years {int(df.year.min())}-{int(df.year.max())} to {args.output}")

if __name__ == "__main__":
    main()

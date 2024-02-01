#!/usr/bin/env python3


from pathlib import Path
from utils import load_demographic_data
PROJECT_ROOT = Path.cwd()

def main():
    """Clean ADNI scanner 1 data."""
    # ----------------------------------------------------------------------------------------
    participants_path = PROJECT_ROOT / 'data' / 'y.csv'
    ids_path = PROJECT_ROOT / 'data' / 'av45.csv'

    output_ids_filename = 'cleaned_ids.csv'
    # ----------------------------------------------------------------------------------------
    # Create experiment's output directory
    outputs_dir = PROJECT_ROOT / 'outputs'
    outputs_dir.mkdir(exist_ok=True)

    dataset = load_demographic_data(participants_path, ids_path)


    output_ids_df = dataset[['IID']]

    assert sum(output_ids_df.duplicated()) == 0

    output_ids_df.to_csv(outputs_dir / output_ids_filename, index=False)


if __name__ == "__main__":
    main()
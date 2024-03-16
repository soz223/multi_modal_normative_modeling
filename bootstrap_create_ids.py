#!/usr/bin/env python3
#"""Script to create the files with the ids of the subjects from UK BIOBANK included in each bootstrap iteration.
#These ids are used to train the normative approach.
#"""
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path.cwd()


def main():

    n_bootstrap = 10

    participants_path = PROJECT_ROOT / 'data' / 'y.csv'

    output_dir = PROJECT_ROOT / 'outputs' 
    output_dir.mkdir(exist_ok=True)
    bootstrap_dir = output_dir / 'bootstrap_analysis'
    bootstrap_dir.mkdir(exist_ok=True)

    np.random.seed(42)

    ids_df = pd.read_csv(participants_path)
    IID = ids_df['IID']

    fraction = 0.8

    ids_dir = bootstrap_dir / 'ids'
    ids_dir.mkdir(exist_ok=True)

    HC_group = ids_df[ids_df['DIA'] == 2]
    MCI_group = ids_df[ids_df['DIA'] == 1]
    AD_group = ids_df[ids_df['DIA'] == 0]

    for i_bootstrap in tqdm(range(n_bootstrap)):
        # get 80% of the HC_group
        HC_80perc = HC_group.sample(frac=fraction, replace=True)
        # get rest of the 20% of the HC_group
        HC_20perc = HC_group[~HC_group.index.isin(HC_80perc.index)]

        train_df = HC_80perc
        # test_df is they combine the rest of the HC_group and the MCI_group and AD_group
        test_df = pd.concat([HC_20perc, MCI_group, AD_group])
        train_ids_filename = 'cleaned_bootstrap_{:03d}.csv'.format(i_bootstrap)
        test_ids_filename = 'cleaned_bootstrap_test_{:03d}.csv'.format(i_bootstrap)

        # only keep the IID column
        train_df = train_df[['IID']]
        test_df = test_df[['IID']]
        train_df.to_csv(ids_dir / train_ids_filename, index=False)
        test_df.to_csv(ids_dir / test_ids_filename, index=False)





# def main():
#     """Creates the csv files with the ids of the subjects used to train the normative model."""
#     # ----------------------------------------------------------------------------------------
#     n_bootstrap = 10
#     ids_path = PROJECT_ROOT / 'outputs' / 'train_ids.csv'
#     # ----------------------------------------------------------------------------------------
#     # Create experiment's output directory
#     outputs_dir = PROJECT_ROOT / 'outputs'
#     bootstrap_dir = outputs_dir / 'bootstrap_analysis'
#     bootstrap_dir.mkdir(exist_ok=True)

#     # Set random seed for random sampling of subjects
#     np.random.seed(42)

#     ids_df = pd.read_csv(ids_path)
#     n_sub = len(ids_df)

#     ids_dir = bootstrap_dir / 'ids'
#     ids_dir.mkdir(exist_ok=True)

#     for i_bootstrap in tqdm(range(n_bootstrap)):
#         bootstrap_ids = ids_df.sample(n=n_sub, replace=True)

#         ids_filename = 'cleaned_bootstrap_{:03d}.csv'.format(i_bootstrap)
#         bootstrap_ids.to_csv(ids_dir / ids_filename, index=False)




if __name__ == "__main__":
    main()

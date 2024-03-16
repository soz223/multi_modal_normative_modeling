#!/usr/bin/env python3


from pathlib import Path
import pandas as pd
from utils import load_demographic_data
PROJECT_ROOT = Path.cwd()

def main():
    n_bootstrap = 10

    ids_path = PROJECT_ROOT / 'data' / 'modality_label576_90.xlsx'

    # read sub sheet

    subsheet_names = ['av45', 'fdg', 'vbm', 'snp']

    # read the data, 

    # define a empty DIA here
    DIA_df = pd.DataFrame()

    for subsheet_name in subsheet_names:
        # read ids_path subsheet_name
        ids_df = pd.read_excel(ids_path, sheet_name=subsheet_name)
        # print('ids_df', ids_df)
        
        # set IID and DIA as DIA_df only 2 columns
        DIA_df['IID'] = ids_df['IID']
        DIA_df['DIA'] = ids_df['DIA']
        # keep IID as index, delete the DIA column
        data_df = ids_df.drop(columns=['DIA'])
        data_df.set_index('IID', inplace=True)

        # fill the NaN with  the mean of the column
        data_df.fillna(data_df.mean(), inplace=True)

        # save them to data/subsheet_name.csv
        data_df.to_csv(PROJECT_ROOT / 'data' / f'{subsheet_name}.csv')

    # change the DIA to, AD is 0, MCI is 1, CN is 2
    # save DIA_df to data/y.csv
    DIA_df['DIA'] = DIA_df['DIA'].map({'AD': 0, 'MCI': 1, 'CN': 2})

    # read data from data/AV45_cov.csv
    cov_df = pd.read_csv(PROJECT_ROOT / 'data' / 'AV45_cov.csv').drop(columns=['DIA'])

    # merge the DIA_df and cov_df on IID
    DIA_df = pd.merge(DIA_df, cov_df, on='IID')

    # save DIA_df to data/y.csv
    DIA_df.to_csv(PROJECT_ROOT / 'data' / 'y.csv', index=False)
    
    

        
    

# def main():
#     """Clean ADNI scanner 1 data."""
#     # ----------------------------------------------------------------------------------------
#     participants_path = PROJECT_ROOT / 'data' / 'y.csv'
#     ids_path = PROJECT_ROOT / 'data' / 'av45.csv'

#     output_ids_filename = 'cleaned_ids.csv'
#     # ----------------------------------------------------------------------------------------
#     # Create experiment's output directory
#     outputs_dir = PROJECT_ROOT / 'outputs'
#     outputs_dir.mkdir(exist_ok=True)

#     dataset = load_demographic_data(participants_path, ids_path)


#     output_ids_df = dataset[['IID']]

#     assert sum(output_ids_df.duplicated()) == 0

#     output_ids_df.to_csv(outputs_dir / output_ids_filename, index=False)


if __name__ == "__main__":
    main()
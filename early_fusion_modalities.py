import pandas as pd
from pathlib import Path
from utils import COLUMNS_HCP, COLUMNS_NAME, load_dataset, COLUMNS_NAME_SNP, COLUMNS_NAME_VBM, COLUMNS_3MODALITIES, COLUMNS_NAME_AAL116, COLUMNS_NAME_HCP_fMRI_100, get_column_name, get_datasets_name, get_hc_label, generate_kfold_ids_endtoend, generate_kfold_ids_with_unigroup, generate_kfold_ids


dataset_resources = ['ADNI', 'ADHD', 'HCPimage']

PROJECT_ROOT = Path.cwd()

for dataset_resource in dataset_resources:
    dataset_names = get_datasets_name(dataset_resource)
    merged_modalities = pd.DataFrame()
    iid = None
    for dataset_name in dataset_names:
        freesurfer_path = PROJECT_ROOT / 'data' / f'{dataset_resource}' / f'{dataset_name}.csv'
        # read to a df
        df = pd.read_csv(freesurfer_path)
        print(f'{dataset_name} shape: {df.shape}')
        # set the iid as idx
        df.set_index('IID', inplace=True)

        # for each column except the iid, rename the column to include the dataset name
        for col in df.columns:
            if col != 'IID':
                df.rename(columns={col: f'{col}_{dataset_name}'}, inplace=True)

        # merge the df using the index
        if iid is None:
            iid = df.index
        else:
            assert all(iid == df.index)
        merged_modalities = pd.concat([merged_modalities, df], axis=1)

    # save the merged df as early_fusion_modalities_{dataset_resource}.csv
    merged_modalities.to_csv(PROJECT_ROOT / 'data'  / dataset_resource / f'early_fusion_modalities_{dataset_resource}.csv')
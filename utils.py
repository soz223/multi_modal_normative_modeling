"""Helper functions and constants."""
from pathlib import Path
import warnings

import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_ROOT = Path.cwd()


# use whole dataset to generate kfold ids
def generate_kfold_ids_endtoend(HC_group, other_group, oversample_percentage=1, n_splits=5, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    kfold_dir = PROJECT_ROOT / 'outputs' / 'kfold_analysis_endtoend'
    kfold_dir.mkdir(parents=True, exist_ok=True)

    all_group = pd.concat([HC_group, other_group])
    for fold, (train_idx, test_idx) in enumerate(kf.split(all_group)):
        train_ids = all_group.iloc[train_idx]['IID']
        test_ids = all_group.iloc[test_idx]['IID']

        oversample_size = int(len(train_ids) * (oversample_percentage))
        train_ids_oversampled = np.random.choice(train_ids, size=oversample_size, replace=True)
        train_ids = pd.Series(train_ids_oversampled)
        # name the train_ids column as IID
        train_ids = pd.DataFrame(train_ids)
        train_ids.columns = ['IID']

        train_ids_path = kfold_dir / f'train_ids_{fold:03d}.csv'
        test_ids_path = kfold_dir / f'test_ids_{fold:03d}.csv'

        train_ids.to_csv(train_ids_path, index=False)
        test_ids.to_csv(test_ids_path, index=False)

        print(f'Fold {fold} - Train IDs saved to: {train_ids_path}, Test IDs saved to: {test_ids_path}')





# The following are the implementations of train, test, and analyze functions with necessary adjustments

def generate_kfold_ids_with_unigroup(HC_group, other_group, oversample_percentage=1, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    kfold_dir = PROJECT_ROOT / 'outputs' / 'kfold_analysis'
    kfold_dir.mkdir(parents=True, exist_ok=True)

    for fold, (train_idx, test_idx) in enumerate(kf.split(HC_group)):
        train_ids = HC_group.iloc[train_idx]['IID']
        test_ids_hc = HC_group.iloc[test_idx]['IID']

        oversample_size = int(len(train_ids) * oversample_percentage)
        train_ids_oversampled = np.random.choice(train_ids, size=oversample_size, replace=True)
        train_ids = pd.DataFrame({'IID': train_ids_oversampled})

        test_ids_other = other_group['IID']
        test_ids = pd.concat([test_ids_hc, test_ids_other])

        train_ids_path = kfold_dir / f'train_ids_{fold:03d}.csv'
        test_ids_path = kfold_dir / f'test_ids_{fold:03d}.csv'

        train_ids.to_csv(train_ids_path, index=False)
        test_ids.to_csv(test_ids_path, index=False)


def generate_kfold_ids(HC_group, other_group, oversample_percentage=1, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    kfold_dir = PROJECT_ROOT / 'outputs' / 'kfold_analysis'
    kfold_dir.mkdir(parents=True, exist_ok=True)
    full_group = pd.concat([HC_group, other_group])

    for fold, (train_idx, test_idx) in enumerate(kf.split(full_group)):
        train_ids = full_group.iloc[train_idx]['IID']
        test_ids = full_group.iloc[test_idx]['IID']


        oversample_size = int(len(train_ids) * oversample_percentage)
        train_ids_oversampled = np.random.choice(train_ids, size=oversample_size, replace=True)
        train_ids = pd.DataFrame({'IID': train_ids_oversampled})


        train_ids_path = kfold_dir / f'train_ids_{fold:03d}.csv'
        test_ids_path = kfold_dir / f'test_ids_{fold:03d}.csv'

        train_ids.to_csv(train_ids_path, index=False)
        test_ids.to_csv(test_ids_path, index=False)



def cliff_delta(X, Y):
    """Calculate the effect size using the Cliff's delta."""
    lx = len(X)
    ly = len(Y)
    mat = np.zeros((lx, ly))
    for i in range(0, lx):
        for j in range(0, ly):
            if X[i] > Y[j]:
                mat[i, j] = 1
            elif Y[j] > X[i]:
                mat[i, j] = -1

    return (np.sum(mat)) / (lx * ly)


def load_dataset(demographic_path, ids_path, freesurfer_path):
    """Load dataset."""

    # do nothing. nothing demographic data is introduced
    demographic_data = load_demographic_data(demographic_path, ids_path)
    # print("demographic_data aaaaaaaaaaaaaaaaaaaa", demographic_data)

    freesurfer_df = pd.read_csv(freesurfer_path)

    dataset_df = pd.merge(freesurfer_df, demographic_data, on='IID')

    return dataset_df


def load_demographic_data(demographic_path, ids_path):
    """Load dataset using selected ids."""

    demographic_df = pd.read_csv(demographic_path)
    demographic_df = demographic_df.dropna()

    print(ids_path)
    
    ids_df = pd.read_csv(ids_path, usecols=['IID'])
    dataset_df = None

    if 'Run_ID' in demographic_df.columns:
        demographic_df['uid'] = demographic_df['participant_id'] + '_' + demographic_df['Session_ID'] + '_run-' + \
                                demographic_df['Run_ID'].apply(str)

        ids_df['uid'] = ids_df['IID'].str.split('_').str[0] + '_' + ids_df['IID'].str.split('_').str[1]+ '_' + ids_df['IID'].str.split('_').str[2]

        dataset_df = pd.merge(ids_df, demographic_df, on='uid')
        dataset_df = dataset_df.drop(columns=['uid'])

    elif 'Session_ID' in demographic_df.columns:
        demographic_df['uid'] = demographic_df['participant_id'] + '_' + demographic_df['Session_ID']

        ids_df['uid'] = ids_df['IID'].str.split('_').str[0] + '_' + ids_df['IID'].str.split('_').str[1]

        dataset_df = pd.merge(ids_df, demographic_df, on='uid')
        dataset_df = dataset_df.drop(columns=['uid'])

    else:
        ids_df['participant_id'] = ids_df['IID']

        ids_df_participant_id = ids_df['participant_id']

        

        dataset_df = pd.merge(ids_df, demographic_df, on='IID')
        # print('ids_df cccccccccccccccccccc', ids_df)
        # print('demographic_df path cccccccccccccccccccc', demographic_path)
        # print("demographic_df cccccccccccccccc", demographic_df)
        # print("dataset_df cccccccccccccccc", dataset_df)
        return dataset_df


    return dataset_df


#COLUMNS_NAME = [(lambda x: 'C'+ str(x))(y) for y in list(range(5,105))]

COLUMNS_HCP = [(lambda x: 'HCP_'+ str(x))(y) for y in list(range(132))]



COLUMNS_3MODALITIES = [
'Precentral_L_av45',
 'Precentral_R_av45',
 'Frontal_Sup_L_av45',
 'Frontal_Sup_R_av45',
 'Frontal_Sup_Orb_L_av45',
 'Frontal_Sup_Orb_R_av45',
 'Frontal_Mid_L_av45',
 'Frontal_Mid_R_av45',
 'Frontal_Mid_Orb_L_av45',
 'Frontal_Mid_Orb_R_av45',
 'Frontal_Inf_Oper_L_av45',
 'Frontal_Inf_Oper_R_av45',
 'Frontal_Inf_Tri_L_av45',
 'Frontal_Inf_Tri_R_av45',
 'Frontal_Inf_Orb_L_av45',
 'Frontal_Inf_Orb_R_av45',
 'Rolandic_Oper_L_av45',
 'Rolandic_Oper_R_av45',
 'Supp_Motor_Area_L_av45',
 'Supp_Motor_Area_R_av45',
 'Olfactory_L_av45',
 'Olfactory_R_av45',
 'Frontal_Sup_Medial_L_av45',
 'Frontal_Sup_Medial_R_av45',
 'Frontal_Med_Orb_L_av45',
 'Frontal_Med_Orb_R_av45',
 'Rectus_L_av45',
 'Rectus_R_av45',
 'Insula_L_av45',
 'Insula_R_av45',
 'Cingulum_Ant_L_av45',
 'Cingulum_Ant_R_av45',
 'Cingulum_Mid_L_av45',
 'Cingulum_Mid_R_av45',
 'Cingulum_Post_L_av45',
 'Cingulum_Post_R_av45',
 'Hippocampus_L_av45',
 'Hippocampus_R_av45',
 'ParaHippocampal_L_av45',
 'ParaHippocampal_R_av45',
 'Amygdala_L_av45',
 'Amygdala_R_av45',
 'Calcarine_L_av45',
 'Calcarine_R_av45',
 'Cuneus_L_av45',
 'Cuneus_R_av45',
 'Lingual_L_av45',
 'Lingual_R_av45',
 'Occipital_Sup_L_av45',
 'Occipital_Sup_R_av45',
 'Occipital_Mid_L_av45',
 'Occipital_Mid_R_av45',
 'Occipital_Inf_L_av45',
 'Occipital_Inf_R_av45',
 'Fusiform_L_av45',
 'Fusiform_R_av45',
 'Postcentral_L_av45',
 'Postcentral_R_av45',
 'Parietal_Sup_L_av45',
 'Parietal_Sup_R_av45',
 'Parietal_Inf_L_av45',
 'Parietal_Inf_R_av45',
 'SupraMarginal_L_av45',
 'SupraMarginal_R_av45',
 'Angular_L_av45',
 'Angular_R_av45',
 'Precuneus_L_av45',
 'Precuneus_R_av45',
 'Paracentral_Lobule_L_av45',
 'Paracentral_Lobule_R_av45',
 'Caudate_L_av45',
 'Caudate_R_av45',
 'Putamen_L_av45',
 'Putamen_R_av45',
 'Pallidum_L_av45',
 'Pallidum_R_av45',
 'Thalamus_L_av45',
 'Thalamus_R_av45',
 'Heschl_L_av45',
 'Heschl_R_av45',
 'Temporal_Sup_L_av45',
 'Temporal_Sup_R_av45',
 'Temporal_Pole_Sup_L_av45',
 'Temporal_Pole_Sup_R_av45',
 'Temporal_Mid_L_av45',
 'Temporal_Mid_R_av45',
 'Temporal_Pole_Mid_L_av45',
 'Temporal_Pole_Mid_R_av45',
 'Temporal_Inf_L_av45',
 'Temporal_Inf_R_av45',
 'Precentral_L_fdg',
 'Precentral_R_fdg',
 'Frontal_Sup_L_fdg',
 'Frontal_Sup_R_fdg',
 'Frontal_Sup_Orb_L_fdg',
 'Frontal_Sup_Orb_R_fdg',
 'Frontal_Mid_L_fdg',
 'Frontal_Mid_R_fdg',
 'Frontal_Mid_Orb_L_fdg',
 'Frontal_Mid_Orb_R_fdg',
 'Frontal_Inf_Oper_L_fdg',
 'Frontal_Inf_Oper_R_fdg',
 'Frontal_Inf_Tri_L_fdg',
 'Frontal_Inf_Tri_R_fdg',
 'Frontal_Inf_Orb_L_fdg',
 'Frontal_Inf_Orb_R_fdg',
 'Rolandic_Oper_L_fdg',
 'Rolandic_Oper_R_fdg',
 'Supp_Motor_Area_L_fdg',
 'Supp_Motor_Area_R_fdg',
 'Olfactory_L_fdg',
 'Olfactory_R_fdg',
 'Frontal_Sup_Medial_L_fdg',
 'Frontal_Sup_Medial_R_fdg',
 'Frontal_Med_Orb_L_fdg',
 'Frontal_Med_Orb_R_fdg',
 'Rectus_L_fdg',
 'Rectus_R_fdg',
 'Insula_L_fdg',
 'Insula_R_fdg',
 'Cingulum_Ant_L_fdg',
 'Cingulum_Ant_R_fdg',
 'Cingulum_Mid_L_fdg',
 'Cingulum_Mid_R_fdg',
 'Cingulum_Post_L_fdg',
 'Cingulum_Post_R_fdg',
 'Hippocampus_L_fdg',
 'Hippocampus_R_fdg',
 'ParaHippocampal_L_fdg',
 'ParaHippocampal_R_fdg',
 'Amygdala_L_fdg',
 'Amygdala_R_fdg',
 'Calcarine_L_fdg',
 'Calcarine_R_fdg',
 'Cuneus_L_fdg',
 'Cuneus_R_fdg',
 'Lingual_L_fdg',
 'Lingual_R_fdg',
 'Occipital_Sup_L_fdg',
 'Occipital_Sup_R_fdg',
 'Occipital_Mid_L_fdg',
 'Occipital_Mid_R_fdg',
 'Occipital_Inf_L_fdg',
 'Occipital_Inf_R_fdg',
 'Fusiform_L_fdg',
 'Fusiform_R_fdg',
 'Postcentral_L_fdg',
 'Postcentral_R_fdg',
 'Parietal_Sup_L_fdg',
 'Parietal_Sup_R_fdg',
 'Parietal_Inf_L_fdg',
 'Parietal_Inf_R_fdg',
 'SupraMarginal_L_fdg',
 'SupraMarginal_R_fdg',
 'Angular_L_fdg',
 'Angular_R_fdg',
 'Precuneus_L_fdg',
 'Precuneus_R_fdg',
 'Paracentral_Lobule_L_fdg',
 'Paracentral_Lobule_R_fdg',
 'Caudate_L_fdg',
 'Caudate_R_fdg',
 'Putamen_L_fdg',
 'Putamen_R_fdg',
 'Pallidum_L_fdg',
 'Pallidum_R_fdg',
 'Thalamus_L_fdg',
 'Thalamus_R_fdg',
 'Heschl_L_fdg',
 'Heschl_R_fdg',
 'Temporal_Sup_L_fdg',
 'Temporal_Sup_R_fdg',
 'Temporal_Pole_Sup_L_fdg',
 'Temporal_Pole_Sup_R_fdg',
 'Temporal_Mid_L_fdg',
 'Temporal_Mid_R_fdg',
 'Temporal_Pole_Mid_L_fdg',
 'Temporal_Pole_Mid_R_fdg',
 'Temporal_Inf_L_fdg',
 'Temporal_Inf_R_fdg',
 'MNI_Amygdala_L_vbm',
 'MNI_Amygdala_R_vbm',
 'MNI_Angular_L_vbm',
 'MNI_Angular_R_vbm',
 'MNI_Calcarine_L_vbm',
 'MNI_Calcarine_R_vbm',
 'MNI_Caudate_L_vbm',
 'MNI_Caudate_R_vbm',
 'MNI_Cingulum_Ant_L_vbm',
 'MNI_Cingulum_Ant_R_vbm',
 'MNI_Cingulum_Mid_L_vbm',
 'MNI_Cingulum_Mid_R_vbm',
 'MNI_Cingulum_Post_L_vbm',
 'MNI_Cingulum_Post_R_vbm',
 'MNI_Cuneus_L_vbm',
 'MNI_Cuneus_R_vbm',
 'MNI_Frontal_Inf_Oper_L_vbm',
 'MNI_Frontal_Inf_Oper_R_vbm',
 'MNI_Frontal_Inf_Orb_L_vbm',
 'MNI_Frontal_Inf_Orb_R_vbm',
 'MNI_Frontal_Inf_Tri_L_vbm',
 'MNI_Frontal_Inf_Tri_R_vbm',
 'MNI_Frontal_Med_Orb_L_vbm',
 'MNI_Frontal_Med_Orb_R_vbm',
 'MNI_Frontal_Mid_L_vbm',
 'MNI_Frontal_Mid_R_vbm',
 'MNI_Frontal_Mid_Orb_L_vbm',
 'MNI_Frontal_Mid_Orb_R_vbm',
 'MNI_Frontal_Sup_L_vbm',
 'MNI_Frontal_Sup_R_vbm',
 'MNI_Frontal_Sup_Medial_L_vbm',
 'MNI_Frontal_Sup_Medial_R_vbm',
 'MNI_Frontal_Sup_Orb_L_vbm',
 'MNI_Frontal_Sup_Orb_R_vbm',
 'MNI_Fusiform_L_vbm',
 'MNI_Fusiform_R_vbm',
 'MNI_Heschl_L_vbm',
 'MNI_Heschl_R_vbm',
 'MNI_Hippocampus_L_vbm',
 'MNI_Hippocampus_R_vbm',
 'MNI_Insula_L_vbm',
 'MNI_Insula_R_vbm',
 'MNI_Lingual_L_vbm',
 'MNI_Lingual_R_vbm',
 'MNI_Occipital_Inf_L_vbm',
 'MNI_Occipital_Inf_R_vbm',
 'MNI_Occipital_Mid_L_vbm',
 'MNI_Occipital_Mid_R_vbm',
 'MNI_Occipital_Sup_L_vbm',
 'MNI_Occipital_Sup_R_vbm',
 'MNI_Olfactory_L_vbm',
 'MNI_Olfactory_R_vbm',
 'MNI_Pallidum_L_vbm',
 'MNI_Pallidum_R_vbm',
 'MNI_ParaHippocampal_L_vbm',
 'MNI_ParaHippocampal_R_vbm',
 'MNI_Paracentral_Lobule_L_vbm',
 'MNI_Paracentral_Lobule_R_vbm',
 'MNI_Parietal_Inf_L_vbm',
 'MNI_Parietal_Inf_R_vbm',
 'MNI_Parietal_Sup_L_vbm',
 'MNI_Parietal_Sup_R_vbm',
 'MNI_Postcentral_L_vbm',
 'MNI_Postcentral_R_vbm',
 'MNI_Precentral_L_vbm',
 'MNI_Precentral_R_vbm',
 'MNI_Precuneus_L_vbm',
 'MNI_Precuneus_R_vbm',
 'MNI_Putamen_L_vbm',
 'MNI_Putamen_R_vbm',
 'MNI_Rectus_L_vbm',
 'MNI_Rectus_R_vbm',
 'MNI_Rolandic_Oper_L_vbm',
 'MNI_Rolandic_Oper_R_vbm',
 'MNI_Supp_Motor_Area_L_vbm',
 'MNI_Supp_Motor_Area_R_vbm',
 'MNI_SupraMarginal_L_vbm',
 'MNI_SupraMarginal_R_vbm',
 'MNI_Temporal_Inf_L_vbm',
 'MNI_Temporal_Inf_R_vbm',
 'MNI_Temporal_Mid_L_vbm',
 'MNI_Temporal_Mid_R_vbm',
 'MNI_Temporal_Pole_Mid_L_vbm',
 'MNI_Temporal_Pole_Mid_R_vbm',
 'MNI_Temporal_Pole_Sup_L_vbm',
 'MNI_Temporal_Pole_Sup_R_vbm',
 'MNI_Temporal_Sup_L_vbm',
 'MNI_Temporal_Sup_R_vbm',
 'MNI_Thalamus_L_vbm',
 'MNI_Thalamus_R_vbm',
 
]
from nilearn.datasets import fetch_atlas_aal
aal_atlas = fetch_atlas_aal()
COLUMNS_NAME_AAL116 = aal_atlas.labels


COLUMNS_NAME = ['Precentral_L',
'Precentral_R',
'Frontal_Sup_L',
'Frontal_Sup_R',
'Frontal_Sup_Orb_L',
'Frontal_Sup_Orb_R',
'Frontal_Mid_L',
'Frontal_Mid_R',
'Frontal_Mid_Orb_L',
'Frontal_Mid_Orb_R',
'Frontal_Inf_Oper_L',
'Frontal_Inf_Oper_R',
'Frontal_Inf_Tri_L',
'Frontal_Inf_Tri_R',
'Frontal_Inf_Orb_L',
'Frontal_Inf_Orb_R',
'Rolandic_Oper_L',
'Rolandic_Oper_R',
'Supp_Motor_Area_L',
'Supp_Motor_Area_R',
'Olfactory_L',
'Olfactory_R',
'Frontal_Sup_Medial_L',
'Frontal_Sup_Medial_R',
'Frontal_Med_Orb_L',
'Frontal_Med_Orb_R',
'Rectus_L',
'Rectus_R',
'Insula_L',
'Insula_R',
'Cingulum_Ant_L',
'Cingulum_Ant_R',
'Cingulum_Mid_L',
'Cingulum_Mid_R',
'Cingulum_Post_L',
'Cingulum_Post_R',
'Hippocampus_L',
'Hippocampus_R',
'ParaHippocampal_L',
'ParaHippocampal_R',
'Amygdala_L',
'Amygdala_R',
'Calcarine_L',
'Calcarine_R',
'Cuneus_L',
'Cuneus_R',
'Lingual_L',
'Lingual_R',
'Occipital_Sup_L',
'Occipital_Sup_R',
'Occipital_Mid_L',
'Occipital_Mid_R',
'Occipital_Inf_L',
'Occipital_Inf_R',
'Fusiform_L',
'Fusiform_R',
'Postcentral_L',
'Postcentral_R',
'Parietal_Sup_L',
'Parietal_Sup_R',
'Parietal_Inf_L',
'Parietal_Inf_R',
'SupraMarginal_L',
'SupraMarginal_R',
'Angular_L',
'Angular_R',
'Precuneus_L',
'Precuneus_R',
'Paracentral_Lobule_L',
'Paracentral_Lobule_R',
'Caudate_L',
'Caudate_R',
'Putamen_L',
'Putamen_R',
'Pallidum_L',
'Pallidum_R',
'Thalamus_L',
'Thalamus_R',
'Heschl_L',
'Heschl_R',
'Temporal_Sup_L',
'Temporal_Sup_R',
'Temporal_Pole_Sup_L',
'Temporal_Pole_Sup_R',
'Temporal_Mid_L',
'Temporal_Mid_R',
'Temporal_Pole_Mid_L',
'Temporal_Pole_Mid_R',
'Temporal_Inf_L',
'Temporal_Inf_R',
]



COLUMNS_NAME_VBM = ['MNI_Amygdala_L',
'MNI_Amygdala_R',
'MNI_Angular_L',
'MNI_Angular_R',
'MNI_Calcarine_L',
'MNI_Calcarine_R',
'MNI_Caudate_L',
'MNI_Caudate_R',
'MNI_Cingulum_Ant_L',
'MNI_Cingulum_Ant_R',
'MNI_Cingulum_Mid_L',
'MNI_Cingulum_Mid_R',
'MNI_Cingulum_Post_L',
'MNI_Cingulum_Post_R',
'MNI_Cuneus_L',
'MNI_Cuneus_R',
'MNI_Frontal_Inf_Oper_L',
'MNI_Frontal_Inf_Oper_R',
'MNI_Frontal_Inf_Orb_L',
'MNI_Frontal_Inf_Orb_R',
'MNI_Frontal_Inf_Tri_L',
'MNI_Frontal_Inf_Tri_R',
'MNI_Frontal_Med_Orb_L',
'MNI_Frontal_Med_Orb_R',
'MNI_Frontal_Mid_L',
'MNI_Frontal_Mid_R',
'MNI_Frontal_Mid_Orb_L',
'MNI_Frontal_Mid_Orb_R',
'MNI_Frontal_Sup_L',
'MNI_Frontal_Sup_R',
'MNI_Frontal_Sup_Medial_L',
'MNI_Frontal_Sup_Medial_R',
'MNI_Frontal_Sup_Orb_L',
'MNI_Frontal_Sup_Orb_R',
'MNI_Fusiform_L',
'MNI_Fusiform_R',
'MNI_Heschl_L',
'MNI_Heschl_R',
'MNI_Hippocampus_L',
'MNI_Hippocampus_R',
'MNI_Insula_L',
'MNI_Insula_R',
'MNI_Lingual_L',
'MNI_Lingual_R',
'MNI_Occipital_Inf_L',
'MNI_Occipital_Inf_R',
'MNI_Occipital_Mid_L',
'MNI_Occipital_Mid_R',
'MNI_Occipital_Sup_L',
'MNI_Occipital_Sup_R',
'MNI_Olfactory_L',
'MNI_Olfactory_R',
'MNI_Pallidum_L',
'MNI_Pallidum_R',
'MNI_ParaHippocampal_L',
'MNI_ParaHippocampal_R',
'MNI_Paracentral_Lobule_L',
'MNI_Paracentral_Lobule_R',
'MNI_Parietal_Inf_L',
'MNI_Parietal_Inf_R',
'MNI_Parietal_Sup_L',
'MNI_Parietal_Sup_R',
'MNI_Postcentral_L',
'MNI_Postcentral_R',
'MNI_Precentral_L',
'MNI_Precentral_R',
'MNI_Precuneus_L',
'MNI_Precuneus_R',
'MNI_Putamen_L',
'MNI_Putamen_R',
'MNI_Rectus_L',
'MNI_Rectus_R',
'MNI_Rolandic_Oper_L',
'MNI_Rolandic_Oper_R',
'MNI_Supp_Motor_Area_L',
'MNI_Supp_Motor_Area_R',
'MNI_SupraMarginal_L',
'MNI_SupraMarginal_R',
'MNI_Temporal_Inf_L',
'MNI_Temporal_Inf_R',
'MNI_Temporal_Mid_L',
'MNI_Temporal_Mid_R',
'MNI_Temporal_Pole_Mid_L',
'MNI_Temporal_Pole_Mid_R',
'MNI_Temporal_Pole_Sup_L',
'MNI_Temporal_Pole_Sup_R',
'MNI_Temporal_Sup_L',
'MNI_Temporal_Sup_R',
'MNI_Thalamus_L',
'MNI_Thalamus_R',
]

COLUMNS_NAME_SNP = ['rs4575098',
'rs6656401',
'rs2093760',
'rs4844610',
'rs4663105',
'rs6733839',
'rs10933431',
'rs35349669',
'rs6448453',
'rs190982',
'rs9271058',
'rs9473117',
'rs9381563',
'rs10948363',
'rs2718058',
'rs4723711',
'rs1859788',
'rs1476679',
'rs12539172',
'rs10808026',
'rs7810606',
'rs11771145',
'rs28834970',
'rs73223431',
'rs4236673',
'rs9331896',
'rs11257238',
'rs7920721',
'rs3740688',
'rs10838725',
'rs983392',
'rs7933202',
'rs2081545',
'rs867611',
'rs10792832',
'rs3851179',
'rs17125924',
'rs17125944',
'rs10498633',
'rs12881735',
'rs12590654',
'rs442495',
'rs59735493',
'rs113260531',
'rs28394864',
'rs111278892',
'rs3752246',
'rs4147929',
'rs41289512',
'rs3865444',
'rs6024870',
'rs6014724',
'rs7274581',
'rs429358',
]

COLUMNS_NAME_PPMI = [f'{i}' for i in range(3485)]

def get_column_name(dataset_resourse, dataset_name):
    if dataset_resourse == 'ADNI':
        if dataset_name == 'av45' or dataset_name == 'fdg':
            columns_name = COLUMNS_NAME
        elif dataset_name == 'snp':
            columns_name = COLUMNS_NAME_SNP
        elif dataset_name == 'vbm':
            columns_name = COLUMNS_NAME_VBM
    elif dataset_resourse == 'HCP':
        columns_name = [(lambda x: dataset_name + '_'+ str(x))(y) for y in list(range(132))]
    elif dataset_resourse == 'ADHD' or dataset_resourse == 'HCPimage':
        columns_name = COLUMNS_NAME_AAL116
    elif dataset_resourse == 'PPMI':
        columns_name = COLUMNS_NAME_PPMI

    # if dataset_name is early_fusion_modalities_{dataset_resourse}, 
    # then the column name is the union of all columns in dataset_names,
    # respectively each dataset_name's columns_name with a _{dataset_name} suffix.
    if dataset_name.startswith('early_fusion_modalities'):
        dataset_names = get_datasets_name(dataset_resourse)
        columns_name = []
        for dataset_name in dataset_names:
            columns_name_of_one = []
            columns_name_of_one += get_column_name(dataset_resourse, dataset_name)
            columns_name_of_one = [(lambda x:  str(x) + '_'+ dataset_name)(y) for y in columns_name_of_one]
            columns_name += columns_name_of_one
        

    return columns_name



def get_datasets_name(dataset_resourse, procedure='SE-PoE'):
    if procedure.startswith('SM'):
        single_modality = procedure.split('-')[-1]
        dataset_names = [single_modality]
        return dataset_names
    if dataset_resourse == 'ADNI':    
        dataset_names = ['av45', 'vbm', 'fdg']
    elif dataset_resourse == 'HCP':
        dataset_names = ['T1_volume', 'mean_T1_intensity', 'mean_FA', 'mean_MD', 'mean_L1', 'mean_L2', 'mean_L3', 'min_BOLD', '25_percentile_BOLD', '50_percentile_BOLD', '75_percentile_BOLD', 'max_BOLD']
    elif dataset_resourse == 'ADHD':
        dataset_names = ['fMRI', 'sMRI']
    elif dataset_resourse == 'PPMI':
        dataset_names = ['PPMI_new_modal1_upper_tri', 'PPMI_new_modal2_upper_tri', 'PPMI_new_modal3_upper_tri']
    elif dataset_resourse == 'HCPimage':
        dataset_names = ['T1w_sMRI', 'T2w_sMRI', 'fMRI']
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset_resourse))
    
    if procedure.startswith('UCA'):
        # add a dataset, which is the union of all datasets in dataset_names
        dataset_names.append(f'early_fusion_modalities_{dataset_resourse}')
    
    return dataset_names




def get_hc_label(dataset_resourse):

    if dataset_resourse == 'ADNI':
        hc_label = 2
    elif dataset_resourse == 'HCP':
        hc_label = 1
    elif dataset_resourse == 'ADHD':
        hc_label = 2
    elif dataset_resourse == 'PPMI':
        hc_label = 1
    elif dataset_resourse == 'HCPimage':
        hc_label = 1
    else:
        raise ValueError('Unknown dataset resource')
    return hc_label
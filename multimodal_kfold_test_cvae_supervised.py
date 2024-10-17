#!/home/songlinzhao/anaconda3/envs/normodiff/bin/python3
"""Inference the predictions of the clinical datasets using the supervised model."""
import argparse
from pathlib import Path
from sklearn.preprocessing import RobustScaler, OneHotEncoder
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split
import torch

from tqdm import tqdm
import copy
from utils import COLUMNS_HCP, COLUMNS_NAME, load_dataset
from os.path import join, exists
from VAE import VAE
from sklearn.preprocessing import RobustScaler
from utils_vae import plot_losses, MyDataset_labels, Logger, reconstruction_deviation, latent_deviation, separate_latent_deviation, latent_pvalues, reconstruction_deviation_seperate_roi
import argparse
from utils import COLUMNS_HCP, COLUMNS_NAME, load_dataset, COLUMNS_NAME_SNP, COLUMNS_NAME_VBM, COLUMNS_3MODALITIES, COLUMNS_NAME_AAL116
from sklearn.model_selection import KFold

from cVAE import cVAE_multimodal_endtoend

PROJECT_ROOT = Path.cwd()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(dataset_resourse, combine, procedure, n_splits=5):
    """Make predictions using trained normative models."""
    # ----------------------------------------------------------------------------
    model_name = 'supervised_cvae'

    participants_path = PROJECT_ROOT / 'data' / dataset_resourse / 'y.csv'
    # dataset_name to str
    # freesurfer_path = PROJECT_ROOT / 'data' / dataset_resourse / (dataset_name + '.csv')

    # ----------------------------------------------------------------------------
    # Create directories structure
    outputs_dir = PROJECT_ROOT / 'outputs'
    kfold_dir = outputs_dir / 'kfold_analysis'
    model_dir = kfold_dir / model_name
    # ids_path = outputs_dir / (dataset_name + '_homogeneous_ids.csv')

    # ----------------------------------------------------------------------------
    # Set random seed
    random_seed = 42
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    MCK = 5
    dc_output_list = [0] * MCK
    
    error_mean = []
    # ----------------------------------------------------------------------------
    if dataset_resourse == 'ADNI':
        if procedure.startswith('SingleModality'):
            dataset_names = ['av45']
        elif procedure.startswith('SE'):
            dataset_names = ['av45', 'vbm', 'fdg']
        elif procedure.startswith('UCA'):
            dataset_names = ['av45', 'vbm', 'fdg', '3modalities']
        else:
            raise ValueError('Unknown procedure: {}'.format(procedure))
    elif dataset_resourse == 'HCP':
        dataset_names = ['T1_volume', 'mean_T1_intensity', 'mean_FA', 'mean_MD', 'mean_L1', 'mean_L2', 'mean_L3', 'min_BOLD', '25_percentile_BOLD', '50_percentile_BOLD', '75_percentile_BOLD', 'max_BOLD']
    elif dataset_resourse == 'ADHD':
        dataset_names = ['sMRI', 'fMRI']
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset_resourse))

    if combine is None:
        raise ValueError(f'Unknown procedure: {procedure}')

    
    modalities = len(dataset_names)

    ids_df = pd.read_csv(participants_path)
    if dataset_resourse == 'ADNI':
        hc_label = 2
    elif dataset_resourse == 'HCP':
        hc_label = 1
    elif dataset_resourse == 'ADHD':
        hc_label = 1
    else:
        raise ValueError('Unknown dataset resource')
    
    HC_group = ids_df[ids_df['DIA'] == hc_label]

    for fold, (train_idx, test_idx) in enumerate(kf.split(HC_group)):

        train_ids_path = kfold_dir / 'train_ids_{:03d}.csv'.format(fold)
        test_ids_path = kfold_dir / 'test_ids_{:03d}.csv'.format(fold)
        
        
        fold_model_dir = model_dir / '{:03d}'.format(fold)
        fold_model_dir.mkdir(exist_ok=True)
        
        test_data_list = []
        clinical_df_list = []
        val_data_list = []

        for dataset_name in dataset_names:
            if dataset_resourse == 'ADNI':
                if dataset_name == 'av45' or dataset_name == 'fdg':
                    columns_name = COLUMNS_NAME
                elif dataset_name == 'snp':
                    columns_name = COLUMNS_NAME_SNP
                elif dataset_name == 'vbm':
                    columns_name = COLUMNS_NAME_VBM
                elif dataset_name == '3modalities':
                    columns_name = COLUMNS_3MODALITIES
            elif dataset_resourse == 'HCP':
                columns_name = [(lambda x: dataset_name + '_'+ str(x))(y) for y in list(range(132))]
            elif dataset_resourse == 'ADHD':
                columns_name = COLUMNS_NAME_AAL116

            freesurfer_path = PROJECT_ROOT / 'data' / dataset_resourse / (dataset_name + '.csv')
            train_dataset_df = load_dataset(participants_path, train_ids_path, freesurfer_path)
            test_dataset_df = load_dataset(participants_path, test_ids_path, freesurfer_path)

            if dataset_resourse == 'ADNI':
                hc_label = 2
            elif dataset_resourse == 'HCP':
                hc_label = 1
            elif dataset_resourse == 'ADHD':
                hc_label = 1    
            else:
                raise ValueError('Unknown dataset resource')

            train_dataset_df = train_dataset_df.loc[train_dataset_df['DIA'] == hc_label]
            train_data = train_dataset_df[columns_name].values
            scaler = RobustScaler()
            train_data = pd.DataFrame(scaler.fit_transform(train_data))     
            
            train_covariates = train_dataset_df[['DIA','AGE', 'PTGENDER']]
            train_covariates.DIA[train_covariates.DIA == 0] = 0

            test_data = test_dataset_df[columns_name].values
            test_data = pd.DataFrame(scaler.transform(test_data))

            # # seperate 20% of test data for validation, randomly, using 42 as seed
            # test_data, val_data = train_test_split(test_data, test_size=0.2, random_state=42)



            test_covariates = test_dataset_df[['DIA','AGE', 'PTGENDER']]
            test_covariates.DIA[test_covariates.DIA == 0] = 0

            bin_labels = list(range(0,27))  
            AGE_bins_test, bin_edges = pd.qcut(test_covariates['AGE'].rank(method="first"), q=27, retbins=True, labels=bin_labels)
            one_hot_AGE = np.eye(27)[AGE_bins_test.values]

            gender_bins_test, bin_edges = pd.qcut(test_covariates['PTGENDER'].rank(method="first"), q=2, retbins=True, labels=list(range(0,2)) )
            one_hot_gender = np.eye(2)[gender_bins_test.values]

            test_data_list.append(test_data)
            clinical_df_list.append(test_dataset_df)

        one_hot_covariates_test = np.concatenate((one_hot_AGE, one_hot_gender), axis=1).astype('float32')

        if exists(join(fold_model_dir, 'cVAE_model.pkl')):
            print('load trained model')
            model = torch.load(join(fold_model_dir, 'cVAE_model.pkl'))  
            model.to(DEVICE)
        else:
            print('firstly train model')
            
        test_prediction_list = model.pred_recon(test_data_list, one_hot_covariates_test, DEVICE, combine=combine)

        output_data = pd.DataFrame(clinical_df_list[0].DIA.values, columns=['DIA'])

        output_data_reconstruction_deviation_list = model.reconstruction_deviation_multimodal(test_data_list, test_prediction_list)

        for dataset_name, test_prediction in zip(dataset_names, test_prediction_list):
            if dataset_resourse == 'ADNI':
                if dataset_name == 'av45' or dataset_name == 'fdg':
                    columns_name = COLUMNS_NAME
                elif dataset_name == 'snp':
                    columns_name = COLUMNS_NAME_SNP
                elif dataset_name == 'vbm':
                    columns_name = COLUMNS_NAME_VBM
                elif dataset_name == '3modalities':
                    columns_name = COLUMNS_3MODALITIES
            elif dataset_resourse == 'HCP':
                columns_name = [(lambda x: dataset_name + '_'+ str(x))(y) for y in list(range(132))]
            elif dataset_resourse == 'ADHD':
                columns_name = COLUMNS_NAME_AAL116
            output_dataset_dir = fold_model_dir / dataset_name
            output_dataset_dir.mkdir(exist_ok=True)

            normalized_df = pd.DataFrame(columns=['participant_id'] + columns_name)
            normalized_df['participant_id'] = clinical_df_list[0]['participant_id']
            normalized_df[columns_name] = test_data_list[dataset_names.index(dataset_name)]
            normalized_df.to_csv(output_dataset_dir / 'normalized_{}.csv'.format(dataset_name), index=False)

            reconstruction_df = pd.DataFrame(columns=['participant_id'] + columns_name)
            reconstruction_df['participant_id'] = clinical_df_list[0]['participant_id']
            print('test_prediction:', test_prediction.shape)
            reconstruction_df[columns_name] = test_prediction
            reconstruction_df.to_csv(output_dataset_dir / 'reconstruction_{}.csv'.format(dataset_name), index=False)

            output_data['reconstruction_deviation'] = output_data_reconstruction_deviation_list[dataset_names.index(dataset_name)]

            error_mean.append(output_data['reconstruction_deviation'])

            reconstruction_error_df = pd.DataFrame(columns=['participant_id', 'Reconstruction error'])
            reconstruction_error_df['participant_id'] = clinical_df_list[0]['participant_id']
            reconstruction_error_df['Reconstruction error'] = output_data['reconstruction_deviation']
            reconstruction_error_df.to_csv(output_dataset_dir / 'reconstruction_error_{}.csv'.format(dataset_name), index=False)


# def main_val(dataset_resourse, procedure, n_splits=5):
#     """Make predictions using trained normative models."""
#     # ----------------------------------------------------------------------------
#     model_name = 'supervised_cvae'

#     participants_path = PROJECT_ROOT / 'data' / dataset_resourse / 'y.csv'
#     # dataset_name to str
#     # freesurfer_path = PROJECT_ROOT / 'data' / dataset_resourse / (dataset_name + '.csv')

#     # ----------------------------------------------------------------------------
#     # Create directories structure
#     outputs_dir = PROJECT_ROOT / 'outputs'
#     kfold_dir = outputs_dir / 'kfold_analysis'
#     model_dir = kfold_dir / model_name
#     # ids_path = outputs_dir / (dataset_name + '_homogeneous_ids.csv')

#     # ----------------------------------------------------------------------------
#     # Set random seed
#     random_seed = 42
#     tf.random.set_seed(random_seed)
#     np.random.seed(random_seed)
    
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

#     MCK = 5
#     dc_output_list = [0] * MCK
    
#     error_mean = []
#     # ----------------------------------------------------------------------------
#     if procedure.startswith('SE'):
#         dataset_names = ['av45', 'vbm', 'fdg']
#     elif procedure.startswith('UCA'):
#         dataset_names = ['av45', 'vbm', 'fdg', '3modalities']
#     else:
#         raise ValueError('Unknown procedure: {}'.format(procedure))


#     if combine is None:
#         raise ValueError(f'Unknown procedure: {procedure}')

    
#     modalities = len(dataset_names)

#     ids_df = pd.read_csv(participants_path)
#     HC_group = ids_df[ids_df['DIA'] == 2]

#     for fold, (train_idx, test_idx) in enumerate(kf.split(HC_group)):

#         train_ids_path = kfold_dir / 'train_ids_{:03d}.csv'.format(fold)
#         test_ids_path = kfold_dir / 'test_ids_{:03d}.csv'.format(fold)
#         val_ids_path = kfold_dir / 'val_ids_{:03d}.csv'.format(fold)
        
        
#         fold_model_dir = model_dir / '{:03d}'.format(fold)
#         fold_model_dir.mkdir(exist_ok=True)
        
#         test_data_list = []
#         clinical_df_list = []
#         val_data_list = []
#         clinical_df_list_val = []

#         for dataset_name in dataset_names:
#             if dataset_resourse == 'ADNI':
#                 if dataset_name == 'av45' or dataset_name == 'fdg':
#                     columns_name = COLUMNS_NAME
#                 elif dataset_name == 'snp':
#                     columns_name = COLUMNS_NAME_SNP
#                 elif dataset_name == 'vbm':
#                     columns_name = COLUMNS_NAME_VBM
#                 elif dataset_name == '3modalities':
#                     columns_name = COLUMNS_3MODALITIES
#             elif dataset_resourse == 'HCP':
#                 columns_name = [(lambda x: dataset_name + '_'+ str(x))(y) for y in list(range(132))]

#             freesurfer_path = PROJECT_ROOT / 'data' / dataset_resourse / (dataset_name + '.csv')
#             train_dataset_df = load_dataset(participants_path, train_ids_path, freesurfer_path)
#             test_dataset_df = load_dataset(participants_path, test_ids_path, freesurfer_path)
#             val_dataset_df = load_dataset(participants_path, val_ids_path, freesurfer_path)

#             if dataset_resourse == 'ADNI':
#                 hc_label = 2
#             elif dataset_resourse == 'HCP':
#                 hc_label = 1
#             else:
#                 raise ValueError('Unknown dataset resource')

#             train_dataset_df = train_dataset_df.loc[train_dataset_df['DIA'] == hc_label]
#             train_data = train_dataset_df[columns_name].values
#             scaler = RobustScaler()
#             train_data = pd.DataFrame(scaler.fit_transform(train_data))     
            
#             train_covariates = train_dataset_df[['DIA','AGE', 'PTGENDER']]
#             train_covariates.DIA[train_covariates.DIA == 0] = 0

#             test_data = test_dataset_df[columns_name].values
#             test_data = pd.DataFrame(scaler.transform(test_data))

#             val_data = val_dataset_df[columns_name].values
#             val_data = pd.DataFrame(scaler.transform(val_data))

#             # # seperate 20% of test data for validation, randomly, using 42 as seed
#             # test_data, val_data = train_test_split(test_data, test_size=0.2, random_state=42)



#             test_covariates = test_dataset_df[['DIA','AGE', 'PTGENDER']]
#             test_covariates.DIA[test_covariates.DIA == 0] = 0

#             bin_labels = list(range(0,27))  
#             AGE_bins_test, bin_edges = pd.qcut(test_covariates['AGE'].rank(method="first"), q=27, retbins=True, labels=bin_labels)
#             one_hot_AGE = np.eye(27)[AGE_bins_test.values]

#             gender_bins_test, bin_edges = pd.qcut(test_covariates['PTGENDER'].rank(method="first"), q=2, retbins=True, labels=list(range(0,2)) )
#             one_hot_gender = np.eye(2)[gender_bins_test.values]


#             val_covariates = val_dataset_df[['DIA','AGE', 'PTGENDER']]
#             val_covariates.DIA[val_covariates.DIA == 0] = 0

#             bin_labels = list(range(0,27))
#             AGE_bins_val, bin_edges = pd.qcut(val_covariates['AGE'].rank(method="first"), q=27, retbins=True, labels=bin_labels)
#             one_hot_AGE_val = np.eye(27)[AGE_bins_val.values]

#             gender_bins_val, bin_edges = pd.qcut(val_covariates['PTGENDER'].rank(method="first"), q=2, retbins=True, labels=list(range(0,2)) )
#             one_hot_gender_val = np.eye(2)[gender_bins_val.values]

#             val_data_list.append(val_data)

#             test_data_list.append(test_data)
#             clinical_df_list.append(test_dataset_df)

#             clinical_df_list_val.append(val_dataset_df)


#         one_hot_covariates_test = np.concatenate((one_hot_AGE, one_hot_gender), axis=1).astype('float32')
#         one_hot_covariates_val = np.concatenate((one_hot_AGE_val, one_hot_gender_val), axis=1).astype('float32')

#         if exists(join(fold_model_dir, 'cVAE_model.pkl')):
#             print('load trained model')
#             model = torch.load(join(fold_model_dir, 'cVAE_model.pkl'))  
#             model.to(DEVICE)
#         else:
#             print('firstly train model')
            
#         test_prediction_list = model.pred_recon(test_data_list, one_hot_covariates_test, DEVICE, combine=combine)
#         val_prediction_list = model.pred_recon(val_data_list, one_hot_covariates_val, DEVICE, combine=combine)

#         output_data = pd.DataFrame(clinical_df_list[0].DIA.values, columns=['DIA'])
#         output_data_reconstruction_deviation_list = model.reconstruction_deviation_multimodal(test_data_list, test_prediction_list)


#         for dataset_name, test_prediction in zip(dataset_names, test_prediction_list):
#             if dataset_resourse == 'ADNI':
#                 if dataset_name == 'av45' or dataset_name == 'fdg':
#                     columns_name = COLUMNS_NAME
#                 elif dataset_name == 'snp':
#                     columns_name = COLUMNS_NAME_SNP
#                 elif dataset_name == 'vbm':
#                     columns_name = COLUMNS_NAME_VBM
#                 elif dataset_name == '3modalities':
#                     columns_name = COLUMNS_3MODALITIES
#             elif dataset_resourse == 'HCP':
#                 columns_name = [(lambda x: dataset_name + '_'+ str(x))(y) for y in list(range(132))]

#             output_dataset_dir = fold_model_dir / dataset_name
#             output_dataset_dir.mkdir(exist_ok=True)

#             normalized_df = pd.DataFrame(columns=['participant_id'] + columns_name)
#             normalized_df['participant_id'] = clinical_df_list[0]['participant_id']
#             normalized_df[columns_name] = test_data_list[dataset_names.index(dataset_name)]
#             normalized_df.to_csv(output_dataset_dir / 'normalized_{}.csv'.format(dataset_name), index=False)

#             reconstruction_df = pd.DataFrame(columns=['participant_id'] + columns_name)
#             reconstruction_df['participant_id'] = clinical_df_list[0]['participant_id']
#             reconstruction_df[columns_name] = test_prediction
#             print('reconstruction_df:', reconstruction_df.shape)
#             print('test_prediction:', test_prediction.shape)
#             reconstruction_df.to_csv(output_dataset_dir / 'reconstruction_{}.csv'.format(dataset_name), index=False)

#             output_data['reconstruction_deviation'] = output_data_reconstruction_deviation_list[dataset_names.index(dataset_name)]

#             error_mean.append(output_data['reconstruction_deviation'])

#             reconstruction_error_df = pd.DataFrame(columns=['participant_id', 'Reconstruction error'])
#             reconstruction_error_df['participant_id'] = clinical_df_list[0]['participant_id']
#             reconstruction_error_df['Reconstruction error'] = output_data['reconstruction_deviation']
#             reconstruction_error_df.to_csv(output_dataset_dir / 'reconstruction_error_{}.csv'.format(dataset_name), index=False)


#         output_data_val = pd.DataFrame(clinical_df_list_val[0].DIA.values, columns=['DIA'])
#         output_data_reconstruction_deviation_list_val = model.reconstruction_deviation_multimodal(val_data_list, val_prediction_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-R', '--dataset_resourse',
                        dest='dataset_resourse',
                        help='Dataset to use for training test and evaluation.',
                        type=str)
    parser.add_argument('-H', '--hz_para_list',
                        dest='hz_para_list',
                        nargs='+',
                        help='List of paras to perform the analysis.',
                        type=int)
    
    parser.add_argument('-C', '--combine',
                        dest='combine',
                        help='how do we combine all modalities.',
                        type=str)
    
    parser.add_argument('-P', '--procedure',
                        dest='procedure',
                        help='Procedure to perform the analysis.',
                        type=str)
    
    parser.add_argument('-K', '--n_splits',
                        dest='n_splits',
                        help='Number of splits for k-fold cross-validation.',
                        type=int, default=5)
    
    args = parser.parse_args()

    if args.hz_para_list is None:
        args.hz_para_list = [110, 110, 10]
    if args.procedure is None:
        args.procedure = 'SE-MoE'
    if args.combine is None:
        args.combine = args.procedure.split('-')[1]
    if args.dataset_resourse is None:
        args.dataset_resourse = 'ADNI'

    # main(args.dataset_resourse, args.procedure, args.n_splits)
    main(args.dataset_resourse, args.combine, args.procedure, args.n_splits)

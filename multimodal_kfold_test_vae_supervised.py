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

import torch

from tqdm import tqdm
import copy
from utils import COLUMNS_HCP, COLUMNS_NAME, load_dataset
from os.path import join, exists
from VAE import VAE_multimodal
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
from utils_vae import plot_losses, MyDataset_labels, Logger, reconstruction_deviation, latent_deviation, separate_latent_deviation, latent_pvalues, reconstruction_deviation_seperate_roi
import argparse
from utils import COLUMNS_HCP, COLUMNS_NAME, load_dataset, COLUMNS_NAME_SNP, COLUMNS_NAME_VBM, COLUMNS_3MODALITIES

PROJECT_ROOT = Path.cwd()

def main(dataset_resourse, procedure, n_splits=5):
    """Make predictions using trained normative models."""
    model_name = 'supervised_vae'

    participants_path = PROJECT_ROOT / 'data' / dataset_resourse / 'y.csv'

    outputs_dir = PROJECT_ROOT / 'outputs'
    kfold_dir = outputs_dir / 'kfold_analysis'
    model_dir = kfold_dir / model_name

    kfold_dir.mkdir(parents=True, exist_ok=True)

    random_seed = 42
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)

    MCK = 5
    dc_output_list = [0] * MCK

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    ids_df = pd.read_csv(participants_path)
    HC_group = ids_df[ids_df['DIA'] == 2]
    other_group = ids_df[ids_df['DIA'] != 2]

    for fold, (train_idx, test_idx) in enumerate(kf.split(HC_group)):
        train_ids = HC_group.iloc[train_idx]['IID']
        test_ids_hc = HC_group.iloc[test_idx]['IID']
        test_ids_other = other_group['IID']
        test_ids = pd.concat([test_ids_hc, test_ids_other])

        train_ids_path = kfold_dir / f'train_ids_{fold:03d}.csv'
        test_ids_path = kfold_dir / f'test_ids_{fold:03d}.csv'

        train_ids.to_csv(train_ids_path, index=False)
        test_ids.to_csv(test_ids_path, index=False)

        if procedure.startswith('SE'):
            dataset_names = ['av45', 'vbm', 'fdg']
        elif procedure.startswith('UCA'):
            dataset_names = ['av45', 'vbm', 'fdg', '3modalities']
        else:
            raise ValueError('Unknown procedure: {}'.format(procedure))

        if procedure.endswith('MoE'):
            combine = 'moe'
        elif procedure.endswith('PoE'):
            combine = 'poe'
        elif procedure.endswith('gPoE'):
            combine = 'gPoE'

        test_data_list = []
        clinical_df_list = []
        modalities = len(dataset_names)

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
                columns_name = [(lambda x: dataset_name + '_' + str(x))(y) for y in list(range(132))]

            freesurfer_path = PROJECT_ROOT / 'data' / dataset_resourse / (dataset_name + '.csv')

            train_dataset_df = load_dataset(participants_path, train_ids_path, freesurfer_path)
            test_dataset_df = load_dataset(participants_path, test_ids_path, freesurfer_path)

            if dataset_resourse == 'ADNI':
                hc_label = 2
            elif dataset_resourse == 'HCP':
                hc_label = 1
            else:
                raise ValueError('Unknown dataset resource')

            train_dataset_df = train_dataset_df.loc[train_dataset_df['DIA'] == hc_label]
            train_data = train_dataset_df[columns_name].values

            test_data = test_dataset_df[columns_name].values

            scaler = RobustScaler()
            train_data = pd.DataFrame(scaler.fit_transform(train_data))

            test_data = pd.DataFrame(scaler.transform(test_data))

            train_covariates = train_dataset_df[['DIA', 'AGE', 'PTGENDER']]
            train_covariates.DIA[train_covariates.DIA == 0] = 0

            test_covariates = test_dataset_df[['DIA', 'AGE', 'PTGENDER']]
            test_covariates.DIA[test_covariates.DIA == 0] = 0
            

            bin_labels = list(range(0, 27))
            AGE_bins_train, bin_edges = pd.qcut(train_covariates['AGE'].rank(method="first"), q=27, retbins=True, labels=bin_labels)
            one_hot_AGE = np.eye(27)[AGE_bins_train.values]

            PTGENDER_bins_train, bin_edges = pd.qcut(train_covariates['PTGENDER'].rank(method="first"), q=2, retbins=True, labels=list(range(0, 2)))
            one_hot_PTGENDER = np.eye(2)[PTGENDER_bins_train.values]

            torch.manual_seed(42)
            use_cuda = torch.cuda.is_available()
            if use_cuda:
                torch.cuda.manual_seed(42)
            DEVICE = torch.device("cuda:1" if use_cuda else "cpu")

            input_dim = train_data.shape[1]
            test_data_list.append(test_data)
            clinical_df_list.append(test_dataset_df)

        one_hot_covariates_test = np.concatenate((one_hot_AGE, one_hot_PTGENDER), axis=1).astype('float32')

        fold_model_dir = model_dir / f'{fold:03d}'
        fold_model_dir.mkdir(exist_ok=True)

        if exists(join(fold_model_dir, 'VAE_model.pkl')):
            print('load trained model')
            model = torch.load(join(fold_model_dir, 'VAE_model.pkl'))
            model.to(DEVICE)
        else:
            print('firstly train model ')

        test_prediction_list = model.pred_recon(test_data_list, combine)
        output_data = pd.DataFrame(clinical_df_list[0].DIA.values, columns=['DIA'])
        output_data_reconstruction_deviation_list = model.reconstruction_deviation_multimodal(test_data_list, test_prediction_list)

        # print('test_data_list[0]:', test_data_list[0])
        # print('test_prediction_list[0]:', test_prediction_list[0])

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
                columns_name = [(lambda x: dataset_name + '_' + str(x))(y) for y in list(range(132))]

            output_dataset_dir = fold_model_dir / dataset_name
            output_dataset_dir.mkdir(exist_ok=True)

            normalized_df = pd.DataFrame(columns=['participant_id'] + columns_name)
            normalized_df['participant_id'] = clinical_df_list[0]['participant_id']
            normalized_df[columns_name] = test_data_list[dataset_names.index(dataset_name)]
            normalized_df.to_csv(output_dataset_dir / 'normalized_{}.csv'.format(dataset_name), index=False)

            reconstruction_df = pd.DataFrame(columns=['participant_id'] + columns_name)
            reconstruction_df['participant_id'] = clinical_df_list[0]['participant_id']
            print('reconstruction_df:', reconstruction_df.shape)
            print('test_prediction:', test_prediction.shape)
            reconstruction_df[columns_name] = test_prediction


            reconstruction_df.to_csv(output_dataset_dir / 'reconstruction_{}.csv'.format(dataset_name), index=False)

            output_data['reconstruction_deviation'] = output_data_reconstruction_deviation_list[dataset_names.index(dataset_name)]
            reconstruction_error_df = pd.DataFrame(columns=['participant_id', 'Reconstruction error'])
            reconstruction_error_df['participant_id'] = clinical_df_list[0]['participant_id']
            reconstruction_error_df['Reconstruction error'] = output_data['reconstruction_deviation']
            reconstruction_error_df.to_csv(output_dataset_dir / 'reconstruction_error_{}.csv'.format(dataset_name), index=False)

        reconstruction_deviation_seperate_roi_df = pd.DataFrame(columns=['participant_id'] + columns_name)
        reconstruction_deviation_seperate_roi_df['participant_id'] = clinical_df_list[0]['participant_id']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-R', '--dataset_resourse',
                        dest='dataset_resourse',
                        help='Dataset to use for training test and evaluation.',
                        type=str)
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

    if args.combine is None:
        args.combine = 'moe'
    if args.procedure is None:
        args.procedure = 'SE-MoE'
    if args.dataset_resourse is None:
        args.dataset_resourse = 'ADNI'

    main(args.dataset_resourse, args.procedure, args.n_splits)

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
from utils import COLUMNS_HCP, COLUMNS_NAME, load_dataset, get_column_name, get_datasets_name, get_hc_label
from os.path import join, exists
# from VAE import VAE
from sklearn.preprocessing import RobustScaler
from utils_vae import plot_losses, MyDataset_labels, Logger, reconstruction_deviation, latent_deviation, separate_latent_deviation, latent_pvalues, reconstruction_deviation_seperate_roi
import argparse
from utils import COLUMNS_HCP, COLUMNS_NAME, load_dataset, COLUMNS_NAME_SNP, COLUMNS_NAME_VBM, COLUMNS_3MODALITIES, COLUMNS_NAME_AAL116
from sklearn.model_selection import KFold

from cVAE import cVAE_multimodal_endtoend, cVAE_multimodal

PROJECT_ROOT = Path.cwd()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    """Make predictions using trained normative models."""
    # ----------------------------------------------------------------------------
    model_name = 'supervised_cvae'

    participants_path = PROJECT_ROOT / 'data' / args.dataset_resourse / 'y.csv'
    # dataset_name to str
    # freesurfer_path = PROJECT_ROOT / 'data' / args.dataset_resourse / (dataset_name + '.csv')

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
    
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=random_seed)

    MCK = 5
    dc_output_list = [0] * MCK
    
    error_mean = []
    # ----------------------------------------------------------------------------
    dataset_names = get_datasets_name(args.dataset_resourse, args.procedure)

    if args.combine is None:
        raise ValueError(f'Unknown procedure: {args.procedure}')

    
    modalities = len(dataset_names)

    ids_df = pd.read_csv(participants_path)
    hc_label = get_hc_label(args.dataset_resourse)
    
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

            columns_name = get_column_name(args.dataset_resourse, dataset_name)

            freesurfer_path = PROJECT_ROOT / 'data' / args.dataset_resourse / (dataset_name + '.csv')
            train_dataset_df = load_dataset(participants_path, train_ids_path, freesurfer_path)
            test_dataset_df = load_dataset(participants_path, test_ids_path, freesurfer_path)

            hc_label = get_hc_label(args.dataset_resourse)

            # train_dataset_df = train_dataset_df.loc[train_dataset_df['DIA'] == hc_label]
            train_data = train_dataset_df[columns_name].values
            scaler = RobustScaler()
            train_data = pd.DataFrame(scaler.fit_transform(train_data))     
            
            train_covariates = train_dataset_df[['DIA','AGE', 'PTGENDER']]
            # train_covariates.DIA[train_covariates.DIA == 0] = 0

            test_data = test_dataset_df[columns_name].values

            # tiv is the total intracranial volume, which is each row's sum of the freesurfer data
            tiv = np.sum(test_data, axis=1)
            tiv = tiv[:, np.newaxis]
            # test_data = np.true_divide(test_data, tiv)

            test_data = pd.DataFrame(scaler.transform(test_data))

            test_covariates = test_dataset_df[['DIA','AGE', 'PTGENDER']]
            # test_covariates.DIA[test_covariates.DIA == 0] = 0

            bin_labels = list(range(0,27))  
            AGE_bins_test, bin_edges = pd.qcut(test_covariates['AGE'].rank(method="first"), q=27, retbins=True, labels=bin_labels)
            one_hot_AGE = np.eye(27)[AGE_bins_test.values]

            gender_bins_test, bin_edges = pd.qcut(test_covariates['PTGENDER'].rank(method="first"), q=2, retbins=True, labels=list(range(0,2)) )
            one_hot_gender = np.eye(2)[gender_bins_test.values]

            test_data_list.append(test_data)
            clinical_df_list.append(test_dataset_df)

        one_hot_covariates_test = np.concatenate((one_hot_AGE, one_hot_gender), axis=1).astype('float32')

        print('fold_model_dir:', fold_model_dir)

        if exists(join(fold_model_dir, 'cVAE_model.pkl')):
            print('load trained model')
            model = torch.load(join(fold_model_dir, 'cVAE_model.pkl'))  
            model.to(DEVICE)
        else:
            print('firstly train model')
            
        test_prediction_list = model.pred_recon(test_data_list, one_hot_covariates_test, DEVICE, args.combine)

        output_data = pd.DataFrame(clinical_df_list[0].DIA.values, columns=['DIA'])

        output_data_reconstruction_deviation_list = model.reconstruction_deviation_multimodal(test_data_list, test_prediction_list)

        # Add this block in the loop after calculating test_prediction and normalized_df
        for dataset_name, test_prediction in zip(dataset_names, test_prediction_list):
            columns_name = get_column_name(args.dataset_resourse, dataset_name)
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

            # New Block: Compute ROI-level reconstruction errors and save to file
            reconstruction_error_roi_df = pd.DataFrame(columns=['participant_id'] + columns_name)
            reconstruction_error_roi_df['participant_id'] = clinical_df_list[0]['participant_id']
            # Store element-wise squared error (MSE values per ROI)
            reconstruction_error_roi_df[columns_name] = (test_data_list[dataset_names.index(dataset_name)] - test_prediction)**2
            # Save to new CSV file
            reconstruction_error_roi_df.to_csv(output_dataset_dir / 'reconstruction_error_roi_{}.csv'.format(dataset_name), index=False)




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

    main(args)

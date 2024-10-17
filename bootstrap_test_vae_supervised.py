#!/home/songlinzhao/anaconda3/envs/normodiff/bin/python3
"""Inference the predictions of the clinical datasets using the supervised model."""
import argparse
from pathlib import Path
from sklearn.preprocessing import RobustScaler, OneHotEncoder
import numpy as np
import pandas as pd
import tensorflow as tf

import torch

from tqdm import tqdm
from utils import COLUMNS_HCP, COLUMNS_NAME, load_dataset, COLUMNS_NAME_SNP, COLUMNS_NAME_VBM, COLUMNS_3MODALITIES
from os.path import join, exists
from VAE import VAE
from utils_vae import plot_losses, Logger, MyDataset_labels, reconstruction_deviation, reconstruction_deviation_seperate_roi

PROJECT_ROOT = Path.cwd()

def main(dataset_name, dataset_resourse):
    """Make predictions using trained normative models."""
    # ----------------------------------------------------------------------------
    n_bootstrap = 10
    model_name = 'supervised_vae'

    participants_path = PROJECT_ROOT / 'data' / dataset_resourse / 'y.csv'
    freesurfer_path = PROJECT_ROOT / 'data' / dataset_resourse / (dataset_name + '.csv')

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

    # ----------------------------------------------------------------------------
    # Create directories structure
    outputs_dir = PROJECT_ROOT / 'outputs'
    bootstrap_dir = outputs_dir / 'bootstrap_analysis'
    model_dir = bootstrap_dir / model_name
    ids_path = outputs_dir / (dataset_name + '_homogeneous_ids.csv')

    #============================================================================
    participants_train = PROJECT_ROOT / 'data' / 'y.csv'
    freesurfer_train = PROJECT_ROOT / 'data' / (dataset_name + '.csv')
    # ----------------------------------------------------------------------------
    ids_dir = bootstrap_dir / 'ids'

    # ----------------------------------------------------------------------------
    # Set random seed
    random_seed = 42
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)

    MCK = 5
    dc_output_list = [0]*MCK
    boostrap_error_mean = []
    # ----------------------------------------------------------------------------
    for i_bootstrap in tqdm(range(n_bootstrap)):
        
        ids_filename = 'cleaned_bootstrap_{:03d}.csv'.format(i_bootstrap)
        ids_train = ids_dir / ids_filename

        bootstrap_train_dir = model_dir / '{:03d}'.format(i_bootstrap)

        dataset_df = load_dataset(participants_train, ids_train, freesurfer_train)

        dataset_df = dataset_df.loc[dataset_df['DIA'] == 2]      
        train_data = dataset_df[columns_name].values
        
        tiv = dataset_df['PTEDUCAT'].values
        tiv = tiv[:, np.newaxis]

        train_data = (np.true_divide(train_data, tiv)).astype('float32')

        scaler = RobustScaler()
        train_data = pd.DataFrame(scaler.fit_transform(train_data))        
        
        #=============================================================================
        bootstrap_model_dir = model_dir / '{:03d}'.format(i_bootstrap)

        output_dataset_dir = bootstrap_model_dir / dataset_name
        output_dataset_dir.mkdir(exist_ok=True)

        test_ids_filename = 'cleaned_bootstrap_test_{:03d}.csv'.format(i_bootstrap)
        ids_path = ids_dir / test_ids_filename
        clinical_df = load_dataset(participants_path, ids_path, freesurfer_path)
        test_data = clinical_df[columns_name].values

        tiv = clinical_df['PTEDUCAT'].values
        tiv = tiv[:, np.newaxis]

        test_data = (np.true_divide(test_data, tiv)).astype('float32')
        
        scaler = RobustScaler()
        test_data = pd.DataFrame(scaler.fit_transform(test_data))

        test_covariates = clinical_df[['DIA', 'PTGENDER', 'AGE']]
        test_covariates.DIA[test_covariates.DIA == 0] = 0

        bin_labels = list(range(0,27))
        AGE_bins_test, bin_edges = pd.qcut(test_covariates['AGE'].rank(method="first"), q=27, retbins=True, labels=bin_labels)
        one_hot_AGE = np.eye(27)[AGE_bins_test.values]

        gender_bins_test, bin_edges = pd.qcut(test_covariates['PTGENDER'].rank(method="first"), q=2, retbins=True, labels=list(range(0,2)) )
        one_hot_gender = np.eye(2)[gender_bins_test.values]

        torch.manual_seed(42)
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.manual_seed(42)
        DEVICE = torch.device("cuda:1" if use_cuda else "cpu")
        input_dim = train_data.shape[1]
        
        if exists(join(bootstrap_train_dir, 'VAE_model.pkl')):
            print('load trained model')
            model = torch.load(join(bootstrap_train_dir, 'VAE_model.pkl'))  
            print(model)
            model.to(DEVICE)
        else:
            print('firstly train model ')
            
        test_latent, test_var = model.pred_latent(test_data, DEVICE)
        test_data_tensor = torch.tensor(test_data.values, dtype=torch.float32, device=DEVICE)
        test_prediction = model.pred_recon(test_data_tensor, DEVICE)
        
        output_data = pd.DataFrame(clinical_df.DIA.values, columns=['DIA'])
        output_data['reconstruction_deviation'] = reconstruction_deviation(test_data.to_numpy(), test_prediction)
        reconstruction_deviation_separate_roi_lists = reconstruction_deviation_seperate_roi(test_data.to_numpy(), test_prediction)
        print(reconstruction_deviation_separate_roi_lists)
        
        normalized_df = pd.DataFrame(columns=['participant_id'] + columns_name)
        normalized_df['participant_id'] = clinical_df['participant_id']
        normalized_df[columns_name] = test_data
        normalized_df.to_csv(output_dataset_dir / 'normalized.csv', index=False)

        reconstruction_df = pd.DataFrame(columns=['participant_id'] + columns_name)
        reconstruction_df['participant_id'] = clinical_df['participant_id']
        reconstruction_df[columns_name] = test_prediction
        reconstruction_df.to_csv(output_dataset_dir / 'reconstruction.csv', index=False)

        encoded_df = pd.DataFrame(columns=['participant_id'] + list(range(test_latent.shape[1])))
        encoded_df['participant_id'] = clinical_df['participant_id']
        encoded_df[list(range(test_latent.shape[1]))] = test_latent
        encoded_df.to_csv(output_dataset_dir / 'encoded.csv', index=False)

        boostrap_error_mean.append(output_data['reconstruction_deviation'])
        
        reconstruction_error_df = pd.DataFrame(columns=['participant_id', 'Reconstruction error'])
        reconstruction_error_df['participant_id'] = clinical_df['participant_id']
        reconstruction_error_df['Reconstruction error'] = output_data['reconstruction_deviation']
        reconstruction_error_df.to_csv(output_dataset_dir / 'reconstruction_error.csv', index=False)
    
    boostrap_error_mean = np.array(boostrap_error_mean)
    boostrap_mean = np.mean(boostrap_error_mean)
    bootsrao_var = np.std(boostrap_error_mean)
    boostrap_list = np.array([boostrap_mean, bootsrao_var])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--dataset_name',
                        dest='dataset_name',
                        help='Dataset name to calculate deviations.')
    
    parser.add_argument('-R', '--dataset_resourse',
                        dest='dataset_resourse',
                        help='Dataset resourse to calculate deviations.')
    
    args = parser.parse_args()

    if args.dataset_name is None:
        args.dataset_name = 'av45'
    if args.dataset_resourse is None:
        args.dataset_resourse = 'ADNI'

    main(args.dataset_name, args.dataset_resourse)

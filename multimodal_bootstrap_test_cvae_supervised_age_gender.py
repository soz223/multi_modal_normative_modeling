#!/home/songlinzhao/anaconda3/envs/normodiff/bin/python3
"""Inference the predictions of the clinical datasets using the supervised model."""
import argparse
from pathlib import Path
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

import torch

from tqdm import tqdm
import copy
from utils import COLUMNS_HCP, COLUMNS_NAME, load_dataset, COLUMNS_NAME_SNP , COLUMNS_NAME_VBM
from os.path import join, exists
from VAE import VAE
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from utils_vae import plot_losses, MyDataset_labels, Logger, reconstruction_deviation, latent_deviation, separate_latent_deviation, latent_pvalues


PROJECT_ROOT = Path.cwd()

def main(dataset_name, comb_label):
    """Make predictions using trained normative models."""
    # ----------------------------------------------------------------------------
    n_bootstrap = 10
    model_name = 'supervised_cvae_age_gender'

    participants_path = PROJECT_ROOT / 'data' / 'y.csv'
    freesurfer_path = PROJECT_ROOT / 'data' / (dataset_name + '.csv')


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
   # bootstrap_dir = PROJECT_ROOT / 'outputs' / 'bootstrap_analysis'
    ids_dir = bootstrap_dir / 'ids'

    #model_dir = bootstrap_dir / model_name
    #model_dir.mkdir(exist_ok=True)
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
        dataset_name = dataset_names = ['av45', 'fdg', 'vbm']
        test_data_list = []
        clinical_df_list = []
        modalities = len(dataset_names)
        for dataset_name in dataset_names:
            if dataset_name == 'snp':
                columns_name = COLUMNS_NAME_SNP
            elif dataset_name == 'vbm':
                columns_name = COLUMNS_NAME_VBM
            else:
                columns_name = COLUMNS_NAME
            freesurfer_path = PROJECT_ROOT / 'data' / (dataset_name + '.csv')
            freesurfer_train = PROJECT_ROOT / 'data' / (dataset_name + '.csv')

            ids_filename = 'cleaned_bootstrap_{:03d}.csv'.format(i_bootstrap)
            ids_train = ids_dir / ids_filename

            bootstrap_train_dir = model_dir / '{:03d}'.format(i_bootstrap)

            # ----------------------------------------------------------------------------
            # Loading data
            dataset_df = load_dataset(participants_train, ids_train, freesurfer_train)

            # ----------------------------------------------------------------------------

            dataset_df = dataset_df.loc[dataset_df['DIA'] == 2]      
            train_data = dataset_df[columns_name].values
            

            tiv = dataset_df['PTEDUCAT'].values
            tiv = tiv[:, np.newaxis]

            train_data = (np.true_divide(train_data, tiv)).astype('float32')

            scaler = RobustScaler()
            train_data = pd.DataFrame(scaler.fit_transform(train_data))  
            
            train_covariates = dataset_df[['DIA','AGE','PTGENDER']]
            train_covariates.DIA[train_covariates.DIA == 0] = 0       #
            #train_covariates['ICV'] =tiv  #        
            #=============================================================================
            bootstrap_model_dir = model_dir / '{:03d}'.format(i_bootstrap)

            output_dataset_dir = bootstrap_model_dir / dataset_name
            output_dataset_dir.mkdir(exist_ok=True)

            # ----------------------------------------------------------------------------
            # Loading data
            test_ids_filename = 'cleaned_bootstrap_test_{:03d}.csv'.format(i_bootstrap)
            ids_path = ids_dir / test_ids_filename
            clinical_df = load_dataset(participants_path, ids_path, freesurfer_path)
            #print(columns_name)
            test_data = clinical_df[columns_name].values

            tiv = clinical_df['PTEDUCAT'].values
            tiv = tiv[:, np.newaxis]

            test_data = (np.true_divide(test_data, tiv)).astype('float32')
            
            scaler = RobustScaler()
            test_data = pd.DataFrame(scaler.fit_transform(test_data))  

            test_covariates = clinical_df[['DIA','AGE','PTGENDER']]
            test_covariates.DIA[test_covariates.DIA == 0] = 0       #
            test_covariates['ICV'] =tiv  #   
            
            bin_labels = list(range(0,10))  
            age_bins_test, bin_edges = pd.cut(test_covariates['AGE'], 10, retbins=True, labels=bin_labels)


            age_bins_test.fillna(0,inplace = True)
            one_hot_age_test = np.eye(10)[age_bins_test.values]

            
            bin_labels3 = list(range(0,5))
            ICV_bins_test, bin_edges = pd.qcut(test_covariates['ICV'], q=5,  retbins=True, labels=bin_labels3, duplicates='drop')

            ICV_bins_test.fillna(0, inplace = True)
            one_hot_ICV_test = np.eye(10)[ICV_bins_test.values]
            
            gender = test_covariates['PTGENDER'].values[:, np.newaxis].astype('float32')
            enc_gender = OneHotEncoder(sparse=False)
            one_hot_gender_test = enc_gender.fit_transform(gender)
            batch_size = 256
            
            torch.manual_seed(42)
            use_cuda = torch.cuda.is_available()
            if use_cuda:
                torch.cuda.manual_seed(42)
            DEVICE = torch.device("cuda:1" if use_cuda else "cpu")
        

            input_dim = train_data.shape[1]

            test_data_list.append(test_data)
            clinical_df_list.append(clinical_df)

        


        if comb_label == 1:
            one_hot_covariates_test = np.concatenate((one_hot_age_test, one_hot_gender_test), axis=1).astype('float32')
        elif comb_label == 2:
            one_hot_covariates_test = np.concatenate((one_hot_age_test, one_hot_ICV_test), axis=1).astype('float32')
        elif comb_label == 3:
            one_hot_covariates_test = np.concatenate((one_hot_gender_test,one_hot_ICV_test), axis=1).astype('float32')
        else:
            one_hot_covariates_test = np.concatenate((one_hot_age_test, one_hot_gender_test,one_hot_ICV_test), axis=1).astype('float32')    



        



        # ----------------------------------------------------------------------------
        # Load trained model and do the reconstruction
        if exists(join(bootstrap_train_dir, 'cVAE_model.pkl')):
            print('load trained model')
            model = torch.load(join(bootstrap_train_dir, 'cVAE_model.pkl'))  
            # print(model)
            model.to(DEVICE)
        else:
            print('firstly train model ')
            
        
        
            
        # test_latent, test_var = model.pred_latent(test_data, one_hot_covariates_test, DEVICE)
        test_prediction_list = model.pred_recon(test_data_list, one_hot_covariates_test, DEVICE)
       
        output_data = pd.DataFrame(clinical_df.DIA.values, columns=['DIA'])
        output_data['reconstruction_deviation'] = model.reconstruction_deviation_multimodal(test_data_list, test_prediction_list)


        
        # normalized_df = pd.DataFrame(columns=['participant_id'] + columns_name)
        # normalized_df['participant_id'] = clinical_df['participant_id']
        # normalized_df[columns_name] = test_data
        # normalized_df.to_csv(output_dataset_dir / 'normalized.csv', index=False)

        


        for dataset_name, test_prediction in zip(dataset_names, test_prediction_list):
        # reconstruction_df = pd.DataFrame(columns=['participant_id'] + columns_name)
        # reconstruction_df['participant_id'] = clinical_df['participant_id']
        # reconstruction_df[columns_name] = test_prediction
        # reconstruction_df.to_csv(output_dataset_dir / 'reconstruction.csv', index=False)
            if dataset_name == 'snp':
                columns_name = COLUMNS_NAME_SNP
            elif dataset_name == 'vbm':
                columns_name = COLUMNS_NAME_VBM
            else:
                columns_name = COLUMNS_NAME

            normalization_df = pd.DataFrame(columns=['participant_id'] + columns_name)
            idx = dataset_names.index(dataset_name)
            normalization_df['participant_id'] = clinical_df_list[idx]['participant_id']
            normalization_df[columns_name] = test_data_list[idx]
            normalization_df.to_csv(output_dataset_dir / 'normalized_{}.csv'.format(dataset_name), index=False)


            
            
            reconstruction_df = pd.DataFrame(columns=['participant_id'] + columns_name)
            reconstruction_df['participant_id'] = clinical_df['participant_id']
            # get the idx of the loop
            
            reconstruction_df[columns_name] = test_prediction_list[idx]
            reconstruction_df.to_csv(output_dataset_dir / 'reconstruction_{}.csv'.format(dataset_name), index=False)



        # encoded_df = pd.DataFrame(columns=['participant_id'] + list(range(test_latent.shape[1])))
        
        # encoded_df['participant_id'] = clinical_df['participant_id']
        # encoded_df[list(range(test_latent.shape[1]))] = test_latent
        # encoded_df.to_csv(output_dataset_dir / 'encoded.csv', index=False)


        boostrap_error_mean.append(output_data['reconstruction_deviation'])

        

        reconstruction_error_df = pd.DataFrame(columns=['participant_id', 'Reconstruction error'])
        reconstruction_error_df['participant_id'] = clinical_df['participant_id']
        reconstruction_error_df['Reconstruction error'] = output_data['reconstruction_deviation']
        # reconstruction_error_df.to_csv(output_dataset_dir / 'reconstruction_error.csv', index=False)
        output_dataset_dir = bootstrap_model_dir / 'multimodal'
        output_dataset_dir.mkdir(exist_ok=True)
        reconstruction_error_df.to_csv(output_dataset_dir / 'reconstruction_error.csv', index=False)
    
    
    # boostrap_error_mean = np.array(boostrap_error_mean)
    # print('boostrap_error_mean:', boostrap_error_mean)
    # boostrap_mean = np.mean(boostrap_error_mean)
    # bootsrao_var = np.std(boostrap_error_mean)
    # boostrap_list = np.array([boostrap_mean, bootsrao_var])
    # np.savetxt("cvae_age_gender_boostrap_mean_std.csv", boostrap_list, delimiter=",")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--dataset_name',
                        dest='dataset_name',
                        help='Dataset name to calculate deviations.')
    parser.add_argument('-L', '--comb_label',
                        dest='comb_label',
                        help='Combination label to perform group analysis.',
                        type=int)
    args = parser.parse_args()


    main(args.dataset_name, args.comb_label)
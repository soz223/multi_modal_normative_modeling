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
from VAE import VAE
from sklearn.preprocessing import RobustScaler
from utils_vae import plot_losses, MyDataset_labels, Logger, reconstruction_deviation, latent_deviation, separate_latent_deviation, latent_pvalues, reconstruction_deviation_seperate_roi
import argparse
from utils import COLUMNS_HCP, COLUMNS_NAME, load_dataset, COLUMNS_NAME_SNP, COLUMNS_NAME_VBM, COLUMNS_3MODALITIES

PROJECT_ROOT = Path.cwd()


def main(dataset_resourse, procedure):
    """Make predictions using trained normative models."""
    # ----------------------------------------------------------------------------
    n_bootstrap = 30
    model_name = 'supervised_cvae'

    participants_path = PROJECT_ROOT / 'data' / dataset_resourse / 'y.csv'
    # dataset_name to str
    # freesurfer_path = PROJECT_ROOT / 'data' / dataset_resourse / (dataset_name + '.csv')

    # ----------------------------------------------------------------------------
    # Create directories structure
    outputs_dir = PROJECT_ROOT / 'outputs'
    bootstrap_dir = outputs_dir / 'bootstrap_analysis'
    model_dir = bootstrap_dir / model_name
    # ids_path = outputs_dir / (dataset_name + '_homogeneous_ids.csv')

    #============================================================================

    participants_train = PROJECT_ROOT / 'data' / dataset_resourse / 'y.csv'
    # freesurfer_train = PROJECT_ROOT / 'data' / dataset_resourse / (dataset_name + '.csv')

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


        # if procedure starts with 'SE':
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
                columns_name = [(lambda x: dataset_name + '_'+ str(x))(y) for y in list(range(132))]

            freesurfer_path = PROJECT_ROOT / 'data' / dataset_resourse / (dataset_name + '.csv')
            freesurfer_train = PROJECT_ROOT / 'data' / dataset_resourse / (dataset_name + '.csv')


            
            ids_filename = 'cleaned_bootstrap_{:03d}.csv'.format(i_bootstrap)
            ids_train = ids_dir / ids_filename

            bootstrap_train_dir = model_dir / '{:03d}'.format(i_bootstrap)
            #bootstrap_train_dir.mkdir(exist_ok=True)

            # ----------------------------------------------------------------------------
            # Loading data
            dataset_df = load_dataset(participants_train, ids_train, freesurfer_train)

            # ----------------------------------------------------------------------------
            if dataset_resourse == 'ADNI':
                hc_label = 2
            elif dataset_resourse == 'HCP':
                hc_label = 1
            else:
                raise ValueError('Unknown dataset resource')

            dataset_df = dataset_df.loc[dataset_df['DIA'] == hc_label]      
            train_data = dataset_df[columns_name].values
            

            # tiv = dataset_df['PTEDUCAT'].values
            # tiv = tiv[:, np.newaxis]

            # train_data = (np.true_divide(train_data, tiv)).astype('float32')

            scaler = RobustScaler()
            train_data = pd.DataFrame(scaler.fit_transform(train_data))     
            
            train_covariates = dataset_df[['DIA','AGE', 'PTGENDER']]
            train_covariates.DIA[train_covariates.DIA == 0] = 0       #
            # train_covariates['ICV'] =tiv  #        
            #=============================================================================
            bootstrap_model_dir = model_dir / '{:03d}'.format(i_bootstrap)

            output_dataset_dir = bootstrap_model_dir / dataset_name
            output_dataset_dir.mkdir(exist_ok=True)

            # ----------------------------------------------------------------------------
            # Loading data
            ids_dir = bootstrap_dir / 'ids'
            test_ids_filename = 'cleaned_bootstrap_test_{:03d}.csv'.format(i_bootstrap)
            ids_path = ids_dir / test_ids_filename
            clinical_df = load_dataset(participants_path, ids_path, freesurfer_path)
            #print(COLUMNS_NAME)
            test_data = clinical_df[columns_name].values

            # tiv = clinical_df['PTEDUCAT'].values
            # tiv = tiv[:, np.newaxis]

            # test_data = (np.true_divide(test_data, tiv)).astype('float32')
            
            scaler = RobustScaler()
            test_data = pd.DataFrame(scaler.fit_transform(test_data))  

            test_covariates = clinical_df[['DIA','PTGENDER', 'AGE']]
            # test_covariates = clinical_df[['DIA', 'AGE']]
            test_covariates.DIA[test_covariates.DIA == 0] = 0       #
            # test_covariates['ICV'] =tiv  #   
            
            bin_labels = list(range(0,27))  
            AGE_bins_test, bin_edges = pd.qcut(test_covariates['AGE'].rank(method="first"), q=27, retbins=True, labels=bin_labels)
            #AGE_bins_test = pd.cut(test_covariates['AGE'], bins=bin_edges, labels=bin_labels)
            one_hot_AGE = np.eye(27)[AGE_bins_test.values]
            #one_hot_AGE_train = np.eye(10)[AGE_bins_train.values]
            
            gender_bins_test, bin_edges = pd.qcut(test_covariates['PTGENDER'].rank(method="first"), q=2, retbins=True, labels=list(range(0,2)) )
            
            one_hot_gender = np.eye(2)[gender_bins_test.values]
            
            
            # bin_labels = list(range(0,3))  
            
            # ICV_bins_test, bin_edges = pd.qcut(test_covariates['ICV'], q=3,  retbins=True, labels=bin_labels, duplicates='drop' )
            # ICV_bins_test.fillna(0, inplace = True)
            # one_hot_ICV_test = np.eye(10)[ICV_bins_test.values]

            batch_size = 256
            
            torch.manual_seed(42)
            use_cuda = torch.cuda.is_available()
            if use_cuda:
                torch.cuda.manual_seed(42)
            DEVICE = torch.device("cuda:1" if use_cuda else "cpu")

            input_dim = train_data.shape[1]
            test_data_list.append(test_data)
            clinical_df_list.append(clinical_df)

        one_hot_covariates_test = np.concatenate((one_hot_AGE, one_hot_gender), axis=1).astype('float32')


        if exists(join(bootstrap_train_dir, 'cVAE_model.pkl')):
            print('load trained model')
            model = torch.load(join(bootstrap_train_dir, 'cVAE_model.pkl'))  
            model.to(DEVICE)
        else:
            print('firstly train model ')
            
       
        # test_latent, test_var = model.pred_latent(test_data, one_hot_covariates_test, DEVICE)
        test_prediction_list = model.pred_recon(test_data_list, one_hot_covariates_test, DEVICE, combine=combine)

        # test_prediction = model.pred_recon(test_data, one_hot_covariates_test, DEVICE)
        
        output_data = pd.DataFrame(clinical_df.DIA.values, columns=['DIA'])
        
        # # output_data['reconstruction_deviation'] = reconstruction_deviation(test_data.to_numpy(), test_prediction)
        # output_data['reconstruction_deviation'] = model.reconstruction_deviation_multimodal(test_data_list, test_prediction_list)

        output_data_reconstruction_deviation_list = model.reconstruction_deviation_multimodal(test_data_list, test_prediction_list)

        print(f"Length of test_data_list: {len(test_data_list)}")
        print(f"Length of test_prediction_list: {len(test_prediction_list)}")
        # reconstruction_deviation = model.reconstruction_deviation_multimodal(test_data_list, test_prediction_list)
        # print(f"Length of reconstruction_deviation: {len(reconstruction_deviation)}")
        print(f"Length of output_data.index: {len(output_data.index)}")

        
        # output_data = pd.DataFrame(clinical_df.DIA.values, columns=['DIA'])
        
        # # output_data['reconstruction_deviation'] = reconstruction_deviation(test_data.to_numpy(), test_prediction)
        # output_data['reconstruction_deviation'] = model.reconstruction_deviation_multimodal(test_data_list, test_prediction_list)

        

        # reconstruction_deviation_separate_roi_lists = reconstruction_deviation_seperate_roi(test_data.to_numpy(), test_prediction)
        # print(reconstruction_deviation_separate_roi_lists)
     
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

            #=============================================================================
            bootstrap_model_dir = model_dir / '{:03d}'.format(i_bootstrap)

            output_dataset_dir = bootstrap_model_dir / dataset_name
            output_dataset_dir.mkdir(exist_ok=True)

            normalized_df = pd.DataFrame(columns=['participant_id'] + columns_name)
            normalized_df['participant_id'] = clinical_df['participant_id']
            normalized_df[columns_name] = test_data_list[dataset_names.index(dataset_name)]
            # normalized_df.to_csv(output_dataset_dir / 'normalized.csv', index=False)
            normalized_df.to_csv(output_dataset_dir / 'normalized_{}.csv'.format(dataset_name), index=False)

            # print('normalized data saved at {}'.format(output_dataset_dir / 'normalized_{}.csv'.format(dataset_name)))

            reconstruction_df = pd.DataFrame(columns=['participant_id'] + columns_name)
            reconstruction_df['participant_id'] = clinical_df['participant_id']
            reconstruction_df[columns_name] = test_prediction
            # reconstruction_df.to_csv(output_dataset_dir / 'reconstruction.csv', index=False)
            reconstruction_df.to_csv(output_dataset_dir / 'reconstruction_{}.csv'.format(dataset_name), index=False)

        # encoded_df = pd.DataFrame(columns=['participant_id'] + list(range(test_latent.shape[1])))
        
        # encoded_df['participant_id'] = clinical_df['participant_id']
        # encoded_df[list(range(test_latent.shape[1]))] = test_latent
        # encoded_df.to_csv(output_dataset_dir / 'encoded.csv', index=False)

            output_data['reconstruction_deviation'] = output_data_reconstruction_deviation_list[dataset_names.index(dataset_name)]

            boostrap_error_mean.append(output_data['reconstruction_deviation'])

            reconstruction_error_df = pd.DataFrame(columns=['participant_id', 'Reconstruction error'])
            reconstruction_error_df['participant_id'] = clinical_df['participant_id']
            reconstruction_error_df['Reconstruction error'] = output_data['reconstruction_deviation']
            # reconstruction_error_df.to_csv(output_dataset_dir / 'reconstruction_error.csv', index=False)
            reconstruction_error_df.to_csv(output_dataset_dir / 'reconstruction_error_{}.csv'.format(dataset_name), index=False)

        

        reconstruction_deviation_seperate_roi_df = pd.DataFrame(columns=['participant_id'] + columns_name)
        reconstruction_deviation_seperate_roi_df['participant_id'] = clinical_df['participant_id']
        # reconstruction_deviation_seperate_roi_df[columns_name] = reconstruction_deviation_separate_roi_lists

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
    
    args = parser.parse_args()

    if args.hz_para_list is None:
        args.hz_para_list = [110, 110, 10]
    if args.combine is None:
        args.combine = 'moe'
    if args.procedure is None:
        args.procedure = 'SE-MoE'
    if args.dataset_resourse is None:
        args.dataset_resourse = 'ADNI'


    main(args.dataset_resourse, args.procedure)
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
from utils import COLUMNS_HCP, COLUMNS_NAME, load_dataset, COLUMNS_NAME_SNP , COLUMNS_NAME_VBM, COLUMNS_3MODALITIES
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
        train_data = dataset_df[COLUMNS_NAME].values
        

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
        #print(COLUMNS_NAME)
        test_data = clinical_df[COLUMNS_NAME].values

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
        #age_bins_train, bin_edges = pd.cut(train_covariates['AGE'], 10, retbins=True, labels=bin_labels)
        #age_bins_test = pd.cut(test_covariates['AGE'], retbins=True, labels=bin_labels)
        #age_bins_train.fillna(0, inplace=True)
        age_bins_test.fillna(0,inplace = True)
        one_hot_age_test = np.eye(10)[age_bins_test.values]
        #one_hot_age_train = np.eye(10)[age_bins_train.values]

        #bin_labels2 = list(range(0,2))
        #gender_bins_train, bin_edges2 = pd.cut(train_covariates['PTGENDER'], 2,  retbins=True, labels=bin_labels2)
        #gender_bins_test = pd.cut(test_covariates['PTGENDER'], bins=bin_edges2, labels=bin_labels2)
        #gender_bins_train.fillna(0, inplace=True)
        #gender_bins_test.fillna(0,inplace = True)        
        #one_hot_gender_test = np.eye(2)[gender_bins_test.values]
        #one_hot_gender_train = np.eye(2)[gender_bins_train.values]
        
        bin_labels3 = list(range(0,5))
        ICV_bins_test, bin_edges = pd.qcut(test_covariates['ICV'], q=5,  retbins=True, labels=bin_labels3, duplicates='drop')
        #ICV_bins_test = pd.cut(test_covariates['ICV'], bins=bin_edges, labels=bin_labels)
        #one_hot_ICV_test = np.eye(10)[ICV_bins_test.values]
        ICV_bins_test.fillna(0, inplace = True)
        one_hot_ICV_test = np.eye(10)[ICV_bins_test.values]
        
        gender = test_covariates['PTGENDER'].values[:, np.newaxis].astype('float32')
        enc_gender = OneHotEncoder(sparse=False)
        one_hot_gender_test = enc_gender.fit_transform(gender)
        
        #age = test_covariates['AGE'].values[:, np.newaxis].astype('float32')
        #enc_age = OneHotEncoder(sparse=False)
        #enc_age = OneHotEncoder(handle_unknown = "ignore",sparse=False)
        #one_hot_age_test = enc_age.fit_transform(age)
        
        #age_train = train_covariates['AGE'].values[:, np.newaxis].astype('float32')
        #age_test = test_covariates['AGE'].values[:, np.newaxis].astype('float32')
        #enc_age = OneHotEncoder(sparse=False)
        #enc_age = OneHotEncoder(handle_unknown = "ignore",sparse=False)
        #one_hot_age = enc_age.fit_transform(age)

        #gender = dataset_df['PTGENDER'].values[:, np.newaxis].astype('float32')
        #enc_gender = OneHotEncoder(sparse=False)
        #one_hot_gender = enc_gender.fit_transform(gender)

        #y_data = np.concatenate((one_hot_age, one_hot_gender), axis=1).astype('float32')
        # ----------------------------------------------------------------------------
        #encoder = keras.models.load_model(bootstrap_model_dir / 'encoder.h5', compile=False)
        #decoder = keras.models.load_model(bootstrap_model_dir / 'decoder.h5', compile=False)

        #scaler = joblib.load(bootstrap_model_dir / 'scaler.joblib')

        #enc_age = joblib.load(bootstrap_model_dir / 'age_encoder.joblib')
        #enc_gender = joblib.load(bootstrap_model_dir / 'gender_encoder.joblib')
        
        #age = clinical_df['AGE'].values[:, np.newaxis].astype('float32')
        #print(age, age.shape)
        #one_hot_age = enc_age.transform(age)
        #print(one_hot_age, one_hot_age.shape)
        
        
        #gender = clinical_df['PTGENDER'].values[:, np.newaxis].astype('float32')
        #one_hot_gender = enc_gender.transform(gender)
        

            
        
        #y_data = np.concatenate((one_hot_age, one_hot_gender), axis=1).astype('float32')
        
        #extended = []
        #for i in range(y_data.shape[1]):
        #    val = "d" + str(i)
        #    extended.append(val)
        # ----------------------------------------------------------------------------
        #x_normalized = scaler.transform(x_dataset)
        #x = np.concatenate((x_normalized, y_data), axis=1)
        #COLUMNS_NAME.append(['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16',
        #                     '17','18','19','20','21','22','23','24','25','26','27'])
        #tmp = copy.copy(COLUMNS_NAME)
        #tmp.extend(extended)
        #print(COLUMNS_NAME)
        #print("aaaaaaaaaaaaaa")
        batch_size = 256
         
        torch.manual_seed(42)
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.manual_seed(42)
        DEVICE = torch.device("cuda:1" if use_cuda else "cpu")
     

        input_dim = train_data.shape[1]
        #one_hot_covariates_train = np.append(one_hot_age_train, one_hot_ICV_train, axis=1)
        #c_dim = one_hot_covariates_train.shape[1]
        
        
        if comb_label == 1:
            one_hot_covariates_test = np.concatenate((one_hot_age_test, one_hot_gender_test), axis=1).astype('float32')
        elif comb_label == 2:
            one_hot_covariates_test = np.concatenate((one_hot_age_test, one_hot_ICV_test), axis=1).astype('float32')
        elif comb_label == 3:
            one_hot_covariates_test = np.concatenate((one_hot_gender_test,one_hot_ICV_test), axis=1).astype('float32')
        else:
            one_hot_covariates_test = np.concatenate((one_hot_age_test, one_hot_gender_test,one_hot_ICV_test), axis=1).astype('float32')    
        
        
        #one_hot_covariates_test = np.concatenate((one_hot_age_test,one_hot_gender_test,  one_hot_ICV_test), axis=1).astype('float32')
        #train_dataset = MyDataset_labels(train_data.to_numpy(), one_hot_covariates_train)    
        #train_dataset = MyDataset(train_data)
        #generator_train = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=False)        
        
        if exists(join(bootstrap_train_dir, 'cVAE_model.pkl')):
            print('load trained model')
            model = torch.load(join(bootstrap_train_dir, 'cVAE_model.pkl'))  
            # print(model)
            model.to(DEVICE)
        else:
            print('firstly train model ')
            
        
        
            
        test_latent, test_var = model.pred_latent(test_data, one_hot_covariates_test, DEVICE)
        #train_latent, _ = model.pred_latent(train_data, one_hot_covariates_train, DEVICE)
        test_prediction = model.pred_recon(test_data, one_hot_covariates_test, DEVICE)
        
        #batch = tf.shape(test_latent)[0]
        #dim = tf.shape(test_latent)[1]
        #for mck in range(MCK):

        #    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        #    z = test_latent + tf.exp(0.5 * test_var) * epsilon
        #    dc_output_list[mck] = model.pred_recon_tensor(z,one_hot_covariates_test,test_latent, test_var, DEVICE)
        #decoder_output = sum(dc_output_list)/len(dc_output_list)
        #decoder_output_numpy = np.array(decoder_output)
        #mean_list = decoder_output_numpy
        #mean_list[i_bootstrap][]
        #variance = sum((dc_output_list - decoder_output)**2)/MCK
        #variance_numpy = np.array(variance)
        #variance_list = variance_numpy
        
        #reconstruction_error = np.mean(np.divide((abs(test_data - mean_list)), np.sqrt(variance_list**2 + c**2)),axis=1)
    
        output_data = pd.DataFrame(clinical_df.DIA.values, columns=['DIA'])
        output_data['reconstruction_deviation'] = reconstruction_deviation(test_data.to_numpy(), test_prediction)
        #output_data['latent_deviation'] = latent_deviation(train_latent, test_latent, test_var)
        #deviation = separate_latent_deviation(train_latent, test_latent, test_var)
        
        
        normalized_df = pd.DataFrame(columns=['participant_id'] + COLUMNS_NAME)
        normalized_df['participant_id'] = clinical_df['participant_id']
        normalized_df[COLUMNS_NAME] = test_data
        normalized_df.to_csv(output_dataset_dir / 'normalized.csv', index=False)

        # ----------------------------------------------------------------------------
        #age = clinical_df['AGE'].values[:, np.newaxis].astype('float32')
        #one_hot_age = enc_age.transform(age)

        #gender = clinical_df['PTGENDER'].values[:, np.newaxis].astype('float32')
        #one_hot_gender = enc_gender.transform(gender)

        #y_data = np.concatenate((one_hot_age, one_hot_gender), axis=1).astype('float32')

        # ----------------------------------------------------------------------------
        #x = np.concatenate((x_normalized, y_data), axis=1)
        #encoded = encoder(x_normalized, training=False)
        #encoded = encoder(x, training=False)
        #reconstruction = decoder(tf.concat([encoded,y_data], axis=1), training=False)
        #reconstruction = decoder(tf.concat([encoded], axis=1), training=False)

        reconstruction_df = pd.DataFrame(columns=['participant_id'] + COLUMNS_NAME)
        reconstruction_df['participant_id'] = clinical_df['participant_id']
        reconstruction_df[COLUMNS_NAME] = test_prediction
        reconstruction_df.to_csv(output_dataset_dir / 'reconstruction.csv', index=False)

        # print('output_dataset_dir: ', output_dataset_dir)
        # print('reconstruction error: ', output_data['reconstruction_deviation'].mean())

        encoded_df = pd.DataFrame(columns=['participant_id'] + list(range(test_latent.shape[1])))
        
        encoded_df['participant_id'] = clinical_df['participant_id']
        encoded_df[list(range(test_latent.shape[1]))] = test_latent
        encoded_df.to_csv(output_dataset_dir / 'encoded.csv', index=False)

        # ----------------------------------------------------------------------------
        #reconstruction_error = np.mean((x - reconstruction) ** 2, axis=1)
        #reconstruction_error = np.mean((x_normalized - reconstruction) ** 2, axis=1)
        boostrap_error_mean.append(output_data['reconstruction_deviation'])

        reconstruction_error_df = pd.DataFrame(columns=['participant_id', 'Reconstruction error'])
        reconstruction_error_df['participant_id'] = clinical_df['participant_id']
        reconstruction_error_df['Reconstruction error'] = output_data['reconstruction_deviation']
        reconstruction_error_df.to_csv(output_dataset_dir / 'reconstruction_error.csv', index=False)
    
    boostrap_error_mean = np.array(boostrap_error_mean)
    print('boostrap_error_mean:', boostrap_error_mean)
    boostrap_mean = np.mean(boostrap_error_mean)
    bootsrao_var = np.std(boostrap_error_mean)
    boostrap_list = np.array([boostrap_mean, bootsrao_var])
    np.savetxt("cvae_age_gender_boostrap_mean_std.csv", boostrap_list, delimiter=",")


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
    if args.dataset_name == 'snp':
        COLUMNS_NAME = COLUMNS_NAME_SNP
    elif args.dataset_name == 'vbm':
        COLUMNS_NAME = COLUMNS_NAME_VBM
    elif args.dataset_name == '3modalities':
        COLUMNS_NAME = COLUMNS_3MODALITIES

    main(args.dataset_name, args.comb_label)
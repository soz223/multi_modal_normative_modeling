#!/home/songlinzhao/anaconda3/envs/normodiff/bin/python3
"""Inference the predictions of the clinical datasets using the supervised model."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import copy
from utils import COLUMNS_HCP, COLUMNS_NAME, load_dataset, COLUMNS_NAME_SNP, COLUMNS_NAME_VBM, COLUMNS_3MODALITIES

PROJECT_ROOT = Path.cwd()


def main(dataset_name):
    """Make predictions using trained normative models."""
    # ----------------------------------------------------------------------------
    n_bootstrap = 10
    model_name = 'supervised_ae'

    # participants_path = PROJECT_ROOT / 'data' / 'HCP' / '[02]-imaging' / 'AV45' / 'AV45_cov.csv'
    # freesurfer_path = PROJECT_ROOT / 'data' / 'HCP' / '[02]-imaging' / 'AV45' / 'AV45_pheno_AAL.csv'
    columns_name = [(lambda x: dataset_name + '_'+ str(x))(y) for y in list(range(132))]
    participants_path = PROJECT_ROOT / 'data' / 'HCP' / 'y.csv'
    freesurfer_path = PROJECT_ROOT / 'data' / 'HCP' / (dataset_name + '.csv')


    # ----------------------------------------------------------------------------
    # Create directories structure
    outputs_dir = PROJECT_ROOT / 'outputs'
    bootstrap_dir = outputs_dir / 'bootstrap_analysis'
    model_dir = bootstrap_dir / model_name
    ids_path = outputs_dir / (dataset_name + '_homogeneous_ids.csv')

    # ----------------------------------------------------------------------------
    # Set random seed
    random_seed = 42
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)
    
    boostrap_error_mean = []
    # ----------------------------------------------------------------------------
    for i_bootstrap in tqdm(range(n_bootstrap)):
        bootstrap_model_dir = model_dir / '{:03d}'.format(i_bootstrap)

        output_dataset_dir = bootstrap_model_dir / dataset_name
        output_dataset_dir.mkdir(exist_ok=True)

        # ----------------------------------------------------------------------------
        # Loading data
        ids_dir = bootstrap_dir / 'ids'
        test_ids_filename = 'cleaned_bootstrap_test_{:03d}.csv'.format(i_bootstrap)
        ids_path = ids_dir / test_ids_filename
        clinical_df = load_dataset(participants_path, ids_path, freesurfer_path)
        
        x_dataset = clinical_df[columns_name].values

        tiv = clinical_df['PTEDUCAT'].values
        tiv = tiv[:, np.newaxis]

        x_dataset = (np.true_divide(x_dataset, tiv)).astype('float32')
        # ----------------------------------------------------------------------------
        # print(bootstrap_model_dir)
        encoder = keras.models.load_model(bootstrap_model_dir / 'encoder.h5', compile=False)
        decoder = keras.models.load_model(bootstrap_model_dir / 'decoder.h5', compile=False)

        scaler = joblib.load(bootstrap_model_dir / 'scaler.joblib')

        enc_age = joblib.load(bootstrap_model_dir / 'age_encoder.joblib')
        enc_gender = joblib.load(bootstrap_model_dir / 'gender_encoder.joblib')
        
        age = clinical_df['AGE'].values[:, np.newaxis].astype('float32')
        #print(age, age.shape)
        one_hot_age2 = enc_age.transform(age)
        #print(one_hot_age, one_hot_age.shape)
        
        bin_labels = list(range(0,10))  
        age_bins_test, bin_edges = pd.cut(clinical_df['AGE'], 10, retbins=True, labels=bin_labels)
        #age_bins_train, bin_edges = pd.cut(train_covariates['AGE'], 10, retbins=True, labels=bin_labels)
        #age_bins_test = pd.cut(test_covariates['AGE'], retbins=True, labels=bin_labels)
        #age_bins_train.fillna(0, inplace=True)
        age_bins_test.fillna(0,inplace = True)
        one_hot_age = np.eye(10)[age_bins_test.values]
        
        
        gender = clinical_df['PTGENDER'].values[:, np.newaxis].astype('float32')
        one_hot_gender = enc_gender.transform(gender)
        

            
        
        y_data = np.concatenate((one_hot_age, one_hot_gender), axis=1).astype('float32')
        
        #extended = []
        #for i in range(y_data.shape[1]):
        #    val = "d" + str(i)
        #    extended.append(val)
        # ----------------------------------------------------------------------------
        x_normalized = scaler.transform(x_dataset)
        #x = np.concatenate((x_normalized, y_data), axis=1)
        #columns_name.append(['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16',
        #                     '17','18','19','20','21','22','23','24','25','26','27'])
        #tmp = copy.copy(columns_name)
        #tmp.extend(extended)
        #print(columns_name)
        normalized_df = pd.DataFrame(columns=['participant_id'] + columns_name)
        normalized_df['participant_id'] = clinical_df['participant_id']
        normalized_df[columns_name] = x_normalized
        normalized_df.to_csv(output_dataset_dir / 'normalized.csv', index=False)

        # ----------------------------------------------------------------------------
        #age = clinical_df['AGE'].values[:, np.newaxis].astype('float32')
        #one_hot_age = enc_age.transform(age)

        #gender = clinical_df['PTGENDER'].values[:, np.newaxis].astype('float32')
        #one_hot_gender = enc_gender.transform(gender)

        #y_data = np.concatenate((one_hot_age, one_hot_gender), axis=1).astype('float32')

        # ----------------------------------------------------------------------------
        #x = np.concatenate((x_normalized, y_data), axis=1)
        encoded = encoder(x_normalized, training=False)
        #encoded = encoder(x, training=False)
        #reconstruction = decoder(tf.concat([encoded,y_data], axis=1), training=False)
        reconstruction = decoder(tf.concat([encoded], axis=1), training=False)

        reconstruction_df = pd.DataFrame(columns=['participant_id'] + columns_name)
        reconstruction_df['participant_id'] = clinical_df['participant_id']
        print('reconstruction:', reconstruction)
        print('reconstruction_df:', reconstruction_df)
        reconstruction_df[columns_name] = reconstruction.numpy()

        reconstruction_df.to_csv(output_dataset_dir / 'reconstruction.csv', index=False)

        encoded_df = pd.DataFrame(columns=['participant_id'] + list(range(encoded.shape[1])))
        
        encoded_df['participant_id'] = clinical_df['participant_id']
        encoded_df[list(range(encoded.shape[1]))] = encoded.numpy()
        encoded_df.to_csv(output_dataset_dir / 'encoded.csv', index=False)

        # ----------------------------------------------------------------------------
        #reconstruction_error = np.mean((x - reconstruction) ** 2, axis=1)
        reconstruction_error = np.mean((x_normalized - reconstruction) ** 2, axis=1)
        boostrap_error_mean.append(np.mean(reconstruction_error))

        reconstruction_error_df = pd.DataFrame(columns=['participant_id', 'Reconstruction error'])
        reconstruction_error_df['participant_id'] = clinical_df['participant_id']
        reconstruction_error_df['Reconstruction error'] = reconstruction_error
        reconstruction_error_df.to_csv(output_dataset_dir / 'reconstruction_error.csv', index=False)
    
    boostrap_error_mean = np.array(boostrap_error_mean)
    boostrap_mean = np.mean(boostrap_error_mean)
    bootsrao_var = np.std(boostrap_error_mean)
    boostrap_list = np.array([boostrap_mean, bootsrao_var])
    np.savetxt("ae_boostrap_mean_std.csv", boostrap_list, delimiter=",")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--dataset_name',
                        dest='dataset_name',
                        help='Dataset name to calculate deviations.')
    args = parser.parse_args()
    if args.dataset_name == 'snp':
        COLUMNS_NAME = COLUMNS_NAME_SNP
    elif args.dataset_name == 'vbm':
        COLUMNS_NAME = COLUMNS_NAME_VBM
    elif args.dataset_name == '3modalities':
        COLUMNS_NAME = COLUMNS_3MODALITIES

    if args.dataset_name == None:
        args.dataset_name = 'T1_volume'
    main(args.dataset_name)
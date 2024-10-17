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

    participants_path = PROJECT_ROOT / 'data' / 'y.csv'
    freesurfer_path = PROJECT_ROOT / 'data' / (dataset_name + '.csv')


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
        dataset_names = ['av45', 'fdg', 'vbm', '3modalities']
        x_normalized_list = []
        n_features_list = []
        decoder_list = []
        encoder_output_list = []
        clinical_df_list = []
        bootstrap_model_dir = model_dir / '{:03d}'.format(i_bootstrap)

        output_dataset_dir = bootstrap_model_dir / dataset_name
        output_dataset_dir.mkdir(exist_ok=True)

        # ----------------------------------------------------------------------------
        # Loading data
        ids_dir = bootstrap_dir / 'ids'
        test_ids_filename = 'cleaned_bootstrap_test_{:03d}.csv'.format(i_bootstrap)
        ids_path = ids_dir / test_ids_filename

        # read scaler_list from scaler.pkl
        scaler_list = joblib.load(bootstrap_model_dir / 'scaler.pkl')
        enc_age_list = joblib.load(bootstrap_model_dir / 'enc_age.pkl')
        enc_gender_list = joblib.load(bootstrap_model_dir / 'enc_gender.pkl')

        for dataset_name in dataset_names:

            
            if dataset_name == 'av45' or dataset_name == 'fdg':
                columns_name = COLUMNS_NAME

            elif dataset_name == 'vbm':
                columns_name = COLUMNS_NAME_VBM
            elif dataset_name == '3modalities':
                columns_name = COLUMNS_3MODALITIES

            freesurfer_path = PROJECT_ROOT / 'data' / (dataset_name + '.csv')

            clinical_df = load_dataset(participants_path, ids_path, freesurfer_path)
            clinical_df_list.append(clinical_df)
            
            
            x_dataset = clinical_df[columns_name].values

            tiv = clinical_df['PTEDUCAT'].values
            tiv = tiv[:, np.newaxis]

            x_dataset = (np.true_divide(x_dataset, tiv)).astype('float32')

            scaler = scaler_list[dataset_names.index(dataset_name)]
            enc_age = enc_age_list[dataset_names.index(dataset_name)]
            enc_gender = enc_gender_list[dataset_names.index(dataset_name)]
            
            age = clinical_df['AGE'].values[:, np.newaxis].astype('float32')
            #print(age, age.shape)
            one_hot_age2 = enc_age.transform(age)
            #print(one_hot_age, one_hot_age.shape)
            
            bin_labels = list(range(0,10))  
            age_bins_test, bin_edges = pd.cut(clinical_df['AGE'], 10, retbins=True, labels=bin_labels)
            age_bins_test.fillna(0,inplace = True)
            one_hot_age = np.eye(10)[age_bins_test.values]
            
            
            gender = clinical_df['PTGENDER'].values[:, np.newaxis].astype('float32')
            one_hot_gender = enc_gender.transform(gender)
            

                
            
            y_data = np.concatenate((one_hot_age, one_hot_gender), axis=1).astype('float32')
            

            x_normalized = scaler.transform(x_dataset)

            normalized_df = pd.DataFrame(columns=['participant_id'] + columns_name)
            normalized_df['participant_id'] = clinical_df['participant_id']
            normalized_df[columns_name] = x_normalized
            # normalized_df.to_csv(output_dataset_dir / 'normalized.csv', index=False)
            normalized_df.to_csv(output_dataset_dir / 'normalized_{}.csv'.format(dataset_name), index=False)
            x_normalized_list.append(x_normalized)

        encoder = keras.models.load_model(bootstrap_model_dir / 'encoder.h5', compile=False)
        decoder = keras.models.load_model(bootstrap_model_dir / 'decoder.h5', compile=False)

        encoder0 = keras.models.load_model(bootstrap_model_dir / 'encoder0.h5', compile=False)
        encoder1 = keras.models.load_model(bootstrap_model_dir / 'encoder1.h5', compile=False)
        encoder2 = keras.models.load_model(bootstrap_model_dir / 'encoder2.h5', compile=False)
        encoder3 = keras.models.load_model(bootstrap_model_dir / 'encoder3.h5', compile=False)
        
        encoder_list = [encoder0, encoder1, encoder2, encoder3]


        
        for i in range(4):
            decoder_list.append(keras.models.load_model(bootstrap_model_dir / 'decoder{}.h5'.format(i), compile=False))

        for i_dataset in range(len(dataset_names)):
            encoder_output_list.append(encoder_list[i_dataset](x_normalized_list[i_dataset], training=False))

        # encoder_output_multimodal = tf.concat(encoder_output_list, axis=1)

        # get the mean of the encoded features
        # for idx in range(len(encoder_output_list)):
        #     if idx == 0:
        #         encoder_output_multimodal = encoder_output_list[idx]
        #     else:
        #         encoder_output_multimodal += encoder_output_list[idx]

        # encoder_output_multimodal /= len(encoder_output_list)

        encoder_output_multimodal = tf.concat(encoder_output_list, axis=1)
        
        reconstruction_df_list = []

        for i_dataset in range(len(dataset_names)):
            if dataset_names[i_dataset] == 'av45' or dataset_names[i_dataset] == 'fdg': 
                columns_name = COLUMNS_NAME
            elif dataset_names[i_dataset] == 'snp':
                columns_name = COLUMNS_NAME_SNP
            elif dataset_names[i_dataset] == 'vbm':
                columns_name = COLUMNS_NAME_VBM
            elif dataset_names[i_dataset] == '3modalities':
                columns_name = COLUMNS_3MODALITIES

            reconstruction = decoder_list[i_dataset](tf.concat([encoder_output_multimodal], axis=1), training=False)
            reconstruction_df = pd.DataFrame(columns=['participant_id'] + columns_name)
            reconstruction_df['participant_id'] = clinical_df_list[i_dataset]['participant_id']

            reconstruction_df[columns_name] = reconstruction.numpy()
            reconstruction_df.to_csv(output_dataset_dir / 'reconstruction_{}.csv'.format(dataset_names[i_dataset]), index=False)

            encoded_df = pd.DataFrame(columns=['participant_id'] + list(range(encoder_output_list[i_dataset].shape[1])))
            encoded_df['participant_id'] = clinical_df_list[i_dataset]['participant_id']
            encoded_df[list(range(encoder_output_list[i_dataset].shape[1]))] = encoder_output_list[i_dataset].numpy()
            encoded_df.to_csv(output_dataset_dir / 'encoded_{}.csv'.format(dataset_names[i_dataset]), index=False)

            reconstruction_error = np.mean((x_normalized_list[i_dataset] - reconstruction) ** 2, axis=1)
            boostrap_error_mean.append(np.mean(reconstruction_error))

            reconstruction_error_df = pd.DataFrame(columns=['participant_id', 'Reconstruction error'])
            reconstruction_error_df['participant_id'] = clinical_df_list[i_dataset]['participant_id']
            reconstruction_error_df['Reconstruction error'] = reconstruction_error
            reconstruction_error_df.to_csv(output_dataset_dir / 'reconstruction_error_{}.csv'.format(dataset_names[i_dataset]), index=False)


    
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

    if args.dataset_name is None:
        args.dataset_name = 'fdg'
    main(args.dataset_name)
#!/usr/bin/env python3
"""Script to train the deterministic supervised adversarial autoencoder."""
from pathlib import Path
import random as rn
import time

import joblib
from sklearn.preprocessing import RobustScaler, OneHotEncoder
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import argparse
from utils import COLUMNS_HCP, COLUMNS_NAME, load_dataset, COLUMNS_NAME_SNP, COLUMNS_NAME_VBM, COLUMNS_3MODALITIES
from models import make_encoder_model_v111, make_decoder_model_v1, make_discriminator_model_v1
import argparse
import numpy as np
from sklearn.cross_decomposition import CCA

PROJECT_ROOT = Path.cwd()

# define device as cuda:1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"



def main(dataset_name, hz_para_list, base_lr=0.0001, max_lr=0.005):
    """Train the normative method on the bootstrapped samples.

    The script also the scaler and the demographic data encoder.
    """
    # ----------------------------------------------------------------------------
    n_bootstrap = 10
    model_name = 'supervised_ae'

    participants_path = PROJECT_ROOT / 'data' / 'y.csv'
    freesurfer_path = PROJECT_ROOT / 'data' / (dataset_name + '.csv')

    # ----------------------------------------------------------------------------
    bootstrap_dir = PROJECT_ROOT / 'outputs' / 'bootstrap_analysis'
    ids_dir = bootstrap_dir / 'ids'

    model_dir = bootstrap_dir / model_name
    model_dir.mkdir(exist_ok=True)

    # ----------------------------------------------------------------------------
    # Set random seed
    random_seed = 42
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)
    rn.seed(random_seed)

    for i_bootstrap in range(n_bootstrap):

        dataset_names = ['av45', 'fdg', 'vbm', '3modalities']
        train_datasets = []
        n_features_list = []
        scaler_list = []
        enc_age_list = []
        enc_gender_list = []


        for dataset_name in dataset_names:
            if dataset_name == 'av45' or dataset_name == 'fdg':
                columns_name = COLUMNS_NAME
            elif dataset_name == 'snp':
                columns_name = COLUMNS_NAME_SNP
            elif dataset_name == 'vbm':
                columns_name = COLUMNS_NAME_VBM
            elif dataset_name == '3modalities':
                columns_name = COLUMNS_3MODALITIES
            freesurfer_path = PROJECT_ROOT / 'data' / (dataset_name + '.csv')
            
            ids_filename = 'cleaned_bootstrap_{:03d}.csv'.format(i_bootstrap)
            ids_path = ids_dir / ids_filename

            bootstrap_model_dir = model_dir / '{:03d}'.format(i_bootstrap)
            bootstrap_model_dir.mkdir(exist_ok=True)

            # ----------------------------------------------------------------------------
            # Loading data
            dataset_df = load_dataset(participants_path, ids_path, freesurfer_path)


            # ----------------------------------------------------------------------------
            dataset_df = dataset_df.loc[dataset_df['DIA'] == 2]      
            x_data = dataset_df[columns_name].values
            

            tiv = dataset_df['PTEDUCAT'].values
            tiv = tiv[:, np.newaxis]

            x_data = (np.true_divide(x_data, tiv)).astype('float32')

            scaler = RobustScaler()
            x_data_normalized = scaler.fit_transform(x_data)
        
            # ----------------------------------------------------------------------------
            age = dataset_df['AGE'].values[:, np.newaxis].astype('float32')
            #enc_age = OneHotEncoder(sparse=False)
            enc_age = OneHotEncoder(handle_unknown = "ignore",sparse=False)
            one_hot_age2 = enc_age.fit_transform(age)
            
            bin_labels = list(range(0,10))  
            age_bins_test, bin_edges = pd.cut(dataset_df['AGE'], 10, retbins=True, labels=bin_labels)

            age_bins_test.fillna(0,inplace = True)
            one_hot_age = np.eye(10)[age_bins_test.values]

            gender = dataset_df['PTGENDER'].values[:, np.newaxis].astype('float32')
            enc_gender = OneHotEncoder(sparse=False)
            one_hot_gender = enc_gender.fit_transform(gender)

            y_data = np.concatenate((one_hot_age, one_hot_gender), axis=1).astype('float32')

            # -------------------------------------------------------------------------------------------------------------
            # Create the dataset iterator
            batch_size = 256
            n_samples = x_data.shape[0]

            train_dataset = tf.data.Dataset.from_tensor_slices((x_data_normalized, y_data))
            train_dataset = train_dataset.shuffle(buffer_size=n_samples)
            train_dataset = train_dataset.batch(batch_size)

            train_datasets.append(train_dataset)

            # -------------------------------------------------------------------------------------------------------------
            # Create models
            n_features = x_data_normalized.shape[1]
            n_features_list.append(n_features)
            n_labels = y_data.shape[1]

            scaler_list.append(scaler)
            enc_age_list.append(enc_age)
            enc_gender_list.append(enc_gender)


            h_dim = hz_para_list[:-1]
            z_dim = hz_para_list[-1]

            print('h_dim:', h_dim)  
            print('z_dim:', z_dim)


        encoder = make_encoder_model_v111(n_features, h_dim, z_dim)
        decoder = make_decoder_model_v1(z_dim, n_features, h_dim) 
        discriminator = make_discriminator_model_v1(z_dim, h_dim)

        encoder0 = make_encoder_model_v111(n_features_list[0], h_dim, z_dim)
        encoder1 = make_encoder_model_v111(n_features_list[1], h_dim, z_dim)
        encoder2 = make_encoder_model_v111(n_features_list[2], h_dim, z_dim)
        encoder3 = make_encoder_model_v111(n_features_list[3], h_dim, z_dim * 3)

        z_dim_decoder = z_dim * 6

        decoder0 = make_decoder_model_v1(z_dim_decoder, n_features_list[0], h_dim)
        decoder1 = make_decoder_model_v1(z_dim_decoder, n_features_list[1], h_dim)
        decoder2 = make_decoder_model_v1(z_dim_decoder, n_features_list[2], h_dim)
        decoder3 = make_decoder_model_v1(z_dim_decoder, n_features_list[3], h_dim)


        # -------------------------------------------------------------------------------------------------------------
        # Define loss functions
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        mse = tf.keras.losses.MeanSquaredError()
        accuracy = tf.keras.metrics.BinaryAccuracy()

        # -------------------------------------------------------------------------------------------------------------
        # Define optimizers
        # base_lr = 0.0001
        # max_lr = 0.005

        step_size = 2 * np.ceil(n_samples / batch_size)

        ae_optimizer = tf.keras.optimizers.Adam(lr=base_lr)

        

        # -------------------------------------------------------------------------------------------------------------
        # Training function
        @tf.function
        def train_step(batch_x, batch_y):
            # -------------------------------------------------------------------------------------------------------------
            # Autoencoder
            with tf.GradientTape() as ae_tape:
                encoder_output = encoder(batch_x, training=True)
                decoder_output = decoder(tf.concat(encoder_output, axis=1), training=True)

                # Autoencoder loss
                ae_loss = mse(batch_x, decoder_output)

            ae_grads = ae_tape.gradient(ae_loss, encoder.trainable_variables + decoder.trainable_variables)
            ae_optimizer.apply_gradients(zip(ae_grads, encoder.trainable_variables + decoder.trainable_variables))

            return ae_loss
        
        @tf.function
        def train_step_multimodal(batch_x0, batch_y0, batch_x1, batch_y1, batch_x2, batch_y2, batch_x3, batch_y3):
            # -------------------------------------------------------------------------------------------------------------
            # Autoencoder
            with tf.GradientTape() as ae_tape:
                encoder_output0 = encoder0(batch_x0, training=True)
                encoder_output1 = encoder1(batch_x1, training=True)
                encoder_output2 = encoder2(batch_x2, training=True)
                encoder_output3 = encoder3(batch_x3, training=True)
                
                # concatenate the output of the 4 encoders, then get the output of the same size of the input
                # encoder_output_multimodal = [encoder_output0, encoder_output1, encoder_output2, encoder_output3]

                # get average of 4 encoder output as the input of the decoder
                # encoder_output_multimodal = (encoder_output0 + encoder_output1 + encoder_output2 + encoder_output3) / 4

                encoder_output_multimodal = tf.concat([encoder_output0, encoder_output1, encoder_output2, encoder_output3], axis=1)

                decoder_output0 = decoder0(tf.concat(encoder_output_multimodal, axis=1), training=True)
                decoder_output1 = decoder1(tf.concat(encoder_output_multimodal, axis=1), training=True)
                decoder_output2 = decoder2(tf.concat(encoder_output_multimodal, axis=1), training=True)
                decoder_output3 = decoder3(tf.concat(encoder_output_multimodal, axis=1), training=True)

                print('batch_x0.shape:', batch_x0.shape)
                print('decoder_output0.shape:', decoder_output0.shape)

                ae_loss0 = mse(batch_x0, decoder_output0)
                ae_loss1 = mse(batch_x1, decoder_output1)
                ae_loss2 = mse(batch_x2, decoder_output2)
                ae_loss3 = mse(batch_x3, decoder_output3)

                ae_loss = ae_loss0 + ae_loss1 + ae_loss2 + ae_loss3

            ae_grads = ae_tape.gradient(ae_loss, encoder0.trainable_variables + encoder1.trainable_variables + encoder2.trainable_variables + encoder3.trainable_variables + decoder0.trainable_variables + decoder1.trainable_variables + decoder2.trainable_variables + decoder3.trainable_variables)
            ae_optimizer.apply_gradients(zip(ae_grads, encoder0.trainable_variables + encoder1.trainable_variables + encoder2.trainable_variables + encoder3.trainable_variables + decoder0.trainable_variables + decoder1.trainable_variables + decoder2.trainable_variables + decoder3.trainable_variables))
            return ae_loss

        # -------------------------------------------------------------------------------------------------------------
        # Training loop
        global_step = 0
        n_epochs = 200
        gamma = 0.98
        scale_fn = lambda x: gamma ** x
        for epoch in range(n_epochs):
            start = time.time()

            epoch_ae_loss_avg = tf.metrics.Mean()
            

            for (batch_x0, batchy0), (batch_x1, batchy1), (batch_x2, batchy2), (batch_x3, batchy3) in zip(train_datasets[0], train_datasets[1], train_datasets[2], train_datasets[3]):
                global_step = global_step + 1
                cycle = np.floor(1 + global_step / (2 * step_size))
                x_lr = np.abs(global_step / step_size - 2 * cycle + 1)
                clr = base_lr + (max_lr - base_lr) * max(0, 1 - x_lr) * scale_fn(cycle)
                ae_optimizer.lr = clr

                ########
                ae_loss = train_step_multimodal(batch_x0, batchy0, batch_x1, batchy1, batch_x2, batchy2, batch_x3, batchy3)

                epoch_ae_loss_avg(ae_loss)


            epoch_time = time.time() - start

            print('{:4d}: TIME: {:.2f} ETA: {:.2f} AE_LOSS: {:.4f}' \
                  .format(epoch, epoch_time,
                          epoch_time * (n_epochs - epoch),
                          epoch_ae_loss_avg.result(),
                          ))

        # Save models
        encoder.save(bootstrap_model_dir / 'encoder.h5')
        decoder.save(bootstrap_model_dir / 'decoder.h5')
        encoder0.save(bootstrap_model_dir / 'encoder0.h5')
        encoder1.save(bootstrap_model_dir / 'encoder1.h5')
        encoder2.save(bootstrap_model_dir / 'encoder2.h5')
        encoder3.save(bootstrap_model_dir / 'encoder3.h5')
        decoder0.save(bootstrap_model_dir / 'decoder0.h5')
        decoder1.save(bootstrap_model_dir / 'decoder1.h5')
        decoder2.save(bootstrap_model_dir / 'decoder2.h5')
        decoder3.save(bootstrap_model_dir / 'decoder3.h5')
        #discriminator.save(bootstrap_model_dir / 'discriminator.h5')

        # Save scaler
        joblib.dump(scaler_list, bootstrap_model_dir / 'scaler.pkl')
        joblib.dump(enc_age_list, bootstrap_model_dir / 'enc_age.pkl')
        joblib.dump(enc_gender_list, bootstrap_model_dir / 'enc_gender.pkl')


if __name__ == "__main__":

    argparse.ArgumentParser()
    # add dataset_name to parse
    parser = argparse.ArgumentParser(description='Train the supervised adversarial autoencoder.')
    parser.add_argument('-D', '--dataset_name',
                    dest='dataset_name',
                    help='Dataset to use for training test and evaluation.',
                    type=str)
    parser.add_argument('-H', '--hz_para_list',
                        dest='hz_para_list',
                        nargs='+',
                        help='List of paras to perform the analysis.',
                        type=int)
    # add base_lr to parse
    # add max_lr to parse
    parser.add_argument('-B', '--base_lr',
                        dest='base_lr',
                        help='Base learning rate.',
                        type=float)
    parser.add_argument('-M', '--max_lr',
                        dest='max_lr',
                        help='Max learning rate.',
                        type=float)
    args = parser.parse_args()

    if args.dataset_name is None:
        args.dataset_name = 'fdg'
    if args.hz_para_list is None:
        args.hz_para_list = [110, 110, 10]
    if args.base_lr is None:
        args.base_lr = 0.0001
    if args.max_lr is None:
        args.max_lr = 0.005
    

    main(args.dataset_name, args.hz_para_list, args.base_lr, args.max_lr)

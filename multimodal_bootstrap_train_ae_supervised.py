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
import tensorflow as tf
from tensorflow.keras import layers


PROJECT_ROOT = Path.cwd()

# define device as cuda:1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
class CrossAttention(layers.Layer):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CrossAttention, self).__init__()
        self.query = layers.Dense(hidden_dim)
        self.key = layers.Dense(hidden_dim)
        self.value = layers.Dense(hidden_dim)
        self.output_layer = layers.Dense(output_dim)
        self.softmax = layers.Softmax(axis=-1)

    def call(self, x1, x2, x3):
        Q1, Q2, Q3 = self.query(x1), self.query(x2), self.query(x3)
        K1, K2, K3 = self.key(x1), self.key(x2), self.key(x3)
        V1, V2, V3 = self.value(x1), self.value(x2), self.value(x3)

        attention_weights_12 = self.softmax(tf.matmul(Q1, K2, transpose_b=True) / tf.sqrt(tf.cast(K2.shape[-1], tf.float32)))
        attention_weights_13 = self.softmax(tf.matmul(Q1, K3, transpose_b=True) / tf.sqrt(tf.cast(K3.shape[-1], tf.float32)))
        attention_weights_23 = self.softmax(tf.matmul(Q2, K3, transpose_b=True) / tf.sqrt(tf.cast(K3.shape[-1], tf.float32)))
        attention_weights_21 = self.softmax(tf.matmul(Q2, K1, transpose_b=True) / tf.sqrt(tf.cast(K1.shape[-1], tf.float32)))
        attention_weights_31 = self.softmax(tf.matmul(Q3, K1, transpose_b=True) / tf.sqrt(tf.cast(K1.shape[-1], tf.float32)))
        attention_weights_32 = self.softmax(tf.matmul(Q3, K2, transpose_b=True) / tf.sqrt(tf.cast(K2.shape[-1], tf.float32)))

        fused_x1 = tf.matmul(attention_weights_12, V2) + tf.matmul(attention_weights_13, V3)
        fused_x2 = tf.matmul(attention_weights_21, V1) + tf.matmul(attention_weights_23, V3)
        fused_x3 = tf.matmul(attention_weights_31, V1) + tf.matmul(attention_weights_32, V2)

        fused_x = tf.concat([fused_x1, fused_x2, fused_x3], axis=-1)

        output = self.output_layer(fused_x)

        return output


class CrossAttentionLatent(layers.Layer):
    def __init__(self, input_dim, hidden_dim):
        super(CrossAttentionLatent, self).__init__()
        self.query = layers.Dense(hidden_dim)
        self.key = layers.Dense(hidden_dim)
        self.value = layers.Dense(hidden_dim)
        self.softmax = layers.Softmax(axis=-1)

    def call(self, x1, x2, x3):
        Q1, Q2, Q3 = self.query(x1), self.query(x2), self.query(x3)
        K1, K2, K3 = self.key(x1), self.key(x2), self.key(x3)
        V1, V2, V3 = self.value(x1), self.value(x2), self.value(x3)

        attention_weights_12 = self.softmax(tf.matmul(Q1, K2, transpose_b=True) / tf.sqrt(tf.cast(K2.shape[-1], tf.float32)))
        attention_weights_13 = self.softmax(tf.matmul(Q1, K3, transpose_b=True) / tf.sqrt(tf.cast(K3.shape[-1], tf.float32)))
        attention_weights_23 = self.softmax(tf.matmul(Q2, K3, transpose_b=True) / tf.sqrt(tf.cast(K3.shape[-1], tf.float32)))
        attention_weights_21 = self.softmax(tf.matmul(Q2, K1, transpose_b=True) / tf.sqrt(tf.cast(K1.shape[-1], tf.float32)))
        attention_weights_31 = self.softmax(tf.matmul(Q3, K1, transpose_b=True) / tf.sqrt(tf.cast(K1.shape[-1], tf.float32)))
        attention_weights_32 = self.softmax(tf.matmul(Q3, K2, transpose_b=True) / tf.sqrt(tf.cast(K2.shape[-1], tf.float32)))

        fused_x1 = tf.matmul(attention_weights_12, V2) + tf.matmul(attention_weights_13, V3)
        fused_x2 = tf.matmul(attention_weights_21, V1) + tf.matmul(attention_weights_23, V3)
        fused_x3 = tf.matmul(attention_weights_31, V1) + tf.matmul(attention_weights_32, V2)

        return fused_x1, fused_x2, fused_x3



def main(dataset_resourse, hz_para_list, base_lr, max_lr, procedure):
    """Train the normative method on the bootstrapped samples.

    The script also the scaler and the demographic data encoder.
    """
    # ----------------------------------------------------------------------------
    n_bootstrap = 10
    model_name = 'supervised_ae'

    

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

        print('Bootstrap iteration: {:03d}'.format(i_bootstrap))


        if dataset_resourse == 'ADNI':
            dataset_names = ['av45', 'fdg', 'vbm']
            # dataset_names = ['av45', 'fdg', 'vbm', '3modalities']
        elif dataset_resourse == 'HCP':
            dataset_names = ['T1_volume', 'mean_T1_intensity', 'mean_FA', 'mean_MD', 'mean_L1', 'mean_L2', 'mean_L3', 'min_BOLD', '25_percentile_BOLD', '50_percentile_BOLD', '75_percentile_BOLD', 'max_BOLD']
        else:
            raise ValueError('Unknown dataset resource')

        train_datasets = []
        n_features_list = []
        scaler_list = []
        enc_age_list = []
        enc_gender_list = []


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

            
            participants_path = PROJECT_ROOT / 'data'  / dataset_resourse / 'y.csv'
            freesurfer_path = PROJECT_ROOT / 'data' / dataset_resourse  / (dataset_name + '.csv')


            
            ids_filename = 'cleaned_bootstrap_{:03d}.csv'.format(i_bootstrap)
            ids_path = ids_dir / ids_filename

            bootstrap_model_dir = model_dir / '{:03d}'.format(i_bootstrap)
            bootstrap_model_dir.mkdir(exist_ok=True)

            # ----------------------------------------------------------------------------
            # Loading data
            dataset_df = load_dataset(participants_path, ids_path, freesurfer_path)

            if dataset_resourse == 'ADNI':
                hc_label = 2
            elif dataset_resourse == 'HCP':
                hc_label = 1
            else:
                raise ValueError('Unknown dataset resource')


            # ----------------------------------------------------------------------------
            dataset_df = dataset_df.loc[dataset_df['DIA'] == hc_label]      
            x_data = dataset_df[columns_name].values
            

            # # tiv = dataset_df['PTEDUCAT'].values
            # # tiv = tiv[:, np.newaxis]

            # x_data = (np.true_divide(x_data, tiv)).astype('float32')

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

            gender = dataset_df['PTGENDER'].values[:, np.newaxis].astype('bool')
            
            enc_gender = OneHotEncoder(sparse=False)
            one_hot_gender = enc_gender.fit_transform(gender)

            print("Known categories:", enc_gender.categories_)


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

        # encoder0 = make_encoder_model_v111(n_features_list[0], h_dim, z_dim)
        # encoder1 = make_encoder_model_v111(n_features_list[1], h_dim, z_dim)
        # encoder2 = make_encoder_model_v111(n_features_list[2], h_dim, z_dim)
        # encoder3 = make_encoder_model_v111(n_features_list[3], h_dim, z_dim)

        # create an encoder list, contains length of dataset_names encoders
        encoder_list = [make_encoder_model_v111(n_features_list[i], h_dim, z_dim) for i in range(len(dataset_names))]

        z_dim = z_dim * len(dataset_names)

        # decoder0 = make_decoder_model_v1(z_dim, n_features_list[0], h_dim)
        # decoder1 = make_decoder_model_v1(z_dim, n_features_list[1], h_dim)
        # decoder2 = make_decoder_model_v1(z_dim, n_features_list[2], h_dim)
        # decoder3 = make_decoder_model_v1(z_dim, n_features_list[3], h_dim)


        # create a decoder list, contains length of dataset_names decoders
        decoder_list = [make_decoder_model_v1(z_dim, n_features_list[i], h_dim) for i in range(len(dataset_names))]


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

        # 假设输入数据的特征维度
        input_dims = [n_features_list[i] for i in range(len(dataset_names))]
        output_dim = 90

        hidden_dim = hz_para_list[-1]  # 使用latent dimension作为hidden dimension
        # cross_attention = CrossAttention(input_dim=hidden_dim, hidden_dim=hidden_dim)  # 假设所有latent dimension相同


        # # 创建Cross Attention机制
        # cross_attention = CrossAttention(input_dim=input_dims[0], hidden_dim=input_dims[0], output_dim=output_dim)  # 假设所有输入维度相同

        cross_attention_latent = CrossAttentionLatent(input_dim=hidden_dim, hidden_dim=hidden_dim)  # 假设所有latent dimension相同
        

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








        # @tf.function
        # def train_step_multimodal(batch_x_list, batch_y_list):
        #     with tf.GradientTape() as ae_tape:
        #         # 获得每个模态的encoder输出
        #         encoder_output_list = [encoder_list[i](batch_x_list[i], training=True) for i in range(len(dataset_names))]
                
        #         # 将encoder输出应用cross attention
        #         fused_output1, fused_output2, fused_output3 = cross_attention_latent(encoder_output_list[0], encoder_output_list[1], encoder_output_list[2])

        #         # 将cross attention的输出连接起来
        #         encoder_output_multimodal = tf.concat([fused_output1, fused_output2, fused_output3], axis=1)

        #         # 将融合后的latent表示输入到decoder中
        #         decoder_output_list = [decoder_list[i](encoder_output_multimodal, training=True) for i in range(len(dataset_names))]
        #         ae_loss_list = [mse(batch_x_list[i], decoder_output_list[i]) for i in range(len(dataset_names))]
        #         ae_loss = tf.reduce_sum(ae_loss_list)

        #         trainable_variables = []
        #         for i in range(len(dataset_names)):
        #             trainable_variables.extend(encoder_list[i].trainable_variables)
        #             trainable_variables.extend(decoder_list[i].trainable_variables)

        #         # 添加Cross Attention层的可训练变量
        #         trainable_variables.extend(cross_attention_latent.trainable_variables)

        #         ae_grads = ae_tape.gradient(ae_loss, trainable_variables)
        #         ae_grads = [grad if grad is not None else tf.zeros_like(var) for grad, var in zip(ae_grads, trainable_variables)]
        #         ae_optimizer.apply_gradients(zip(ae_grads, trainable_variables))

        #     return ae_loss
        
        @tf.function
        def train_step_multimodal(batch_x_list, batch_y_list):
            # -------------------------------------------------------------------------------------------------------------
            # Autoencoder
            with tf.GradientTape() as ae_tape:

                encoder_output_list = [encoder_list[i](batch_x_list[i], training=True) for i in range(len(dataset_names))]

                encoder_output_multimodal = tf.concat(encoder_output_list, axis=1)
                
                if procedure == 'SE-DC':
                    encoder_output_multimodal = tf.concat(encoder_output_list, axis=1)
                elif procedure == 'SE-CA':
                    fused_output1, fused_output2, fused_output3 = cross_attention_latent(encoder_output_list[0], encoder_output_list[1], encoder_output_list[2])
                    encoder_output_multimodal = tf.concat([fused_output1, fused_output2, fused_output3], axis=1)
                elif procedure == 'SE-Mixture':
                    # for encoder_output in encoder_output_list, get a new output of their sum
                    encoder_output_mixture = tf.reduce_sum(encoder_output_list, axis=0)
                    encoder_output_multimodal = tf.concat([encoder_output_mixture for i in range(len(dataset_names))], axis=1)


                    
                    

                decoder_output_list = [decoder_list[i](encoder_output_multimodal, training=True) for i in range(len(dataset_names))]

                ae_loss_list = [mse(batch_x_list[i], decoder_output_list[i]) for i in range(len(dataset_names))]

                ae_loss = tf.reduce_sum(ae_loss_list)

                # Step 1: Create the list of trainable variables
                trainable_variables = []
                for i in range(len(dataset_names)):
                    trainable_variables.extend(encoder_list[i].trainable_variables)
                    trainable_variables.extend(decoder_list[i].trainable_variables)

                # Step 2: Calculate the gradients
                ae_grads = ae_tape.gradient(ae_loss, trainable_variables)

                # Step 3: Check the lengths of gradients and variables
                print("Length of gradients:", len(ae_grads))
                print("Length of trainable variables:", len(trainable_variables))

                # Ensure there are no None gradients
                ae_grads = [grad if grad is not None else tf.zeros_like(var) for grad, var in zip(ae_grads, trainable_variables)]

                # Step 4: Apply the gradients
                ae_optimizer.apply_gradients(zip(ae_grads, trainable_variables))

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



            for batches in zip(*train_datasets):
                batch_x_list = [batch[0] for batch in batches]
                batch_y_list = [batch[1] for batch in batches]
                # Process batch_x_list and batch_y_list as needed

                global_step = global_step + 1
                cycle = np.floor(1 + global_step / (2 * step_size))
                x_lr = np.abs(global_step / step_size - 2 * cycle + 1)
                clr = base_lr + (max_lr - base_lr) * max(0, 1 - x_lr) * scale_fn(cycle)
                ae_optimizer.lr = clr

                ########
                # ae_loss = train_step_multimodal(batch_x0, batchy0, batch_x1, batchy1, batch_x2, batchy2, batch_x3, batchy3)
                ae_loss = train_step_multimodal(batch_x_list, batch_y_list)

                # print batch_x_list to see if all data is loaded
                # print('batch_x_list:', batch_x_list)

                # check null values in batch_x_list
                

                epoch_ae_loss_avg(ae_loss)


            epoch_time = time.time() - start

            #print('{:4d}: TIME: {:.2f} ETA: {:.2f} AE_LOSS: {:.4f} DC_LOSS: {:.4f} DC_ACC: {:.4f} GEN_LOSS: {:.4f}' \
            print('{:4d}: TIME: {:.2f} ETA: {:.2f} AE_LOSS: {:.4f}' \
                  .format(epoch, epoch_time,
                          epoch_time * (n_epochs - epoch),
                          epoch_ae_loss_avg.result(),
                          #epoch_dc_loss_avg.result(),
                          #epoch_dc_acc_avg.result(),
                          #epoch_gen_loss_avg.result()
                          ))

        # Save models
        # encoder.save(bootstrap_model_dir / 'encoder.h5')
        # decoder.save(bootstrap_model_dir / 'decoder.h5')
        # encoder0.save(bootstrap_model_dir / 'encoder0.h5')
        # encoder1.save(bootstrap_model_dir / 'encoder1.h5')
        # encoder2.save(bootstrap_model_dir / 'encoder2.h5')
        # encoder3.save(bootstrap_model_dir / 'encoder3.h5')
        # decoder0.save(bootstrap_model_dir / 'decoder0.h5')
        # decoder1.save(bootstrap_model_dir / 'decoder1.h5')
        # decoder2.save(bootstrap_model_dir / 'decoder2.h5')
        # decoder3.save(bootstrap_model_dir / 'decoder3.h5')
        #discriminator.save(bootstrap_model_dir / 'discriminator.h5')


        # save the encoder list and decoder list
        for i in range(len(dataset_names)):
            encoder_list[i].save(bootstrap_model_dir / ('encoder' + str(i) + '.h5'))
            decoder_list[i].save(bootstrap_model_dir / ('decoder' + str(i) + '.h5'))

        # load the encoder list and decoder list
        encoder_list = [tf.keras.models.load_model(bootstrap_model_dir / ('encoder' + str(i) + '.h5')) for i in range(len(dataset_names))]
        decoder_list = [tf.keras.models.load_model(bootstrap_model_dir / ('decoder' + str(i) + '.h5')) for i in range(len(dataset_names))]

        # Save scaler
        joblib.dump(scaler_list, bootstrap_model_dir / 'scaler.pkl')
        joblib.dump(enc_age_list, bootstrap_model_dir / 'enc_age.pkl')
        joblib.dump(enc_gender_list, bootstrap_model_dir / 'enc_gender.pkl')


if __name__ == "__main__":

    argparse.ArgumentParser()
    # add dataset_name to parse
    parser = argparse.ArgumentParser(description='Train the supervised adversarial autoencoder.')
    parser.add_argument('-R', '--dataset_resourse',
                    dest='dataset_resourse',
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
    
    parser.add_argument('-P', '--procedure',
                        dest='procedure',
                        help='Procedure to perform the analysis.',
                        type=str)


    
    args = parser.parse_args()

    if args.dataset_resourse is None:
        args.dataset_resourse = 'ADNI'
    if args.hz_para_list is None:
        args.hz_para_list = [110, 110, 10]
    if args.base_lr is None:
        args.base_lr = 0.000001
    if args.max_lr is None:
        args.max_lr = 0.00005
    if args.procedure is None:
        args.procedure = 'SE-DC'


    main(args.dataset_resourse, args.hz_para_list, args.base_lr, args.max_lr, args.procedure)

#!/home/songlinzhao/anaconda3/envs/normodiff/bin/python3
"""Script to train the deterministic supervised adversarial autoencoder."""
from pathlib import Path
import random as rn
#import time
import pandas as pd
from sklearn.preprocessing import RobustScaler, OneHotEncoder
#import joblib
from sklearn.preprocessing import RobustScaler
import numpy as np
import tensorflow as tf

from utils import COLUMNS_NAME, load_dataset, COLUMNS_NAME_SNP, COLUMNS_NAME_VBM
#from models import make_encoder_model_v1, make_decoder_model_v1, make_discriminator_model_v1
from utils_vae import plot_losses, MyDataset_labels, Logger
import torch
from cVAE import cVAE

from os.path import join
from utils_vae import plot_losses, Logger, MyDataset
import argparse



PROJECT_ROOT = Path.cwd()


def main(dataset_name):
    """Train the normative method on the bootstrapped samples.

    The script also the scaler and the demographic data encoder.
    """
    # ----------------------------------------------------------------------------
    n_bootstrap = 10
    model_name = 'supervised_cvae'

    # Set the path of the participants file
    participants_path = PROJECT_ROOT / 'data' / 'y.csv'
    # dataset_name to str
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
        ids_filename = 'cleaned_bootstrap_{:03d}.csv'.format(i_bootstrap)
        ids_path = ids_dir / ids_filename

        bootstrap_model_dir = model_dir / '{:03d}'.format(i_bootstrap)
        bootstrap_model_dir.mkdir(exist_ok=True)

        # ----------------------------------------------------------------------------
        # Loading data
        dataset_df = load_dataset(participants_path, ids_path, freesurfer_path)

        # ----------------------------------------------------------------------------

        dataset_df = dataset_df.loc[dataset_df['DIA'] == 2]      
        train_data = dataset_df[COLUMNS_NAME].values
        

        tiv = dataset_df['PTEDUCAT'].values
        tiv = tiv[:, np.newaxis]

        train_data = (np.true_divide(train_data, tiv)).astype('float32')

        scaler = RobustScaler()
        train_data = scaler.fit_transform(train_data)
        train_data = pd.DataFrame(train_data)
        
        train_covariates = dataset_df[['DIA','PTGENDER', 'AGE']]
        train_covariates.DIA[train_covariates.DIA == 0] = 0       #
        train_covariates['ICV'] =tiv  #
        
        bin_labels = list(range(0,27))                          
        AGE_bins_train, bin_edges = pd.qcut(train_covariates['AGE'].rank(method="first"), q=27, retbins=True, labels=bin_labels)
        #AGE_bins_test = pd.cut(test_covariates['AGE'], bins=bin_edges, labels=bin_labels)
        #one_hot_AGE_test = np.eye(10)[AGE_bins_test.values]
        one_hot_AGE = np.eye(27)[AGE_bins_train.values]
        
        PTGENDER_bins_train, bin_edges = pd.qcut(train_covariates['PTGENDER'].rank(method="first"), q=2, retbins=True, labels=list(range(0,2)) )
        #AGE_bins_test = pd.cut(test_covariates['AGE'], bins=bin_edges, labels=bin_labels)
        #one_hot_AGE_test = np.eye(10)[AGE_bins_test.values]
        one_hot_PTGENDER = np.eye(2)[PTGENDER_bins_train.values]
        
        bin_labels = list(range(0,3))      
        ICV_bins_train, bin_edges = pd.qcut(train_covariates['ICV'], q=3,  retbins=True, labels=bin_labels, duplicates='drop')
        #ICV_bins_test = pd.cut(test_covariates['ICV'], bins=bin_edges, labels=bin_labels)
        #one_hot_ICV_test = np.eye(10)[ICV_bins_test.values]
        ICV_bins_train.fillna(0, inplace = True)
        one_hot_ICV_train = np.eye(10)[ICV_bins_train.values]
        # ----------------------------------------------------------------------------
        #AGE = dataset_df['AGE'].values[:, np.newaxis].astype('float32')
        #enc_AGE = OneHotEncoder(sparse=False)
        #enc_AGE = OneHotEncoder(handle_unknown = "ignore",sparse=False)
        #one_hot_AGE = enc_AGE.fit_transform(AGE)

        #PTGENDER = dataset_df['PTGENDER'].values[:, np.newaxis].astype('float32')
        #enc_PTGENDER = OneHotEncoder(sparse=False)
        #one_hot_PTGENDER = enc_PTGENDER.fit_transform(PTGENDER)

        #y_data = np.concatenate((one_hot_AGE, one_hot_PTGENDER), axis=1).astype('float32')

        # -------------------------------------------------------------------------------------------------------------
        # Create the dataset iterator
        batch_size = 256
        n_samples = train_data.shape[0]

        #train_dataset = tf.data.Dataset.from_tensor_slices((x_data_normalized, y_data))
        #train_dataset = train_dataset.shuffle(buffer_size=n_samples)
        #train_dataset = train_dataset.batch(batch_size)

        torch.manual_seed(42)
        use_cuda =  torch.cuda.is_available()
        if use_cuda:
            torch.cuda.manual_seed(42)
        DEVICE = torch.device("cuda:1" if use_cuda else "cpu")
        
        input_dim = train_data.shape[1]
        one_hot_covariates_train = np.concatenate((one_hot_AGE, one_hot_PTGENDER, one_hot_ICV_train), axis=1).astype('float32')
        c_dim = one_hot_covariates_train.shape[1]
        train_dataset = MyDataset_labels(train_data.to_numpy(), one_hot_covariates_train)    
        #train_dataset = MyDataset(train_data)
        generator_train = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=False)
        # -------------------------------------------------------------------------------------------------------------
        # Create models
        #n_features = x_data_normalized.shape[1]
        #n_labels = y_data.shape[1]
        h_dim = [100,100]
        z_dim = 20

        #encoder = make_encoder_model_v1(n_features, h_dim, z_dim)
        #decoder = make_decoder_model_v1(z_dim + n_labels, n_features, h_dim) 
        #discriminator = make_discriminator_model_v1(z_dim, h_dim)

        # -------------------------------------------------------------------------------------------------------------
        # Define loss functions
        #cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        #mse = tf.keras.losses.MeanSquaredError()
        #accuracy = tf.keras.metrics.BinaryAccuracy()

        #def discriminator_loss(real_output, fake_output):
        #    loss_real = cross_entropy(tf.ones_like(real_output), real_output)
        #    loss_fake = cross_entropy(tf.zeros_like(fake_output), fake_output)
        #    return loss_fake + loss_real

        #def generator_loss(fake_output):
        #    return cross_entropy(tf.ones_like(fake_output), fake_output)

        # -------------------------------------------------------------------------------------------------------------
        # Define optimizers
        #base_lr = 0.0001
        #max_lr = 0.005

        #step_size = 2 * np.ceil(n_samples / batch_size)

        #ae_optimizer = tf.keras.optimizers.Adam(lr=base_lr)
        #dc_optimizer = tf.keras.optimizers.Adam(lr=base_lr)
        #gen_optimizer = tf.keras.optimizers.Adam(lr=base_lr)

        # -------------------------------------------------------------------------------------------------------------
        # Training function
        #@tf.function
        #def train_step(batch_x, batch_y):
            # -------------------------------------------------------------------------------------------------------------
            # Autoencoder
            #with tf.GradientTape() as ae_tape:
            #    encoder_output = encoder(batch_x, training=True)
            #    decoder_output = decoder(tf.concat([encoder_output, batch_y], axis=1), training=True)

                # Autoencoder loss
            #    ae_loss = mse(batch_x, decoder_output)

            #ae_grads = ae_tape.gradient(ae_loss, encoder.trainable_variables + decoder.trainable_variables)
            #ae_optimizer.apply_gradients(zip(ae_grads, encoder.trainable_variables + decoder.trainable_variables))

            # -------------------------------------------------------------------------------------------------------------
            # Discriminator
            # with tf.GradientTape() as dc_tape:
            #     real_distribution = tf.random.normal([batch_x.shape[0], z_dim], mean=0.0, stddev=1.0)
            #     encoder_output = encoder(batch_x, training=True)

            #     dc_real = discriminator(real_distribution, training=True)
            #     dc_fake = discriminator(encoder_output, training=True)

            #     # Discriminator Loss
            #     dc_loss = discriminator_loss(dc_real, dc_fake)

            #     # Discriminator Acc
            #     dc_acc = accuracy(tf.concat([tf.ones_like(dc_real), tf.zeros_like(dc_fake)], axis=0),
            #                       tf.concat([dc_real, dc_fake], axis=0))

            # dc_grads = dc_tape.gradient(dc_loss, discriminator.trainable_variables)
            # dc_optimizer.apply_gradients(zip(dc_grads, discriminator.trainable_variables))

            # # -------------------------------------------------------------------------------------------------------------
            # # Generator (Encoder)
            # with tf.GradientTape() as gen_tape:
            #     encoder_output = encoder(batch_x, training=True)
            #     dc_fake = discriminator(encoder_output, training=True)

            #     # Generator loss
            #     gen_loss = generator_loss(dc_fake)

            # gen_grads = gen_tape.gradient(gen_loss, encoder.trainable_variables)
            # gen_optimizer.apply_gradients(zip(gen_grads, encoder.trainable_variables))

            #return ae_loss, dc_loss, dc_acc, gen_loss
            #return ae_loss

        # -------------------------------------------------------------------------------------------------------------
        # Training loop
        global_step = 0
        n_epochs = 200
        gamma = 0.98
        scale_fn = lambda x: gamma ** x
        base_lr = 0.0001
        max_lr = 0.005

        print('train model')
        model = cVAE(input_dim=input_dim, hidden_dim=h_dim, latent_dim=z_dim, c_dim=c_dim, learning_rate=0.0001, non_linear=True)
        model.to(DEVICE)
        
        step_size = 2 * np.ceil(n_samples / batch_size)
        
        for epoch in range(n_epochs): 
            for batch_idx, batch in enumerate(generator_train): 
                global_step = global_step + 1
                cycle = np.floor(1 + global_step / (2 * step_size))
                x_lr = np.abs(global_step / step_size - 2 * cycle + 1)
                clr = base_lr + (max_lr - base_lr) * max(0, 1 - x_lr) * scale_fn(cycle)
                model.optimizer = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=clr)
                data_curr = batch[0].to(DEVICE)
                cov = batch[1].to(DEVICE)
                fwd_rtn = model.forward(data_curr, cov)
                loss = model.loss_function(data_curr, fwd_rtn)
                model.optimizer.zero_grad()
                loss['total'].backward()
                model.optimizer.step() 
                if batch_idx == 0:
                    to_print = 'Train Epoch:' + str(epoch) + ' ' + 'Train batch: ' + str(batch_idx) + ' '+ ', '.join([k + ': ' + str(round(v.item(), 3)) for k, v in loss.items()])
                    print(to_print)        
                    if epoch == 0:
                        log_keys = list(loss.keys())
                        logger = Logger()
                        logger.on_train_init(log_keys)
                    else:
                        logger.on_step_fi(loss)
        plot_losses(logger, bootstrap_model_dir, 'training')
        model_path = join(bootstrap_model_dir, 'cVAE_model.pkl')
        torch.save(model, model_path)
            
            #start = time.time()

            #epoch_ae_loss_avg = tf.metrics.Mean()
            #epoch_dc_loss_avg = tf.metrics.Mean()
            #epoch_dc_acc_avg = tf.metrics.Mean()
            #epoch_gen_loss_avg = tf.metrics.Mean()

            #for _, (batch_x, batch_y) in enumerate(train_dataset):
            #    global_step = global_step + 1
            #    cycle = np.floor(1 + global_step / (2 * step_size))
            #    x_lr = np.abs(global_step / step_size - 2 * cycle + 1)
            #    clr = base_lr + (max_lr - base_lr) * max(0, 1 - x_lr) * scale_fn(cycle)
            #    ae_optimizer.lr = clr
                #dc_optimizer.lr = clr
                #gen_optimizer.lr = clr
                
                #batch_x = tf.concat([batch_x, batch_y], axis=1)
                #print(batch_x.shape, batch_y.shape)
                #ae_loss, dc_loss, dc_acc, gen_loss = train_step(batch_x, batch_y)
            #    ae_loss = train_step(batch_x, batch_y)

            #    epoch_ae_loss_avg(ae_loss)
                #epoch_dc_loss_avg(dc_loss)
                #epoch_dc_acc_avg(dc_acc)
                #epoch_gen_loss_avg(gen_loss)

            #epoch_time = time.time() - start

            #print('{:4d}: TIME: {:.2f} ETA: {:.2f} AE_LOSS: {:.4f} DC_LOSS: {:.4f} DC_ACC: {:.4f} GEN_LOSS: {:.4f}' \
            #print('{:4d}: TIME: {:.2f} ETA: {:.2f} AE_LOSS: {:.4f}' \
            #      .format(epoch, epoch_time,
            #              epoch_time * (n_epochs - epoch),
            #              epoch_ae_loss_avg.result(),
                          #epoch_dc_loss_avg.result(),
                          #epoch_dc_acc_avg.result(),
                          #epoch_gen_loss_avg.result()
            #              ))






        
        # Save models
        #encoder.save(bootstrap_model_dir / 'encoder.h5')
        #decoder.save(bootstrap_model_dir / 'decoder.h5')
        #discriminator.save(bootstrap_model_dir / 'discriminator.h5')

        # Save scaler
        #joblib.dump(scaler, bootstrap_model_dir / 'scaler.joblib')

        #joblib.dump(enc_AGE, bootstrap_model_dir / 'AGE_encoder.joblib')
        #joblib.dump(enc_PTGENDER, bootstrap_model_dir / 'PTGENDER_encoder.joblib')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-D', '--dataset_name',
                        dest='dataset_name',
                        help='Dataset to use for training test and evaluation.',
                        type=str)
    args = parser.parse_args()

    if args.dataset_name == 'snp':
        COLUMNS_NAME = COLUMNS_NAME_SNP
    elif args.dataset_name == 'vbm':
        COLUMNS_NAME = COLUMNS_NAME_VBM

    main(args.dataset_name)

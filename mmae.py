#!/home/songlinzhao/anaconda3/envs/normodiff/bin/python3
"""Unified script for training, testing, and group analysis using the supervised AE model."""

import argparse
from pathlib import Path
import random as rn
import time
import joblib
from sklearn.preprocessing import RobustScaler, OneHotEncoder
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from os.path import join, exists
from tqdm import tqdm
from utils import COLUMNS_HCP, COLUMNS_NAME, load_dataset, COLUMNS_NAME_SNP, COLUMNS_NAME_VBM, COLUMNS_3MODALITIES, cliff_delta
from models import make_encoder_model_v111, make_decoder_model_v1, make_discriminator_model_v1
from scipy import stats
from sklearn.metrics import roc_curve, auc
from numpy import interp, linspace
import matplotlib.pyplot as plt
import joblib

PROJECT_ROOT = Path.cwd()
result_baseline = './result_baseline/'

# Training function
def train(dataset_resourse, hz_para_list, base_lr, max_lr, procedure, epochs):
    """Train the supervised AE model on the bootstrapped samples."""
    n_bootstrap = 10
    model_name = 'supervised_ae'
    bootstrap_dir = PROJECT_ROOT / 'outputs' / 'bootstrap_analysis'
    ids_dir = bootstrap_dir / 'ids'
    model_dir = bootstrap_dir / model_name
    model_dir.mkdir(exist_ok=True)

    random_seed = 42
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)
    rn.seed(random_seed)

    for i_bootstrap in range(n_bootstrap):
        print('Bootstrap iteration: {:03d}'.format(i_bootstrap))
        if dataset_resourse == 'ADNI':
            dataset_names = ['av45', 'fdg', 'vbm']
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
                if dataset_name in ['av45', 'fdg']:
                    columns_name = COLUMNS_NAME
                elif dataset_name == 'snp':
                    columns_name = COLUMNS_NAME_SNP
                elif dataset_name == 'vbm':
                    columns_name = COLUMNS_NAME_VBM
                elif dataset_name == '3modalities':
                    columns_name = COLUMNS_3MODALITIES
            elif dataset_resourse == 'HCP':
                columns_name = [(lambda x: dataset_name + '_' + str(x))(y) for y in range(132)]

            participants_path = PROJECT_ROOT / 'data' / dataset_resourse / 'y.csv'
            freesurfer_path = PROJECT_ROOT / 'data' / dataset_resourse / (dataset_name + '.csv')
            ids_filename = 'cleaned_bootstrap_{:03d}.csv'.format(i_bootstrap)
            ids_path = ids_dir / ids_filename
            bootstrap_model_dir = model_dir / '{:03d}'.format(i_bootstrap)
            bootstrap_model_dir.mkdir(exist_ok=True)

            dataset_df = load_dataset(participants_path, ids_path, freesurfer_path)
            if dataset_resourse == 'ADNI':
                hc_label = 2
            elif dataset_resourse == 'HCP':
                hc_label = 1
            else:
                raise ValueError('Unknown dataset resource')

            dataset_df = dataset_df.loc[dataset_df['DIA'] == hc_label]
            x_data = dataset_df[columns_name].values
            scaler = RobustScaler()
            x_data_normalized = scaler.fit_transform(x_data)

            age = dataset_df['AGE'].values[:, np.newaxis].astype('float32')
            enc_age = OneHotEncoder(handle_unknown="ignore", sparse=False)
            one_hot_age2 = enc_age.fit_transform(age)
            bin_labels = list(range(0, 10))
            age_bins_test, bin_edges = pd.cut(dataset_df['AGE'], 10, retbins=True, labels=bin_labels)
            age_bins_test.fillna(0, inplace=True)
            one_hot_age = np.eye(10)[age_bins_test.values]

            gender = dataset_df['PTGENDER'].values[:, np.newaxis].astype('bool')
            enc_gender = OneHotEncoder(sparse=False)
            one_hot_gender = enc_gender.fit_transform(gender)

            y_data = np.concatenate((one_hot_age, one_hot_gender), axis=1).astype('float32')
            batch_size = 256
            n_samples = x_data.shape[0]
            train_dataset = tf.data.Dataset.from_tensor_slices((x_data_normalized, y_data))
            train_dataset = train_dataset.shuffle(buffer_size=n_samples)
            train_dataset = train_dataset.batch(batch_size)
            train_datasets.append(train_dataset)
            n_features = x_data_normalized.shape[1]
            n_features_list.append(n_features)
            scaler_list.append(scaler)
            enc_age_list.append(enc_age)
            enc_gender_list.append(enc_gender)

        h_dim = hz_para_list[:-1]
        z_dim = hz_para_list[-1]
        encoder_list = [make_encoder_model_v111(n_features_list[i], h_dim, z_dim) for i in range(len(dataset_names))]
        z_dim = z_dim * len(dataset_names)
        decoder_list = [make_decoder_model_v1(z_dim, n_features_list[i], h_dim) for i in range(len(dataset_names))]
        # cross_attention_latent = CrossAttentionLatent(input_dim=hz_para_list[-1], hidden_dim=hz_para_list[-1])

        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        mse = tf.keras.losses.MeanSquaredError()
        accuracy = tf.keras.metrics.BinaryAccuracy()

        step_size = 2 * np.ceil(n_samples / batch_size)
        ae_optimizer = tf.keras.optimizers.Adam(lr=base_lr)

        @tf.function
        def train_step_multimodal(batch_x_list, batch_y_list):
            with tf.GradientTape() as ae_tape:
                encoder_output_list = [encoder_list[i](batch_x_list[i], training=True) for i in range(len(dataset_names))]
                if procedure == 'SE-DC':
                    encoder_output_multimodal = tf.concat(encoder_output_list, axis=1)
                elif procedure == 'SE-CA':
                    fused_output1, fused_output2, fused_output3 = cross_attention_latent(encoder_output_list[0], encoder_output_list[1], encoder_output_list[2])
                    encoder_output_multimodal = tf.concat([fused_output1, fused_output2, fused_output3], axis=1)
                elif procedure == 'SE-Mixture':
                    encoder_output_mixture = tf.reduce_sum(encoder_output_list, axis=0)
                    encoder_output_multimodal = tf.concat([encoder_output_mixture for i in range(len(dataset_names))], axis=1)
                decoder_output_list = [decoder_list[i](encoder_output_multimodal, training=True) for i in range(len(dataset_names))]
                ae_loss_list = [mse(batch_x_list[i], decoder_output_list[i]) for i in range(len(dataset_names))]
                ae_loss = tf.reduce_sum(ae_loss_list)
                trainable_variables = []
                for i in range(len(dataset_names)):
                    trainable_variables.extend(encoder_list[i].trainable_variables)
                    trainable_variables.extend(decoder_list[i].trainable_variables)
                ae_grads = ae_tape.gradient(ae_loss, trainable_variables)
                ae_grads = [grad if grad is not None else tf.zeros_like(var) for grad, var in zip(ae_grads, trainable_variables)]
                ae_optimizer.apply_gradients(zip(ae_grads, trainable_variables))
            return ae_loss

        global_step = 0
        n_epochs = epochs
        gamma = 0.98
        scale_fn = lambda x: gamma ** x
        for epoch in range(n_epochs):
            start = time.time()
            epoch_ae_loss_avg = tf.metrics.Mean()
            for batches in zip(*train_datasets):
                batch_x_list = [batch[0] for batch in batches]
                batch_y_list = [batch[1] for batch in batches]
                global_step = global_step + 1
                cycle = np.floor(1 + global_step / (2 * step_size))
                x_lr = np.abs(global_step / step_size - 2 * cycle + 1)
                clr = base_lr + (max_lr - base_lr) * max(0, 1 - x_lr) * scale_fn(cycle)
                ae_optimizer.lr = clr
                ae_loss = train_step_multimodal(batch_x_list, batch_y_list)
                epoch_ae_loss_avg(ae_loss)
            epoch_time = time.time() - start
            print('{:4d}: TIME: {:.2f} ETA: {:.2f} AE_LOSS: {:.4f}'.format(epoch, epoch_time, epoch_time * (n_epochs - epoch), epoch_ae_loss_avg.result()))

        for i in range(len(dataset_names)):
            encoder_list[i].save(bootstrap_model_dir / ('encoder' + str(i) + '.h5'))
            decoder_list[i].save(bootstrap_model_dir / ('decoder' + str(i) + '.h5'))
        joblib.dump(scaler_list, bootstrap_model_dir / 'scaler.pkl')
        joblib.dump(enc_age_list, bootstrap_model_dir / 'enc_age.pkl')
        joblib.dump(enc_gender_list, bootstrap_model_dir / 'enc_gender.pkl')

# Testing function
def test(dataset_resourse):
    """Make predictions using trained normative models."""
    n_bootstrap = 10
    model_name = 'supervised_ae'
    participants_path = PROJECT_ROOT / 'data' / dataset_resourse / 'y.csv'
    bootstrap_dir = PROJECT_ROOT / 'outputs' / 'bootstrap_analysis'
    model_dir = bootstrap_dir / model_name
    random_seed = 42
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)

    boostrap_error_mean = []

    for i_bootstrap in tqdm(range(n_bootstrap)):
        if dataset_resourse == 'ADNI':
            dataset_names = ['av45', 'fdg', 'vbm']
        elif dataset_resourse == 'HCP':
            dataset_names = ['T1_volume', 'mean_T1_intensity', 'mean_FA', 'mean_MD', 'mean_L1', 'mean_L2', 'mean_L3', 'min_BOLD', '25_percentile_BOLD', '50_percentile_BOLD', '75_percentile_BOLD', 'max_BOLD']
        else:
            raise ValueError('Unknown dataset resource')

        x_normalized_list = []
        n_features_list = []
        encoder_output_list = []
        clinical_df_list = []
        bootstrap_model_dir = model_dir / '{:03d}'.format(i_bootstrap)
        ids_dir = bootstrap_dir / 'ids'
        test_ids_filename = 'cleaned_bootstrap_test_{:03d}.csv'.format(i_bootstrap)
        ids_path = ids_dir / test_ids_filename
        scaler_list = joblib.load(bootstrap_model_dir / 'scaler.pkl')
        enc_age_list = joblib.load(bootstrap_model_dir / 'enc_age.pkl')
        enc_gender_list = joblib.load(bootstrap_model_dir / 'enc_gender.pkl')

        for dataset_name in dataset_names:
            if dataset_resourse == 'ADNI':
                if dataset_name in ['av45', 'fdg']:
                    columns_name = COLUMNS_NAME
                elif dataset_name == 'snp':
                    columns_name = COLUMNS_NAME_SNP
                elif dataset_name == 'vbm':
                    columns_name = COLUMNS_NAME_VBM
                elif dataset_name == '3modalities':
                    columns_name = COLUMNS_3MODALITIES
            elif dataset_resourse == 'HCP':
                columns_name = [(lambda x: dataset_name + '_' + str(x))(y) for y in range(132)]
            participants_path = PROJECT_ROOT / 'data' / dataset_resourse / 'y.csv'
            freesurfer_path = PROJECT_ROOT / 'data' / dataset_resourse / (dataset_name + '.csv')
            output_dataset_dir = bootstrap_model_dir / dataset_name
            output_dataset_dir.mkdir(exist_ok=True)
            clinical_df = load_dataset(participants_path, ids_path, freesurfer_path)
            clinical_df_list.append(clinical_df)
            x_dataset = clinical_df[columns_name].values
            scaler = scaler_list[dataset_names.index(dataset_name)]
            enc_age = enc_age_list[dataset_names.index(dataset_name)]
            enc_gender = enc_gender_list[dataset_names.index(dataset_name)]
            age = clinical_df['AGE'].values[:, np.newaxis].astype('float32')
            one_hot_age = enc_age.transform(age)
            bin_labels = list(range(0, 10))
            age_bins_test, bin_edges = pd.cut(clinical_df['AGE'], 10, retbins=True, labels=bin_labels)
            age_bins_test.fillna(0, inplace=True)
            one_hot_age = np.eye(10)[age_bins_test.values]
            gender = clinical_df['PTGENDER'].values[:, np.newaxis].astype('bool')
            one_hot_gender = enc_gender.transform(gender)
            y_data = np.concatenate((one_hot_age, one_hot_gender), axis=1).astype('float32')
            x_normalized = scaler.transform(x_dataset)
            normalized_df = pd.DataFrame(columns=['participant_id'] + columns_name)
            normalized_df['participant_id'] = clinical_df['participant_id']
            normalized_df[columns_name] = x_normalized
            normalized_df.to_csv(output_dataset_dir / 'normalized_{}.csv'.format(dataset_name), index=False)
            x_normalized_list.append(x_normalized)

        encoder_list = [tf.keras.models.load_model(bootstrap_model_dir / ('encoder' + str(i) + '.h5')) for i in range(len(dataset_names))]
        decoder_list = [tf.keras.models.load_model(bootstrap_model_dir / ('decoder' + str(i) + '.h5')) for i in range(len(dataset_names))]
        for i_dataset in range(len(dataset_names)):
            encoder_output_list.append(encoder_list[i_dataset](x_normalized_list[i_dataset], training=False))

        encoder_output_multimodal = tf.concat(encoder_output_list, axis=1)
        reconstruction_df_list = []
        for i_dataset in range(len(dataset_names)):
            dataset_name = dataset_names[i_dataset]
            if dataset_resourse == 'ADNI':
                if dataset_name in ['av45', 'fdg']:
                    columns_name = COLUMNS_NAME
                elif dataset_name == 'snp':
                    columns_name = COLUMNS_NAME_SNP
                elif dataset_name == 'vbm':
                    columns_name = COLUMNS_NAME_VBM
                elif dataset_name == '3modalities':
                    columns_name = COLUMNS_3MODALITIES
            elif dataset_resourse == 'HCP':
                columns_name = [(lambda x: dataset_name + '_' + str(x))(y) for y in range(132)]
            output_dataset_dir = bootstrap_model_dir / dataset_name
            output_dataset_dir.mkdir(exist_ok=True)
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

# Group analysis function
def group_analysis(dataset_resourse, hz_para_list, hc_label, disease_label, procedure, epochs):
    """Perform the group analysis."""
    n_bootstrap = 10
    model_name = 'supervised_ae'
    bootstrap_dir = PROJECT_ROOT / 'outputs' / 'bootstrap_analysis'
    model_dir = bootstrap_dir / model_name
    ids_dir = bootstrap_dir / 'ids'

    tpr_list = []
    auc_roc_list = []
    accuracy_list = []
    sensitivity_list = []
    specificity_list = []
    significance_ratio_list = []

    for i_bootstrap in tqdm(range(n_bootstrap)):
        bootstrap_model_dir = model_dir / '{:03d}'.format(i_bootstrap)
        if dataset_resourse == 'ADNI':
            dataset_names = ['av45', 'fdg', 'vbm']
        elif dataset_resourse == 'HCP':
            dataset_names = ['T1_volume', 'mean_T1_intensity', 'mean_FA', 'mean_MD', 'mean_L1', 'mean_L2', 'mean_L3', 'min_BOLD', '25_percentile_BOLD', '50_percentile_BOLD', '75_percentile_BOLD', 'max_BOLD']
        else:
            raise ValueError('Unknown dataset resource')

        reconstruction_error_df_list = []
        for i_dataset, dataset_name in enumerate(dataset_names):
            output_dataset_dir = bootstrap_model_dir / dataset_name
            output_dataset_dir.mkdir(exist_ok=True)
            analysis_dir = output_dataset_dir / '{:02d}_vs_{:02d}'.format(hc_label, disease_label)
            analysis_dir.mkdir(exist_ok=True)
            test_ids_filename = 'cleaned_bootstrap_test_{:03d}.csv'.format(i_bootstrap)
            ids_path = ids_dir / test_ids_filename
            participants_path = PROJECT_ROOT / 'data' / dataset_resourse / 'y.csv'
            freesurfer_path = PROJECT_ROOT / 'data' / dataset_resourse / (dataset_name + '.csv')
            clinical_df = load_dataset(participants_path, ids_path, freesurfer_path)
            clinical_df = clinical_df.set_index('participant_id')

            if dataset_resourse == 'ADNI':
                if dataset_name in ['av45', 'fdg']:
                    columns_name = COLUMNS_NAME
                elif dataset_name == 'snp':
                    columns_name = COLUMNS_NAME_SNP
                elif dataset_name == 'vbm':
                    columns_name = COLUMNS_NAME_VBM
                elif dataset_name == '3modalities':
                    columns_name = COLUMNS_3MODALITIES
            elif dataset_resourse == 'HCP':
                columns_name = [(lambda x: dataset_name + '_' + str(x))(y) for y in range(132)]

            normalized_df = pd.read_csv(output_dataset_dir / 'normalized_{}.csv'.format(dataset_name), index_col='participant_id')
            reconstruction_df = pd.read_csv(output_dataset_dir / 'reconstruction_{}.csv'.format(dataset_name), index_col='participant_id')
            reconstruction_error_df = pd.read_csv(output_dataset_dir / 'reconstruction_error_{}.csv'.format(dataset_name), index_col='participant_id')
            reconstruction_error_df_list.append(reconstruction_error_df)

        reconstruction_error_df_averaged = reconstruction_error_df_list[0]
        for i in range(1, len(reconstruction_error_df_list)):
            reconstruction_error_df_averaged += reconstruction_error_df_list[i]
        reconstruction_error_df_averaged /= len(reconstruction_error_df_list)

        # roc_auc, tpr, accuracy, accuracy_in_hc, accuracy_in_ad, recall, specificity, significance_ratio = compute_classification_performance(reconstruction_error_df_averaged, clinical_df, hc_label, disease_label)
        roc_auc, accuracy, recall, specificity, significance_ratio = compute_classification_performance(reconstruction_error_df_averaged, clinical_df, hc_label, disease_label)
        auc_roc_list.append(roc_auc)
        accuracy_list.append(accuracy)
        sensitivity_list.append(recall)
        specificity_list.append(specificity)
        significance_ratio_list.append(significance_ratio)

    comparison_dir = bootstrap_dir / dataset_name / ('{:02d}_vs_{:02d}'.format(hc_label, disease_label))
    comparison_dir.mkdir(exist_ok=True)

    auc_roc_list = np.array(auc_roc_list)
    accuracy_list = np.array(accuracy_list)
    sensitivity_list = np.array(sensitivity_list)
    specificity_list = np.array(specificity_list)
    significance_ratio_list = auc_roc_list / (1 - auc_roc_list)

    if hc_label == 2 and disease_label == 0:
        compare_name = 'HC_vs_AD'
    elif hc_label == 2 and disease_label == 1:
        compare_name = 'HC_vs_MCI'
    elif hc_label == 1 and disease_label == 0:
        compare_name = 'MCI_vs_AD'

    with open(result_baseline + 'mm.txt', 'a') as f:
        f.write(f'Experiment settings: AE. Data resource: {dataset_resourse}. Procedure: {procedure}. compare_name: {compare_name}. Epochs: {epochs}\n')
        f.write('ROC-AUC: $ {:0.2f} \pm {:0.2f} $ \n'.format(np.mean(auc_roc_list) * 100, np.std(auc_roc_list) * 100))
        f.write('Accuracy: $ {:0.2f} \pm {:0.2f} $ \n'.format(np.mean(accuracy_list) * 100, np.std(accuracy_list) * 100))
        f.write('Sensitivity: $ {:0.2f} \pm {:0.2f} $ \n'.format(np.mean(sensitivity_list) * 100, np.std(sensitivity_list) * 100))
        f.write('Specificity: $ {:0.2f} \pm {:0.2f} $ \n'.format(np.mean(specificity_list) * 100, np.std(specificity_list) * 100))
        f.write('Significance ratio: $ {:0.2f} \pm {:0.2f} $ \n'.format(np.mean(significance_ratio_list), np.std(significance_ratio_list)))
        f.write('hz_para_list: ' + str(hz_para_list) + '\n')
        f.write('\n\n\n')
    np.savetxt("ae_auc_and_std.csv", np.concatenate((auc_roc_list, [np.std(auc_roc_list)])), delimiter=",")
    auc_roc_df = pd.DataFrame(columns=['ROC-AUC'], data=auc_roc_list)
    auc_roc_df.to_csv(comparison_dir / 'auc_rocs.csv', index=False)

def compute_classification_performance(reconstruction_error_df, clinical_df, hc_label, disease_label):
    """Calculate the AUCs and accuracy of the normative model."""
    error_hc = reconstruction_error_df.loc[clinical_df['DIA'] == hc_label]['Reconstruction error']
    error_patient = reconstruction_error_df.loc[clinical_df['DIA'] == disease_label]['Reconstruction error']
    labels = list(np.zeros_like(error_hc)) + list(np.ones_like(error_patient))
    predictions = list(error_hc) + list(error_patient)
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    predicted_labels = [1 if e > optimal_threshold else 0 for e in predictions]
    accuracy = (np.array(predicted_labels) == np.array(labels)).mean()
    recall = np.sum((np.array(predicted_labels) == 1) & (np.array(labels) == 1)) / np.sum(np.array(labels) == 1)
    specificity = np.sum((np.array(predicted_labels) == 0) & (np.array(labels) == 0)) / np.sum(np.array(labels) == 0)
    significance_ratio = roc_auc / (1 - roc_auc)
    return roc_auc, accuracy, recall, specificity, significance_ratio

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-R', '--dataset_resourse', dest='dataset_resourse', help='Dataset to use for training test and evaluation.', type=str)
    parser.add_argument('-H', '--hz_para_list', dest='hz_para_list', nargs='+', help='List of paras to perform the analysis.', type=int)
    parser.add_argument('-B', '--base_lr', dest='base_lr', help='Base learning rate.', type=float)
    parser.add_argument('-M', '--max_lr', dest='max_lr', help='Max learning rate.', type=float)
    parser.add_argument('-P', '--procedure', dest='procedure', help='Procedure to perform the analysis.', type=str)
    parser.add_argument('-E', '--epochs', dest='epochs', help='Number of epochs to train the model.', type=int)

    args = parser.parse_args()

    if args.dataset_resourse is None:
        args.dataset_resourse = 'ADNI'
    if args.hz_para_list is None:
        args.hz_para_list = [110, 110, 10]
    if args.base_lr is None:
        args.base_lr = 0.001
    if args.max_lr is None:
        args.max_lr = 0.05
    if args.procedure is None:
        args.procedure = 'SE-DC'
    if args.epochs is None:
        args.epochs = 200

    # train(args.dataset_resourse, args.hz_para_list, args.base_lr, args.max_lr, args.procedure, args.epochs)
    # test(args.dataset_resourse)
    group_analysis(args.dataset_resourse, args.hz_para_list, 2, 0, args.procedure, args.epochs)
    group_analysis(args.dataset_resourse, args.hz_para_list, 2, 1, args.procedure, args.epochs)
    group_analysis(args.dataset_resourse, args.hz_para_list, 1, 0, args.procedure, args.epochs)

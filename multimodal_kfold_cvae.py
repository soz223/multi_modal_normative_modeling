#!/usr/bin/env python3
"""
Script to train, test, and analyze the deterministic supervised adversarial autoencoder with an integrated MLP for diagnosis.
"""

import argparse
import os
import random as rn
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler

from torch import nn, optim
from torch.utils.data import Dataset

# Import your own modules (ensure these are available in your environment)
from cVAE import Encoder, Decoder, ProductOfExperts, MixtureOfExperts, MoPoE
from utils import (COLUMNS_3MODALITIES, COLUMNS_HCP, COLUMNS_NAME,
                   COLUMNS_NAME_SNP, COLUMNS_NAME_VBM, load_dataset)
from utils_vae import (Logger, MyDataset_labels, plot_losses)

# Set global constants
PROJECT_ROOT = Path.cwd()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class cVAE_multimodal_endtoend(nn.Module):
    def __init__(self, 
                 input_dim_list, 
                 hidden_dim, 
                 latent_dim,
                 c_dim, 
                 learning_rate=0.01, 
                 modalities=3,
                 non_linear=False):
        
        super().__init__()
        self.input_dim_list = input_dim_list
        self.hidden_dim = hidden_dim + [latent_dim]
        self.latent_dim = latent_dim
        self.c_dim = c_dim
        self.modalities = modalities
        self.learning_rate = learning_rate

        # Initialize encoders and decoders for each modality
        self.encoder_list = nn.ModuleList([
            Encoder(input_dim=input_dim_list[i], hidden_dim=self.hidden_dim, c_dim=c_dim, non_linear=non_linear) 
            for i in range(modalities)
        ])
        self.decoder_list = nn.ModuleList([
            Decoder(input_dim=input_dim_list[i], hidden_dim=self.hidden_dim, c_dim=c_dim, non_linear=non_linear) 
            for i in range(modalities)
        ])

        # Parameters for combining modalities
        self.alpha_m_list = nn.ParameterList([
            nn.Parameter(torch.randn(1, requires_grad=True)) 
            for _ in range(modalities)
        ])

        # Calculate total input size for the MLP (sum of ROIs from all modalities)
        self.total_input_size = sum(self.input_dim_list)

        # Add MLP for diagnosis, mlp is just the equal of input
        self.mlp = nn.Sequential(
            nn.Linear(self.total_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Assuming binary classification
            nn.Sigmoid()
        )

        # Include all parameters in optimizer
        # self.optimizer1 = optim.Adam(
        #     list(self.parameters()),
        #     lr=self.learning_rate
        # )

        # do not include the parameters of the MLP in the optimizer
        self.optimizer1 = optim.Adam(
            [p for model in self.encoder_list for p in model.parameters()] +
            [p for model in self.decoder_list for p in model.parameters()] +
            list(self.alpha_m_list.parameters()),  
            lr=self.learning_rate
        )
        self.criterion = nn.BCELoss()  # For binary classification

    def product_of_experts(self, mus, variances):
        return ProductOfExperts()(mus, variances)
    
    def mixture_of_experts(self, mus, variances):
        return MixtureOfExperts()(mus, variances)
    
    def mixture_of_product_of_experts(self, mus, variances):
        return MoPoE()(mus, variances)

    def encode(self, x, c, m):
        return self.encoder_list[m](x, c)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(mu)
        return mu + eps*std

    def decode(self, z, c, m):
        return self.decoder_list[m](z, c)

    def calc_kl(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean(0)
    
    def calc_ll(self, x, x_recon):
        x_recon_mean = x_recon.loc
        recon_loss = nn.MSELoss(reduction='mean')(x_recon_mean, x)
        return -recon_loss  # Negative because we minimize the negative log-likelihood

    def combine_latent(self, mus, variances, combine):
        combine = combine.lower()
        if combine == 'poe':
            mu_multimodal, variance_multimodal = self.product_of_experts(mus, variances)
        elif combine == 'gpoe':
            alpha_m = torch.softmax(torch.stack([param for param in self.alpha_m_list]), dim=0).reshape(self.modalities, 1, 1)
            mu_multimodal = torch.sum(mus * alpha_m / variances, dim=0) / torch.sum(alpha_m / variances, dim=0)
            variance_multimodal = 1 / torch.sum(alpha_m / variances, dim=0)
        elif combine == 'moe':
            mu_multimodal, variance_multimodal = self.mixture_of_experts(mus, variances)
        elif combine == 'mopoe':
            mu_multimodal, variance_multimodal = self.mixture_of_product_of_experts(mus, variances)
        else:
            raise ValueError('No such combination method')
        return mu_multimodal, variance_multimodal

    def forward_multimodal(self, xes, cs, combine):
        self.zero_grad()
        xes = [x.type(torch.float32) for x in xes]
        cs = [c.type(torch.float32) for c in cs]

        # Encode each modality
        mus_all, logvars_all = zip(*[self.encode(xes[i], cs[i], i) for i in range(self.modalities)])
        mus, logvars = torch.stack(mus_all), torch.stack(logvars_all)
        variances = torch.exp(logvars)

        # Combine latent representations
        mu_multimodal, variance_multimodal = self.combine_latent(mus, variances, combine)
        logvar_multimodal = torch.log(variance_multimodal)
        z_multimodal = self.reparameterise(mu_multimodal, logvar_multimodal)

        # Decode to reconstruct inputs
        x_recons = [self.decode(z_multimodal, cs[i], i) for i in range(self.modalities)]
        
        # Compute reconstruction deviations per ROI (without averaging over features)
        recon_errors = [ (xes[i] - x_recons[i].loc)**2 for i in range(self.modalities) ]
        # Concatenate reconstruction errors from all modalities along the feature dimension
        recon_errors = torch.cat(recon_errors, dim=1)  # Shape: (batch_size, total_num_ROIs)

        # Pass reconstruction deviations through MLP to get diagnosis
        # diagnosis = self.mlp(recon_errors)

        return {
            'x_recons': x_recons,
            'mu_multimodal': mu_multimodal,
            'logvar_multimodal': logvar_multimodal,
            'recon_errors': recon_errors,
            # 'diagnosis': diagnosis
        }

    def loss_function_multimodal(self, xes, fwd_rtn, labels):
        losses = {'total': 0, 'kl': 0, 'll': 0}
        for i in range(self.modalities):
            kl = self.calc_kl(fwd_rtn['mu_multimodal'], fwd_rtn['logvar_multimodal'])
            recon = self.calc_ll(xes[i], fwd_rtn['x_recons'][i])
            total = kl - recon
            losses['total'] += total
            losses['kl'] += kl
            losses['ll'] += recon
            # losses['classification_loss'] += self.criterion(fwd_rtn['diagnosis'], labels.view(-1, 1))
        
        return losses

    # def loss_function_multimodal(self, xes, fwd_rtn, labels):
    #     losses = {'total': 0, 'kl': 0, 'll': 0, 'classification_loss': 0}
    #     kl = self.calc_kl(fwd_rtn['mu_multimodal'], fwd_rtn['logvar_multimodal'])
    #     recon_loss = 0
    #     for i in range(self.modalities):
    #         recon = self.calc_ll(xes[i], fwd_rtn['x_recons'][i])
    #         recon_loss += recon
    #     recon_loss /= self.modalities  # Average reconstruction loss
    #     classification_loss = self.criterion(fwd_rtn['diagnosis'], labels.view(-1, 1))

    #     # total_loss = kl - recon_loss + classification_loss
    #     total_loss = (kl + recon_loss) * self.modalities

    #     losses['total'] = total_loss
    #     losses['kl'] = kl
    #     losses['ll'] = recon_loss
    #     losses['classification_loss'] = classification_loss

    #     return losses

    def pred_recon(self, xes, c, DEVICE, combine):
        with torch.no_grad():
            tensor_c = torch.tensor(c, dtype=torch.float32).to(DEVICE)
            xes_tensors = [torch.tensor(xes[i].values, dtype=torch.float32).to(DEVICE) for i in range(self.modalities)]
            mus_all, logvars_all = zip(*[self.encode(xes_tensors[i], tensor_c, i) for i in range(self.modalities)])
            mus, logvars = torch.stack(mus_all), torch.stack(logvars_all)
            variances = torch.exp(logvars)

            mu_multimodal, variance_multimodal = self.combine_latent(mus, variances, combine)
            logvar_multimodal = torch.log(variance_multimodal)
            z_multimodal = self.reparameterise(mu_multimodal, logvar_multimodal)
            x_recons = [self.decode(z_multimodal, tensor_c, i) for i in range(self.modalities)]

            # Compute reconstruction deviations per ROI
            recon_errors = [ (xes_tensors[i] - x_recons[i].loc)**2 for i in range(self.modalities) ]
            # Concatenate reconstruction errors from all modalities along the feature dimension
            recon_errors = torch.cat(recon_errors, dim=1)  # Shape: (batch_size, total_num_ROIs)

            # Pass reconstruction deviations through MLP to get diagnosis
            # diagnosis = self.mlp(recon_errors)

            return [x_recon.loc.cpu().detach().numpy() for x_recon in x_recons]

    def reconstruction_deviation_multimodal(self, xes, x_preds):
        deviations = []
        for m in range(self.modalities):
            deviation = np.sum((xes[m] - x_preds[m])**2, axis=1)/xes[m].shape[1]
            deviations.append(deviation)
        return deviations

# Dataset class that includes labels
class MyDataset_labels(Dataset):
    def __init__(self, data, covariates, labels):
        self.data = data.astype(np.float32)
        self.covariates = covariates.astype(np.float32)
        self.labels = labels.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = torch.from_numpy(self.data[idx])
        covariates = torch.from_numpy(self.covariates[idx])
        label = torch.tensor(self.labels[idx])
        return data, covariates, label


# The following are the implementations of train, test, and analyze functions with necessary adjustments

def generate_kfold_ids(HC_group, other_group, oversample_percentage=1, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    kfold_dir = PROJECT_ROOT / 'outputs' / 'kfold_analysis'
    kfold_dir.mkdir(parents=True, exist_ok=True)

    for fold, (train_idx, test_idx) in enumerate(kf.split(HC_group)):
        train_ids = HC_group.iloc[train_idx]['IID']
        test_ids_hc = HC_group.iloc[test_idx]['IID']

        oversample_size = int(len(train_ids) * oversample_percentage)
        train_ids_oversampled = np.random.choice(train_ids, size=oversample_size, replace=True)
        train_ids = pd.DataFrame({'IID': train_ids_oversampled})

        test_ids_other = other_group['IID']
        test_ids = pd.concat([test_ids_hc, test_ids_other])

        train_ids_path = kfold_dir / f'train_ids_{fold:03d}.csv'
        test_ids_path = kfold_dir / f'test_ids_{fold:03d}.csv'

        train_ids.to_csv(train_ids_path, index=False)
        test_ids.to_csv(test_ids_path, index=False)

def train(args):
    """Train the normative method using k-fold cross-validation."""
    n_splits = args.n_splits
    oversample_percentage = args.oversample_percentage
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    model_name = 'supervised_cvae'

    output_dir = PROJECT_ROOT / 'outputs'
    output_dir.mkdir(exist_ok=True)
    kfold_dir = output_dir / 'kfold_analysis'
    kfold_dir.mkdir(exist_ok=True)
    model_dir = kfold_dir / model_name
    model_dir.mkdir(exist_ok=True)

    # Set random seed
    random_seed = 42
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)
    rn.seed(random_seed)

    dataset_resourse = args.dataset_resourse
    procedure = args.procedure
    combine = args.combine
    hz_para_list = args.hz_para_list
    epochs = args.epochs

    if dataset_resourse == 'ADNI':
        if procedure.startswith('SE'):
            dataset_names = ['av45', 'vbm', 'fdg']
        elif procedure.startswith('UCA'):
            dataset_names = ['av45', 'vbm', 'fdg', '3modalities']
        else:
            raise ValueError('Unknown procedure: {}'.format(procedure))
    elif dataset_resourse == 'HCP':
        dataset_names = ['T1_volume', 'mean_T1_intensity', 'mean_FA', 'mean_MD',
                         'mean_L1', 'mean_L2', 'mean_L3', 'min_BOLD',
                         '25_percentile_BOLD', '50_percentile_BOLD', '75_percentile_BOLD',
                         'max_BOLD']
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset_resourse))

    modalities = len(dataset_names)

    participants_path = PROJECT_ROOT / 'data' / dataset_resourse / 'y.csv'
    ids_df = pd.read_csv(participants_path)

    hc_label = 2 if dataset_resourse == 'ADNI' else 1

    HC_group = ids_df[ids_df['DIA'] == hc_label]
    other_group = ids_df[ids_df['DIA'] == 0]

    generate_kfold_ids(HC_group, other_group, oversample_percentage=oversample_percentage, n_splits=n_splits)

    for fold, (train_idx, test_idx) in enumerate(kf.split(HC_group)):
        train_ids_path = kfold_dir / f'train_ids_{fold:03d}.csv'
        test_ids_path = kfold_dir / f'test_ids_{fold:03d}.csv'

        fold_model_dir = model_dir / f'{fold:03d}'
        fold_model_dir.mkdir(exist_ok=True)
        generator_train_list = []
        input_dim_list = []

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
                columns_name = [f"{dataset_name}_{i}" for i in range(132)]

            freesurfer_path = PROJECT_ROOT / 'data' / dataset_resourse / f'{dataset_name}.csv'

            train_dataset_df = load_dataset(participants_path, train_ids_path, freesurfer_path)
            train_dataset_df = train_dataset_df.loc[train_dataset_df['DIA'] == hc_label]
            train_data = train_dataset_df[columns_name].values

            scaler = RobustScaler()
            train_data = pd.DataFrame(scaler.fit_transform(train_data))

            train_data = train_data.astype(np.float32)


            train_covariates = train_dataset_df[['DIA', 'PTGENDER', 'AGE']]
            train_covariates['DIA'] = train_covariates['DIA'].replace(0, 0)

            # One-hot encode covariates
            AGE_bins_train = pd.qcut(train_covariates['AGE'].rank(method="first"), q=27, labels=False)
            one_hot_AGE = np.eye(27)[AGE_bins_train]

            PTGENDER_bins_train = pd.qcut(train_covariates['PTGENDER'].rank(method="first"), q=2, labels=False)
            one_hot_PTGENDER = np.eye(2)[PTGENDER_bins_train]

            # Prepare labels if DIA  is 2, then it is hc, if it is 0, then it is AD. do not consider MCI which is 1
            train_labels = train_covariates['DIA'].apply(lambda x: 0 if x == hc_label else 1).values.astype(np.float32)
            

            batch_size = 256

            input_dim = train_data.shape[1]
            one_hot_covariates_train = np.concatenate((one_hot_AGE, one_hot_PTGENDER), axis=1).astype('float32')

            one_hot_covariates_train = one_hot_covariates_train.astype(np.float32)


            c_dim = one_hot_covariates_train.shape[1]
            train_dataset = MyDataset_labels(train_data.to_numpy(), one_hot_covariates_train, train_labels)




            generator_train = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=False)

            generator_train_list.append(generator_train)
            input_dim_list.append(input_dim)

        h_dim = hz_para_list[:-1]
        z_dim = hz_para_list[-1]

        global_step = 0
        n_epochs = epochs
        gamma = 0.98
        scale_fn = lambda x: gamma ** x
        base_lr = 1e-6
        max_lr = 5e-5

        print('Training model...')
        model = cVAE_multimodal_endtoend(input_dim_list=input_dim_list, hidden_dim=h_dim, latent_dim=z_dim,
                                c_dim=c_dim, learning_rate=1e-4, modalities=modalities, non_linear=True)
        model.to(DEVICE)

        n_samples = len(train_dataset)
        step_size = 2 * np.ceil(n_samples / batch_size)

        for epoch in range(n_epochs):
            for batch_idx, batches in enumerate(zip(*generator_train_list)):
                global_step += 1
                cycle = np.floor(1 + global_step / (2 * step_size))
                x_lr = np.abs(global_step / step_size - 2 * cycle + 1)
                clr = base_lr + (max_lr - base_lr) * max(0, 1 - x_lr) * scale_fn(cycle)
                for param_group in model.optimizer1.param_groups:
                    param_group['lr'] = clr

                data_curr_list = []
                cov_list = []
                labels = None
                for modal in range(modalities):
                    data_curr = batches[modal][0].to(DEVICE)
                    cov = batches[modal][1].to(DEVICE)
                    label = batches[modal][2].to(DEVICE)
                    data_curr_list.append(data_curr)
                    cov_list.append(cov)
                    if labels is None:
                        labels = label

                fwd_rtn = model.forward_multimodal(data_curr_list, cov_list, combine)
                loss = model.loss_function_multimodal(data_curr_list, fwd_rtn, labels)

                model.optimizer1.zero_grad()
                loss['total'].backward()
                model.optimizer1.step()

                if batch_idx == 0:
                    to_print = f"Train Epoch: {epoch} Batch: {batch_idx} " + \
                               ', '.join([f"{k}: {v.item():.3f}" for k, v in loss.items()])
                    print(to_print)
                    if epoch == 0 and fold == 0:
                        log_keys = list(loss.keys())
                        logger = Logger()
                        logger.on_train_init(log_keys)
                    else:
                        logger.on_step_fi(loss)
        plot_losses(logger, fold_model_dir, 'training')
        model_path = fold_model_dir / 'cVAE_model.pkl'
        torch.save(model, model_path)

def test(args):
    """Inference the predictions of the clinical datasets using the supervised model."""
    n_splits = args.n_splits
    model_name = 'supervised_cvae'
    dataset_resourse = args.dataset_resourse
    procedure = args.procedure
    combine = args.combine

    outputs_dir = PROJECT_ROOT / 'outputs'
    kfold_dir = outputs_dir / 'kfold_analysis'
    model_dir = kfold_dir / model_name

    participants_path = PROJECT_ROOT / 'data' / dataset_resourse / 'y.csv'
    ids_df = pd.read_csv(participants_path)
    hc_label = 2 if dataset_resourse == 'ADNI' else 1
    HC_group = ids_df[ids_df['DIA'] == hc_label]

    if procedure.startswith('SE'):
        dataset_names = ['av45', 'vbm', 'fdg']
    elif procedure.startswith('UCA'):
        dataset_names = ['av45', 'vbm', 'fdg', '3modalities']
    else:
        raise ValueError('Unknown procedure: {}'.format(procedure))

    modalities = len(dataset_names)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(kf.split(HC_group)):
        train_ids_path = kfold_dir / f'train_ids_{fold:03d}.csv'
        test_ids_path = kfold_dir / f'test_ids_{fold:03d}.csv'

        fold_model_dir = model_dir / f'{fold:03d}'
        fold_model_dir.mkdir(exist_ok=True)

        test_data_list = []
        clinical_df_list = []

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
                columns_name = [f"{dataset_name}_{i}" for i in range(132)]

            freesurfer_path = PROJECT_ROOT / 'data' / dataset_resourse / f'{dataset_name}.csv'
            train_dataset_df = load_dataset(participants_path, train_ids_path, freesurfer_path)
            test_dataset_df = load_dataset(participants_path, test_ids_path, freesurfer_path)

            train_dataset_df = train_dataset_df.loc[train_dataset_df['DIA'] == hc_label]
            train_data = train_dataset_df[columns_name].values
            scaler = RobustScaler()
            train_data = pd.DataFrame(scaler.fit_transform(train_data))

            test_data = test_dataset_df[columns_name].values
            test_data = pd.DataFrame(scaler.transform(test_data))

            test_covariates = test_dataset_df[['DIA', 'AGE', 'PTGENDER']]
            test_covariates['DIA'] = test_covariates['DIA'].replace(0, 0)

            AGE_bins_test = pd.qcut(test_covariates['AGE'].rank(method="first"), q=27, labels=False)
            one_hot_AGE = np.eye(27)[AGE_bins_test]

            gender_bins_test = pd.qcut(test_covariates['PTGENDER'].rank(method="first"), q=2, labels=False)
            one_hot_gender = np.eye(2)[gender_bins_test]

            test_data_list.append(test_data)
            clinical_df_list.append(test_dataset_df)

        one_hot_covariates_test = np.concatenate((one_hot_AGE, one_hot_gender), axis=1).astype('float32')

        model_path = fold_model_dir / 'cVAE_model.pkl'
        if model_path.exists():
            print('Loading trained model...')
            model = torch.load(model_path)
            model.to(DEVICE)
        else:
            print('Model not found, please train the model first.')
            return

        test_prediction_list = model.pred_recon(test_data_list, one_hot_covariates_test, DEVICE, combine=combine)
        
        output_data = pd.DataFrame(clinical_df_list[0]['DIA'].values, columns=['DIA'])
        output_data_reconstruction_deviation_list = model.reconstruction_deviation_multimodal(test_data_list, test_prediction_list)

        for idx, (dataset_name, test_prediction) in enumerate(zip(dataset_names, test_prediction_list)):
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
                columns_name = [f"{dataset_name}_{i}" for i in range(132)]

            output_dataset_dir = fold_model_dir / dataset_name
            output_dataset_dir.mkdir(exist_ok=True)

            normalized_df = pd.DataFrame(test_data_list[idx], columns=columns_name)
            normalized_df['participant_id'] = clinical_df_list[0]['participant_id'].values
            normalized_df.to_csv(output_dataset_dir / f'normalized_{dataset_name}.csv', index=False)

            reconstruction_df = pd.DataFrame(test_prediction, columns=columns_name)
            reconstruction_df['participant_id'] = clinical_df_list[0]['participant_id'].values
            reconstruction_df.to_csv(output_dataset_dir / f'reconstruction_{dataset_name}.csv', index=False)

            output_data['reconstruction_deviation'] = output_data_reconstruction_deviation_list[idx]
            reconstruction_error_df = pd.DataFrame({
                'participant_id': clinical_df_list[0]['participant_id'].values,
                'Reconstruction error': output_data['reconstruction_deviation']
            })
            reconstruction_error_df.to_csv(output_dataset_dir / f'reconstruction_error_{dataset_name}.csv', index=False)

        # use MSE to calculate the diagnosis
        diagnosis = np.mean(output_data_reconstruction_deviation_list, axis=0)
        # Save diagnosis results
        diagnosis_df = pd.DataFrame({
            'participant_id': clinical_df_list[0]['participant_id'].values,
            'Diagnosis': diagnosis.flatten(),
            'True_Label': clinical_df_list[0]['DIA'].apply(lambda x: 0 if x == hc_label else 1).values
        })
        diagnosis_df.to_csv(fold_model_dir / 'diagnosis_results.csv', index=False)

        # Print diagnosis results
        print(f"Fold {fold}:")
        # print(diagnosis_df[['participant_id', 'Diagnosis', 'True_Label']])

        # # Optionally, compute and print accuracy for this fold
        # predicted_labels = (diagnosis.flatten() >= 0.5).astype(int)
        # true_labels = diagnosis_df['True_Label'].values
        # accuracy = np.mean(predicted_labels == true_labels)
        # print(f"Accuracy for fold {fold}: {accuracy:.4f}\n")



def analyze(args):
    """Perform the group analysis."""
    model_name = 'supervised_cvae'
    dataset_resourse = args.dataset_resourse
    procedure = args.procedure
    combine = args.combine
    epochs = args.epochs
    oversample_percentage = args.oversample_percentage
    n_splits = args.n_splits

    outputs_dir = PROJECT_ROOT / 'outputs'
    kfold_dir = outputs_dir / 'kfold_analysis'
    model_dir = kfold_dir / model_name

    participants_path = PROJECT_ROOT / 'data' / dataset_resourse / 'y.csv'
    ids_df = pd.read_csv(participants_path)
    hc_label = 2 if dataset_resourse == 'ADNI' else 1
    HC_group = ids_df[ids_df['DIA'] == hc_label]

    if procedure.startswith('SE'):
        dataset_names = ['av45', 'vbm', 'fdg']
    elif procedure.startswith('UCA'):
        dataset_names = ['av45', 'vbm', 'fdg', '3modalities']
    else:
        raise ValueError('Unknown procedure: {}'.format(procedure))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    auc_roc_list = []
    accuracy_list = []
    sensitivity_list = []
    specificity_list = []
    significance_ratio_list = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(HC_group)):
        fold_model_dir = model_dir / f'{fold:03d}'
        diagnosis_results_path = fold_model_dir / 'diagnosis_results.csv'

        if not diagnosis_results_path.exists():
            print(f"Diagnosis results not found for fold {fold}. Please run the test function first.")
            continue

        diagnosis_df = pd.read_csv(diagnosis_results_path)

        # True labels and predicted probabilities
        true_labels = diagnosis_df['True_Label'].values
        predicted_probs = diagnosis_df['Diagnosis'].values

        # Compute ROC AUC
        fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs)
        roc_auc = auc(fpr, tpr)
        auc_roc_list.append(roc_auc)

        # Optimal threshold using Youden's J statistic
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        # Convert probabilities to binary labels
        predicted_labels = (predicted_probs >= optimal_threshold).astype(int)

        # Compute metrics
        accuracy = np.mean(predicted_labels == true_labels)
        TP = np.sum((predicted_labels == 1) & (true_labels == 1))
        TN = np.sum((predicted_labels == 0) & (true_labels == 0))
        FP = np.sum((predicted_labels == 1) & (true_labels == 0))
        FN = np.sum((predicted_labels == 0) & (true_labels == 1))

        recall = TP / (TP + FN) if (TP + FN) > 0 else 0  # Sensitivity
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

        accuracy_list.append(accuracy)
        sensitivity_list.append(recall)
        specificity_list.append(specificity)
        significance_ratio = roc_auc / (1 - roc_auc) if roc_auc < 1 else float('inf')
        significance_ratio_list.append(significance_ratio)

        # Print metrics for the fold
        print(f"Fold {fold}:")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Sensitivity (Recall): {recall:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"Optimal Threshold: {optimal_threshold:.4f}")
        print()

    # Compute mean and standard deviation of metrics across folds
    mean_auc = np.mean(auc_roc_list)
    std_auc = np.std(auc_roc_list)
    mean_accuracy = np.mean(accuracy_list)
    std_accuracy = np.std(accuracy_list)
    mean_sensitivity = np.mean(sensitivity_list)
    std_sensitivity = np.std(sensitivity_list)
    mean_specificity = np.mean(specificity_list)
    std_specificity = np.std(specificity_list)
    mean_significance_ratio = np.mean(significance_ratio_list)
    std_significance_ratio = np.std(significance_ratio_list)

    # Print overall metrics
    print("Overall Performance:")
    print(f"Mean ROC AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Mean Sensitivity: {mean_sensitivity:.4f} ± {std_sensitivity:.4f}")
    print(f"Mean Specificity: {mean_specificity:.4f} ± {std_specificity:.4f}")
    print(f"Mean Significance Ratio: {mean_significance_ratio:.4f} ± {std_significance_ratio:.4f}")

    # Optionally, save the results to a file
    results_dir = outputs_dir / 'analysis_results'
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / 'performance_metrics.txt'
    with open(results_file, 'w') as f:
        f.write("Overall Performance:\n")
        f.write(f"Mean ROC AUC: {mean_auc:.4f} ± {std_auc:.4f}\n")
        f.write(f"Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}\n")
        f.write(f"Mean Sensitivity: {mean_sensitivity:.4f} ± {std_sensitivity:.4f}\n")
        f.write(f"Mean Specificity: {mean_specificity:.4f} ± {std_specificity:.4f}\n")
        f.write(f"Mean Significance Ratio: {mean_significance_ratio:.4f} ± {std_significance_ratio:.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train, Test, and Analyze the model.')
    parser.add_argument('action', choices=['train', 'test', 'analyze'], help='Action to perform: train, test, or analyze.')

    # Common arguments
    parser.add_argument('-R', '--dataset_resourse', type=str, default='ADNI', help='Dataset to use for training test and evaluation.')
    parser.add_argument('-H', '--hz_para_list', nargs='+', type=int, default=[110, 110, 10], help='List of paras to perform the analysis.')
    parser.add_argument('-C', '--combine', type=str, help='How to combine all modalities.')
    parser.add_argument('-P', '--procedure', type=str, default='SE-MoE', help='Procedure to perform the analysis.')
    parser.add_argument('-E', '--epochs', type=int, default=200, help='Number of epochs to train the model.')
    parser.add_argument('-K', '--n_splits', type=int, default=5, help='Number of splits for k-fold cross-validation.')
    parser.add_argument('-O', '--oversample_percentage', type=float, default=1, help='Percentage of oversampling of the training data.')

    args = parser.parse_args()

    if args.combine is None:
        args.combine = args.procedure.split('-')[1]
    if args.action == 'train':
        train(args)
    elif args.action == 'test':
        test(args)
    elif args.action == 'analyze':
        analyze(args)

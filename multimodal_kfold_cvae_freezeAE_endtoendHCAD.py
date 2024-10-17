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

from utils import load_dataset
# plt and sns are not used in this script
import matplotlib.pyplot as plt
import seaborn as sns
# tsne
from sklearn.manifold import TSNE

# Import your own modules (ensure these are available in your environment)
from cVAE import Encoder, Decoder, ProductOfExperts, MixtureOfExperts, MoPoE, ResidualBlock, mlp, mlp_residual
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
            xes_tensors = [torch.tensor(xes[i], dtype=torch.float32).to(DEVICE) for i in range(self.modalities)]
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
            # deviation = np.sum((xes[m] - x_preds[m])**2, axis=1)/xes[m].shape[1]
            deviation = (xes[m] - x_preds[m])**2
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
    other_group = ids_df[ids_df['DIA'] != hc_label]


    # the only usage of other_group here, is to provide the test_ids. all heritage to test_ids
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

# def test(args):
#     """Inference the predictions of the clinical datasets using the supervised model."""
#     n_splits = args.n_splits
#     model_name = 'supervised_cvae'
#     dataset_resourse = args.dataset_resourse
#     procedure = args.procedure
#     combine = args.combine

#     outputs_dir = PROJECT_ROOT / 'outputs'
#     kfold_dir = outputs_dir / 'kfold_analysis'
#     model_dir = kfold_dir / model_name

#     participants_path = PROJECT_ROOT / 'data' / dataset_resourse / 'y.csv'
#     ids_df = pd.read_csv(participants_path)
#     hc_label = 2 if dataset_resourse == 'ADNI' else 1
#     HC_group = ids_df[ids_df['DIA'] == hc_label]

#     if procedure.startswith('SE'):
#         dataset_names = ['av45', 'vbm', 'fdg']
#     elif procedure.startswith('UCA'):
#         dataset_names = ['av45', 'vbm', 'fdg', '3modalities']
#     else:
#         raise ValueError('Unknown procedure: {}'.format(procedure))

#     modalities = len(dataset_names)
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    


#     for fold, (train_idx, test_idx) in enumerate(kf.split(HC_group)):
#         train_ids_path = kfold_dir / f'train_ids_{fold:03d}.csv'
#         test_ids_path = kfold_dir / f'test_ids_{fold:03d}.csv'

#         fold_model_dir = model_dir / f'{fold:03d}'
#         fold_model_dir.mkdir(exist_ok=True)

#         test_data_list = []
#         clinical_df_list = []

        

#         for dataset_name in dataset_names:
#             if dataset_resourse == 'ADNI':
#                 if dataset_name in ['av45', 'fdg']:
#                     columns_name = COLUMNS_NAME
#                 elif dataset_name == 'snp':
#                     columns_name = COLUMNS_NAME_SNP
#                 elif dataset_name == 'vbm':
#                     columns_name = COLUMNS_NAME_VBM
#                 elif dataset_name == '3modalities':
#                     columns_name = COLUMNS_3MODALITIES
#             elif dataset_resourse == 'HCP':
#                 columns_name = [f"{dataset_name}_{i}" for i in range(132)]

#             freesurfer_path = PROJECT_ROOT / 'data' / dataset_resourse / f'{dataset_name}.csv'
#             train_dataset_df = load_dataset(participants_path, train_ids_path, freesurfer_path)
#             test_dataset_df = load_dataset(participants_path, test_ids_path, freesurfer_path)

#             train_dataset_df = train_dataset_df.loc[train_dataset_df['DIA'] == hc_label]
#             train_data = train_dataset_df[columns_name].values
#             scaler = RobustScaler()
#             train_data = pd.DataFrame(scaler.fit_transform(train_data))

#             test_data = test_dataset_df[columns_name].values
#             test_data = pd.DataFrame(scaler.transform(test_data))

#             test_covariates = test_dataset_df[['DIA', 'AGE', 'PTGENDER']]
#             test_covariates['DIA'] = test_covariates['DIA'].replace(0, 0)

#             AGE_bins_test = pd.qcut(test_covariates['AGE'].rank(method="first"), q=27, labels=False)
#             one_hot_AGE = np.eye(27)[AGE_bins_test]

#             gender_bins_test = pd.qcut(test_covariates['PTGENDER'].rank(method="first"), q=2, labels=False)
#             one_hot_gender = np.eye(2)[gender_bins_test]

#             test_data_list.append(test_data)
#             clinical_df_list.append(test_dataset_df)

#         one_hot_covariates_test = np.concatenate((one_hot_AGE, one_hot_gender), axis=1).astype('float32')

#         model_path = fold_model_dir / 'cVAE_model.pkl'
#         if model_path.exists():
#             print('Loading trained model...')
#             model = torch.load(model_path)
#             model.to(DEVICE)
#         else:
#             print('Model not found, please train the model first.')
#             return

#         test_prediction_list = model.pred_recon(test_data_list, one_hot_covariates_test, DEVICE, combine=combine)
        
#         output_data = pd.DataFrame(clinical_df_list[0]['DIA'].values, columns=['DIA'])
#         output_data_reconstruction_deviation_list = model.reconstruction_deviation_multimodal(test_data_list, test_prediction_list)

#         for idx, (dataset_name, test_prediction) in enumerate(zip(dataset_names, test_prediction_list)):
#             if dataset_resourse == 'ADNI':
#                 if dataset_name in ['av45', 'fdg']:
#                     columns_name = COLUMNS_NAME
#                 elif dataset_name == 'snp':
#                     columns_name = COLUMNS_NAME_SNP
#                 elif dataset_name == 'vbm':
#                     columns_name = COLUMNS_NAME_VBM
#                 elif dataset_name == '3modalities':
#                     columns_name = COLUMNS_3MODALITIES
#             elif dataset_resourse == 'HCP':
#                 columns_name = [f"{dataset_name}_{i}" for i in range(132)]

#             output_dataset_dir = fold_model_dir / dataset_name
#             output_dataset_dir.mkdir(exist_ok=True)

#             normalized_df = pd.DataFrame(test_data_list[idx], columns=columns_name)
#             normalized_df['participant_id'] = clinical_df_list[0]['participant_id'].values
#             normalized_df.to_csv(output_dataset_dir / f'normalized_{dataset_name}.csv', index=False)

#             reconstruction_df = pd.DataFrame(test_prediction, columns=columns_name)
#             reconstruction_df['participant_id'] = clinical_df_list[0]['participant_id'].values
#             reconstruction_df.to_csv(output_dataset_dir / f'reconstruction_{dataset_name}.csv', index=False)

#             output_data['reconstruction_deviation'] = output_data_reconstruction_deviation_list[idx]
#             reconstruction_error_df = pd.DataFrame({
#                 'participant_id': clinical_df_list[0]['participant_id'].values,
#                 'Reconstruction error': output_data['reconstruction_deviation']
#             })
#             reconstruction_error_df.to_csv(output_dataset_dir / f'reconstruction_error_{dataset_name}.csv', index=False)

#         # # use MSE to calculate the diagnosis
#         diagnosis = np.mean(output_data_reconstruction_deviation_list, axis=0)

#         # Save diagnosis results
#         diagnosis_df = pd.DataFrame({
#             'participant_id': clinical_df_list[0]['participant_id'].values,
#             'Diagnosis': diagnosis.flatten(),
#             'True_Label': clinical_df_list[0]['DIA'].apply(lambda x: 0 if x == hc_label else 1).values
#         })
#         diagnosis_df.to_csv(fold_model_dir / 'diagnosis_results.csv', index=False)

#         # Print diagnosis results
#         print(f"Fold {fold}:")
#         # print(diagnosis_df[['participant_id', 'Diagnosis', 'True_Label']])

#         # # Optionally, compute and print accuracy for this fold
#         # predicted_labels = (diagnosis.flatten() >= 0.5).astype(int)
#         # true_labels = diagnosis_df['True_Label'].values
#         # accuracy = np.mean(predicted_labels == true_labels)
#         # print(f"Accuracy for fold {fold}: {accuracy:.4f}\n")















# def test(args):
#     """Inference the predictions of the clinical datasets using the supervised model."""
#     n_splits = args.n_splits
#     model_name = 'supervised_cvae'
#     dataset_resourse = args.dataset_resourse
#     procedure = args.procedure
#     combine = args.combine

#     outputs_dir = PROJECT_ROOT / 'outputs'
#     kfold_dir = outputs_dir / 'kfold_analysis'
#     model_dir = kfold_dir / model_name

#     participants_path = PROJECT_ROOT / 'data' / dataset_resourse / 'y.csv'
#     ids_df = pd.read_csv(participants_path)
#     hc_label = 2 if dataset_resourse == 'ADNI' else 1
#     mci_label = 1  # Assuming MCI is labeled 1 in your dataset
#     ad_label = 0  # Assuming AD is labeled 0 in your dataset
#     HC_group = ids_df[ids_df['DIA'] == hc_label]

#     if procedure.startswith('SE'):
#         dataset_names = ['av45', 'vbm', 'fdg']
#     elif procedure.startswith('UCA'):
#         dataset_names = ['av45', 'vbm', 'fdg', '3modalities']
#     else:
#         raise ValueError('Unknown procedure: {}'.format(procedure))

#     modalities = len(dataset_names)
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

#     for fold, (train_idx, test_idx) in enumerate(kf.split(HC_group)):
#         train_ids_path = kfold_dir / f'train_ids_{fold:03d}.csv'
#         test_ids_path = kfold_dir / f'test_ids_{fold:03d}.csv'

#         fold_model_dir = model_dir / f'{fold:03d}'
#         fold_model_dir.mkdir(exist_ok=True)

#         # Load the trained AE model
#         model_path = fold_model_dir / 'cVAE_model.pkl'
#         if model_path.exists():
#             print('Loading trained model...')
#             model = torch.load(model_path)
#             model.to(DEVICE)
#             model.eval()  # Set the model to evaluation mode
#         else:
#             print('Model not found, please train the model first.')
#             return

#         # --- Step 1: Compute Reconstruction Deviations for Training Data (HC) ---
#         # (Unchanged)

#         # --- Step 2: Compute Reconstruction Deviations for Test Data (HC, MCI, AD) ---
#         # (Unchanged)
#         # --- Step 1: Compute Reconstruction Deviations for Training Data (HC) ---
#         # Load and preprocess training data
#         train_data_list = []
#         train_covariates_list = []
#         train_clinical_df = None

#         for dataset_name in dataset_names:
#             # Load training data
#             freesurfer_path = PROJECT_ROOT / 'data' / dataset_resourse / f'{dataset_name}.csv'
#             train_dataset_df = load_dataset(participants_path, train_ids_path, freesurfer_path)

#             # get the columns name
#             if dataset_resourse == 'ADNI':
#                 if dataset_name in ['av45', 'fdg']:
#                     columns_name = COLUMNS_NAME
#                 elif dataset_name == 'snp':
#                     columns_name = COLUMNS_NAME_SNP
#                 elif dataset_name == 'vbm':
#                     columns_name = COLUMNS_NAME_VBM
#                 elif dataset_name == '3modalities':
#                     columns_name = COLUMNS_3MODALITIES
#             elif dataset_resourse == 'HCP':
#                 columns_name = [f"{dataset_name}_{i}" for i in range(132)]
                

#             # Filter HC only
#             train_dataset_df = train_dataset_df.loc[train_dataset_df['DIA'] == hc_label]
#             train_data = train_dataset_df[columns_name].values

#             # Fit scaler on training data
#             scaler = RobustScaler()
#             train_data_scaled = scaler.fit_transform(train_data)
#             train_data_scaled = pd.DataFrame(train_data_scaled)

#             # Prepare covariates
#             train_covariates = train_dataset_df[['DIA', 'AGE', 'PTGENDER']]
#             AGE_bins_train = pd.qcut(train_covariates['AGE'].rank(method="first"), q=27, labels=False)
#             one_hot_AGE_train = np.eye(27)[AGE_bins_train]
#             gender_bins_train = pd.qcut(train_covariates['PTGENDER'].rank(method="first"), q=2, labels=False)
#             one_hot_gender_train = np.eye(2)[gender_bins_train]
#             one_hot_covariates_train = np.concatenate((one_hot_AGE_train, one_hot_gender_train), axis=1).astype('float32')

#             train_data_list.append(train_data_scaled.values)
#             train_covariates_list.append(one_hot_covariates_train)
#             train_clinical_df = train_dataset_df  # Save for labels

#         # Compute deviations for training data
#         train_predictions_list= model.pred_recon(train_data_list, train_covariates_list[0], DEVICE, combine=combine)
#         train_deviation_list = model.reconstruction_deviation_multimodal(train_data_list, train_predictions_list)
#         # Flatten deviations per sample
#         # print('train_deviation_list:', train_deviation_list)
#         train_recon_errors = np.concatenate(train_deviation_list, axis=1)
#         # print('train_recon_errors:', train_recon_errors.shape)
#         # train_recon_errors = train_deviation_list
        
#         train_labels = np.zeros(train_recon_errors.shape[0], dtype=np.float32)  # HC labels are 2

#         # --- Step 2: Compute Reconstruction Deviations for Test Data (HC and AD) ---
#         test_data_list = []
#         test_covariates_list = []
#         test_clinical_df = None

#         for dataset_name in dataset_names:
#             # Load test data
#             freesurfer_path = PROJECT_ROOT / 'data' / dataset_resourse / f'{dataset_name}.csv'
#             test_dataset_df = load_dataset(participants_path, test_ids_path, freesurfer_path)

#             # get the columns name
#             if dataset_resourse == 'ADNI':
#                 if dataset_name in ['av45', 'fdg']:
#                     columns_name = COLUMNS_NAME
#                 elif dataset_name == 'snp':
#                     columns_name = COLUMNS_NAME_SNP
#                 elif dataset_name == 'vbm':
#                     columns_name = COLUMNS_NAME_VBM
#                 elif dataset_name == '3modalities':
#                     columns_name = COLUMNS_3MODALITIES
#             elif dataset_resourse == 'HCP':
#                 columns_name = [f"{dataset_name}_{i}" for i in range(132)]

#             # print('test_dataset:', test_dataset_df)

#             # Scale test data using training scaler
#             test_data = test_dataset_df[columns_name].values
#             test_data_scaled = pd.DataFrame(scaler.transform(test_data)Z)

#             # Prepare covariates
#             test_covariates = test_dataset_df[['DIA', 'AGE', 'PTGENDER']]
#             AGE_bins_test = pd.qcut(test_covariates['AGE'].rank(method="first"), q=27, labels=False)
#             one_hot_AGE_test = np.eye(27)[AGE_bins_test]
#             gender_bins_test = pd.qcut(test_covariates['PTGENDER'].rank(method="first"), q=2, labels=False)
#             one_hot_gender_test = np.eye(2)[gender_bins_test]
#             one_hot_covariates_test = np.concatenate((one_hot_AGE_test, one_hot_gender_test), axis=1).astype('float32')

#             test_data_list.append(test_data_scaled.values)
#             test_covariates_list.append(one_hot_covariates_test)
#             test_clinical_df = test_dataset_df  # Save for labels

#         # Compute deviations for test data
#         test_predictions_list = model.pred_recon(test_data_list, test_covariates_list[0], DEVICE, combine=combine)
#         test_deviation_list = model.reconstruction_deviation_multimodal(test_data_list, test_predictions_list)
#         # print('test_deviation_list:', test_deviation_list)
#         for i in test_deviation_list:
#             print(i.shape)
#         test_recon_errors = np.concatenate(test_deviation_list, axis=1)
#         # print('test_recon_errors:', test_recon_errors.shape)
#         test_labels = test_clinical_df['DIA'].apply(lambda x: 0 if x == hc_label else 1).values.astype(np.float32)

#         # --- Step 3: Modify for each classification (HC vs AD, HC vs MCI, MCI vs AD) ---

#         def filter_labels_for_comparison(test_labels, group1, group2):
#             valid_indices = np.where((test_labels == group1) | (test_labels == group2))[0]
#             binary_labels = test_labels[valid_indices]
#             # Map the two groups to binary labels (0 and 1)
#             binary_labels = np.where(binary_labels == group1, 0, 1)
#             return valid_indices, binary_labels

#         # HC vs AD
#         hc_vs_ad_indices, hc_vs_ad_labels = filter_labels_for_comparison(test_labels, hc_label, ad_label)
#         mlp_train_data_hc_ad = test_recon_errors[hc_vs_ad_indices]
#         mlp_train_labels_hc_ad = hc_vs_ad_labels

#         # HC vs MCI
#         hc_vs_mci_indices, hc_vs_mci_labels = filter_labels_for_comparison(test_labels, hc_label, mci_label)
#         mlp_train_data_hc_mci = test_recon_errors[hc_vs_mci_indices]
#         mlp_train_labels_hc_mci = hc_vs_mci_labels

#         # MCI vs AD
#         mci_vs_ad_indices, mci_vs_ad_labels = filter_labels_for_comparison(test_labels, mci_label, ad_label)
#         mlp_train_data_mci_ad = test_recon_errors[mci_vs_ad_indices]
#         mlp_train_labels_mci_ad = mci_vs_ad_labels
        

#         # --- Step 4: Train the New MLP Classifier for each comparison ---

#         def train_mlp(mlp_train_data, mlp_train_labels, fold_model_dir, comparison_name):
#             mlp_input_dim = mlp_train_data.shape[1]

#             if args.mlptype == 'mlp_residual':
#                 mlp = nn.Sequential(
#                     nn.Linear(mlp_input_dim, 512),
#                     nn.LayerNorm(512),
#                     nn.LeakyReLU(0.1),
#                     nn.Dropout(0.3),
#                     ResidualBlock(512, 256),
#                     ResidualBlock(512, 256),
#                     ResidualBlock(512, 256),
#                     nn.Linear(512, 128),
#                     nn.LayerNorm(128),
#                     nn.LeakyReLU(0.1),
#                     nn.Dropout(0.3),
#                     nn.Linear(128, 1),
#                     nn.Sigmoid()
#                 ).to(DEVICE)
#             elif args.mlptype == 'mlp':
#                 mlp = nn.Sequential(
#                     nn.Linear(mlp_input_dim, 256),
#                     nn.BatchNorm1d(256),
#                     nn.ReLU(),
#                     nn.Dropout(0.3),
#                     nn.Linear(256, 128),
#                     nn.BatchNorm1d(128),
#                     nn.ReLU(),
#                     nn.Dropout(0.3),
#                     nn.Linear(128, 64),
#                     nn.BatchNorm1d(64),
#                     nn.ReLU(),
#                     nn.Dropout(0.3),
#                     nn.Linear(64, 32),
#                     nn.ReLU(),
#                     nn.Linear(32, 1),
#                     nn.Sigmoid()
#                 ).to(DEVICE)

#             # Loss function and optimizer
#             criterion = nn.BCELoss()
#             optimizer = optim.Adam(mlp.parameters(), lr=args.test_lr)
#             num_epochs = args.test_epochs

#             # Convert data to tensors
#             mlp_train_data_tensor = torch.tensor(mlp_train_data, dtype=torch.float32).to(DEVICE)
#             mlp_train_labels_tensor = torch.tensor(mlp_train_labels, dtype=torch.float32).unsqueeze(1).to(DEVICE)

#             # Training loop
#             mlp.train()
#             for epoch in range(num_epochs):
#                 optimizer.zero_grad()
#                 outputs = mlp(mlp_train_data_tensor)
#                 loss = criterion(outputs, mlp_train_labels_tensor)
#                 loss.backward()
#                 optimizer.step()

#                 if epoch % 10 == 0:
#                     print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")

#             # Save model
#             torch.save(mlp.state_dict(), fold_model_dir / f'{comparison_name}_mlp_model.pth')

#         # Train the MLP for each binary comparison
#         train_mlp(mlp_train_data_hc_ad, mlp_train_labels_hc_ad, fold_model_dir, 'HC_vs_AD')
#         train_mlp(mlp_train_data_hc_mci, mlp_train_labels_hc_mci, fold_model_dir, 'HC_vs_MCI')
#         train_mlp(mlp_train_data_mci_ad, mlp_train_labels_mci_ad, fold_model_dir, 'MCI_vs_AD')



#         # --- Step 5: Predict and Evaluate for Remaining Test Data for each comparison ---
#         def evaluate_mlp(mlp_train_data, mlp_train_labels, fold_model_dir, comparison_name):
#             mlp_test_data_tensor = torch.tensor(mlp_train_data, dtype=torch.float32).to(DEVICE)
#             with torch.no_grad():
#                 diagnosis = mlp(mlp_test_data_tensor).cpu().numpy().flatten()

#             remaining_test_indices = np.setdiff1d(np.arange(test_recon_errors.shape[0]), 

#             # Save diagnosis results
#             diagnosis_df = pd.DataFrame({
#                 'participant_id': test_clinical_df['participant_id'].values[remaining_test_indices],
#                 'Diagnosis': diagnosis,
#                 'True_Label': mlp_train_labels
#             })
#             diagnosis_df.to_csv(fold_model_dir / f'{comparison_name}_diagnosis_results.csv', index=False)

#             # Compute and print accuracy for this fold
#             fpr, tpr, thresholds = roc_curve(mlp_train_labels, diagnosis)
#             optimal_idx = np.argmax(tpr - fpr)
#             optimal_threshold = thresholds[optimal_idx]
#             predicted_labels = (diagnosis >= optimal_threshold).astype(int)

#             accuracy = np.mean(predicted_labels == mlp_train_labels)
#             print(f"{comparison_name}: Accuracy for fold {fold}: {accuracy:.4f}\n")

#         evaluate_mlp(mlp_train_data_hc_ad, mlp_train_labels_hc_ad, fold_model_dir, 'HC_vs_AD')
#         evaluate_mlp(mlp_train_data_hc_mci, mlp_train_labels_hc_mci, fold_model_dir, 'HC_vs_MCI')
#         evaluate_mlp(mlp_train_data_mci_ad, mlp_train_labels_mci_ad, fold_model_dir, 'MCI_vs_AD')

#         # --- Step 6: Visualize the Results (Optional) ---
#         # You can reuse the t-SNE visualization part for each binary classification















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
    mci_label = 1  # MCI is labeled 1 in dataset
    ad_label = 0  # AD is labeled 0 in dataset
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

        # Load the trained AE model
        model_path = fold_model_dir / 'cVAE_model.pkl'
        if model_path.exists():
            print('Loading trained model...')
            model = torch.load(model_path)
            model.to(DEVICE)
            model.eval()  # Set the model to evaluation mode
        else:
            print('Model not found, please train the model first.')
            return

        # --- Step 1: Compute Reconstruction Deviations for Training Data (HC) ---
        # Load and preprocess training data
        train_data_list = []
        train_covariates_list = []
        train_clinical_df = None

        scaler_list = []

        for dataset_name in dataset_names:
            # Load training data
            freesurfer_path = PROJECT_ROOT / 'data' / dataset_resourse / f'{dataset_name}.csv'
            train_dataset_df = load_dataset(participants_path, train_ids_path, freesurfer_path)

            # get the columns name
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
                

            # Filter HC only
            train_dataset_df = train_dataset_df.loc[train_dataset_df['DIA'] == hc_label]
            train_data = train_dataset_df[columns_name].values

            # Fit scaler on training data
            scaler = RobustScaler()
            train_data_scaled = scaler.fit_transform(train_data)
            scaler_list.append(scaler)
            train_data_scaled = pd.DataFrame(train_data_scaled)

            # Prepare covariates
            train_covariates = train_dataset_df[['DIA', 'AGE', 'PTGENDER']]
            AGE_bins_train = pd.qcut(train_covariates['AGE'].rank(method="first"), q=27, labels=False)
            one_hot_AGE_train = np.eye(27)[AGE_bins_train]
            gender_bins_train = pd.qcut(train_covariates['PTGENDER'].rank(method="first"), q=2, labels=False)
            one_hot_gender_train = np.eye(2)[gender_bins_train]
            one_hot_covariates_train = np.concatenate((one_hot_AGE_train, one_hot_gender_train), axis=1).astype('float32')

            train_data_list.append(train_data_scaled.values)
            train_covariates_list.append(one_hot_covariates_train)
            train_clinical_df = train_dataset_df  # Save for labels

        # Compute deviations for training data
        train_predictions_list= model.pred_recon(train_data_list, train_covariates_list[0], DEVICE, combine=combine)
        train_deviation_list = model.reconstruction_deviation_multimodal(train_data_list, train_predictions_list)
        # Flatten deviations per sample
        # print('train_deviation_list:', train_deviation_list)
        train_recon_errors = np.concatenate(train_deviation_list, axis=1)
        # print('train_recon_errors:', train_recon_errors.shape)
        # train_recon_errors = train_deviation_list
        
        train_labels = np.zeros(train_recon_errors.shape[0], dtype=np.float32)  # HC labels are 2

        # --- Step 2: Compute Reconstruction Deviations for Test Data (HC and AD) ---
        test_data_list = []
        test_covariates_list = []
        test_clinical_df = None

        # Mapping comparison patterns to labels
        comparison_map = {
            'HC_vs_AD': (hc_label, ad_label),
            'HC_vs_MCI': (hc_label, mci_label),
            'MCI_vs_AD': (mci_label, ad_label)
        }

        # Assign healthier and disease labels based on the comparison pattern
        healthier_label, disease_label = comparison_map.get(args.compare_pattern)

        

        for dataset_name in dataset_names:
            # Load test data
            freesurfer_path = PROJECT_ROOT / 'data' / dataset_resourse / f'{dataset_name}.csv'
            test_dataset_df = load_dataset(participants_path, test_ids_path, freesurfer_path)
            # print('freesurfer_path:', freesurfer_path)

            # the test_dataset_df is for full test process. so should include 2 types of labels of all
            # first we extract them all from test_dataset_df, then use them later
            test_dataset_df = test_dataset_df.loc[(test_dataset_df['DIA'] == healthier_label) | (test_dataset_df['DIA'] == disease_label)]

            # get the columns name
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

            # print('test_dataset:', test_dataset_df)

            # Scale test data using training scaler
            test_data = test_dataset_df[columns_name].values
            # test_data_scaled = pd.DataFrame(scaler.transform(test_data))
            index = dataset_names.index(dataset_name)
            test_data_scaled = pd.DataFrame(scaler_list[index].transform(test_data))

            # Prepare covariates
            test_covariates = test_dataset_df[['DIA', 'AGE', 'PTGENDER']]
            AGE_bins_test = pd.qcut(test_covariates['AGE'].rank(method="first"), q=27, labels=False)
            one_hot_AGE_test = np.eye(27)[AGE_bins_test]
            gender_bins_test = pd.qcut(test_covariates['PTGENDER'].rank(method="first"), q=2, labels=False)
            one_hot_gender_test = np.eye(2)[gender_bins_test]
            one_hot_covariates_test = np.concatenate((one_hot_AGE_test, one_hot_gender_test), axis=1).astype('float32')

            test_data_list.append(test_data_scaled.values)
            test_covariates_list.append(one_hot_covariates_test)
            test_clinical_df = test_dataset_df  # Save for labels

        # Compute deviations for test data
        test_predictions_list = model.pred_recon(test_data_list, test_covariates_list[0], DEVICE, combine=combine)
        test_deviation_list = model.reconstruction_deviation_multimodal(test_data_list, test_predictions_list)
        # print('test_deviation_list:', test_deviation_list)
        for i in test_deviation_list:
            print(i.shape)
        test_recon_errors = np.concatenate(test_deviation_list, axis=1)
        # print('test_recon_errors:', test_recon_errors.shape)
        # label_healthier = 1 and label_disease = 0. 
        # test_labels = test_clinical_df['DIA'].apply(lambda x: 0 if x == hc_label else 1).values.astype(np.float32)
        test_labels = test_clinical_df['DIA'].apply(lambda x: 1 if x == healthier_label else 0).values.astype(np.float32)
        

        
        

        # --- Step 3: Prepare MLP Data ---
        # Select a portion of AD individuals from test data for MLP training


        if healthier_label == hc_label:
            ad_indices = np.where(test_labels == 0)[0]
            hc_indices = np.where(test_labels == 1)[0]

            num_ad_samples = int(len(ad_indices) * 0.4)  # Use 20% of AD samples for MLP training
            ad_train_indices = np.random.choice(ad_indices, size=num_ad_samples, replace=False)
            ad_test_indices = np.setdiff1d(ad_indices, ad_train_indices)
            
            remaining_test_indices = np.concatenate((ad_test_indices, hc_indices))  # Remaining test data

            # Combine HC deviations from training data with selected AD deviations
            mlp_train_data = np.concatenate((train_recon_errors, test_recon_errors[ad_train_indices]), axis=0)
            mlp_train_labels = np.concatenate((train_labels, test_labels[ad_train_indices]), axis=0)
            print('this is case 1, and num_ad_samples is {}, and size of train data is {}'.format(num_ad_samples, train_labels.shape))

        else:
            ad_indices = np.where(test_labels == 0)[0]
            hc_indices = np.where(test_labels == 1)[0]

            num_ad_samples = int(len(ad_indices) * 0.2)  # Use 20% of AD samples for MLP training
            num_hc_samples = int(len(hc_indices) * 0.2)  # Use 20% of HC- samples for MLP training

            ad_train_indices = np.random.choice(ad_indices, size=num_ad_samples, replace=False)
            hc_train_indices = np.random.choice(hc_indices, size=num_hc_samples, replace=False)

            ad_test_indices = np.setdiff1d(ad_indices, ad_train_indices)
            hc_test_indices = np.setdiff1d(hc_indices, hc_train_indices)

            remaining_test_indices = np.concatenate((ad_test_indices, hc_test_indices))  # Remaining test data

            # Combine HC deviations from training data with selected AD deviations
            mlp_train_data = np.concatenate((test_recon_errors[ad_train_indices], test_recon_errors[hc_train_indices]), axis=0)
            mlp_train_labels = np.concatenate((test_labels[ad_train_indices], test_labels[hc_train_indices]), axis=0)
            print('this is case 2, and num_ad_samples is {}, and size of hc samples is {}"'.format(num_ad_samples, num_hc_samples))

        # --- Step 4: Train the New MLP Classifier ---
        # Define the MLP model
        print('Training MLP model...')
        mlp_input_dim = mlp_train_data.shape[1]


        



        if args.mlptype == 'mlp_residual':
            mlp = nn.Sequential(
                nn.Linear(mlp_input_dim, 512),   # Increased neurons to 512
                nn.LayerNorm(512),               # Layer Normalization
                nn.LeakyReLU(0.1),               # Leaky ReLU activation
                nn.Dropout(0.3),                 # Dropout for regularization

                # Residual Blocks (512 -> 256 -> 512)
                ResidualBlock(512, 256),
                ResidualBlock(512, 256),
                ResidualBlock(512, 256),

                # Final Layers
                nn.Linear(512, 128),             # Decreasing neuron count
                nn.LayerNorm(128),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.3),
                nn.Linear(128, 1),
                nn.Sigmoid()                     # Output layer for binary classification
            ).to(DEVICE)
        elif args.mlptype == 'mlp':
            mlp = nn.Sequential(
                nn.Linear(mlp_input_dim, 256),    # Increased neurons to 256
                nn.BatchNorm1d(256),              # Batch Normalization
                nn.ReLU(),
                nn.Dropout(0.3),                  # Dropout for regularization
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),                # Added extra layer with 32 neurons
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()                      # Final layer for binary classification
            ).to(DEVICE)





        # Loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(mlp.parameters(), lr=args.test_lr)
        num_epochs = args.test_epochs

        print('mlp_train_data:', mlp_train_data.shape)

        # Convert data to tensors
        mlp_train_data_tensor = torch.tensor(mlp_train_data, dtype=torch.float32).to(DEVICE)
        mlp_train_labels_tensor = torch.tensor(mlp_train_labels, dtype=torch.float32).unsqueeze(1).to(DEVICE)

        # Training loop
        mlp.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = mlp(mlp_train_data_tensor)
            loss = criterion(outputs, mlp_train_labels_tensor)
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")

        # --- Step 5: Predict on Remaining Test Data ---
        mlp.eval()
        mlp_test_data_tensor = torch.tensor(test_recon_errors[remaining_test_indices], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            diagnosis = mlp(mlp_test_data_tensor).cpu().numpy().flatten()

        # Save diagnosis results
        diagnosis_df = pd.DataFrame({
            'participant_id': test_clinical_df['participant_id'].values[remaining_test_indices],
            'Diagnosis': diagnosis,
            'True_Label': test_labels[remaining_test_indices]
        })
        diagnosis_df.to_csv(fold_model_dir / 'diagnosis_results.csv', index=False)

        # Compute and print accuracy for this fold
        # predicted_labels = (diagnosis >= 0.5).astype(int)
        # use J statistic to find the optimal threshold
        fpr, tpr, thresholds = roc_curve(test_labels[remaining_test_indices], diagnosis)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        predicted_labels = (diagnosis >= optimal_threshold).astype(int)
        
        true_labels = test_labels[remaining_test_indices]
        accuracy = np.mean(predicted_labels == true_labels)
        print(f"Fold {fold}:")
        print(f"Accuracy for fold {fold}: {accuracy:.4f}\n")

        # --- Step 6: Visualize the Results ---
        # # use t-SNE to visualize the reconstruction errors of test data
        # tsne = TSNE(n_components=2, random_state=42)
        # tsne_results = tsne.fit_transform(test_recon_errors[remaining_test_indices])
        # tsne_df = pd.DataFrame(tsne_results, columns=['tsne_1', 'tsne_2'])
        # tsne_df['Diagnosis'] = test_labels[remaining_test_indices]
        # tsne_df['Predicted'] = diagnosis
        # # draw the t-SNE plot
        # plt.figure(figsize=(10, 6))
        # sns.scatterplot(x='tsne_1', y='tsne_2', hue='Diagnosis', style='Predicted', data=tsne_df, palette='viridis')
        # plt.title(f'Fold {fold} t-SNE Plot')
        # plt.savefig(fold_model_dir / 't-SNE_plot.png')
        # plt.close()


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
    print(f"compare_pattern: {args.compare_pattern}")
    print(f"Mean ROC AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Mean Sensitivity: {mean_sensitivity:.4f} ± {std_sensitivity:.4f}")
    print(f"Mean Specificity: {mean_specificity:.4f} ± {std_specificity:.4f}")
    print(f"Mean Significance Ratio: {mean_significance_ratio:.4f} ± {std_significance_ratio:.4f}")


    # Optionally, save the results to a file
    results_dir = outputs_dir / 'analysis_results'
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / 'performance_metrics.txt'
    with open(results_file, 'a') as f:
        # write hyperparameters
        f.write(f"hyperparameters: {args}\n")
        f.write("Overall Performance:\n")
        f.write(f"compare_pattern: {args.compare_pattern}\n")
        f.write(f"Mean ROC AUC: {mean_auc:.4f} ± {std_auc:.4f}\n")
        f.write(f"Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}\n")
        f.write(f"Mean Sensitivity: {mean_sensitivity:.4f} ± {std_sensitivity:.4f}\n")
        f.write(f"Mean Specificity: {mean_specificity:.4f} ± {std_specificity:.4f}\n")
        f.write(f"Mean Significance Ratio: {mean_significance_ratio:.4f} ± {std_significance_ratio:.4f}\n")
        f.write("\n")

    return mean_auc, std_auc, mean_accuracy, std_accuracy, mean_sensitivity, std_sensitivity, mean_specificity, std_specificity, mean_significance_ratio, std_significance_ratio


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
    parser.add_argument('-TestE', '--test_epochs', type=int, default=200, help='Number of epochs to train the model.')
    parser.add_argument('-TestLR', '--test_lr', type=float, default=1e-4, help='Learning rate for the model.')
    parser.add_argument('-MLP', '--mlptype', type=str, default='mlp', help='Type of mlp to use for the analysis.')
    # parser.add_argument('-Compare', '--compare_pattern', type=str, default='HC_vs_AD', help='Comparison pattern to use for the analysis.')
    

    args = parser.parse_args()

    if args.combine is None:
        args.combine = args.procedure.split('-')[1]

    # Perform action based on 'args.action'
    if args.action == 'train':
        train(args)
    elif args.action == 'test':
        # Initialize lists to store results across comparisons
        auc_means, accuracy_means, sensitivity_means, specificity_means, significance_ratio_means = [], [], [], [], []
        auc_stds, accuracy_stds, sensitivity_stds, specificity_stds, significance_ratio_stds = [], [], [], [], []

        # Loop through comparison patterns and collect metrics
        for compare_pattern in ['HC_vs_AD', 'HC_vs_MCI', 'MCI_vs_AD']:
            args.compare_pattern = compare_pattern
            test(args)
            
            # Analyze results
            mean_auc, std_auc, mean_accuracy, std_accuracy, mean_sensitivity, std_sensitivity, mean_specificity, std_specificity, mean_significance_ratio, std_significance_ratio = analyze(args)
            
            # Store the results
            auc_means.append(mean_auc)
            accuracy_means.append(mean_accuracy)
            sensitivity_means.append(mean_sensitivity)
            specificity_means.append(mean_specificity)
            significance_ratio_means.append(mean_significance_ratio)

            auc_stds.append(std_auc)
            accuracy_stds.append(std_accuracy)
            sensitivity_stds.append(std_sensitivity)
            specificity_stds.append(std_specificity)
            significance_ratio_stds.append(std_significance_ratio)

        # Calculate average metrics across comparisons
        avg_auc_mean = np.mean(auc_means)
        avg_accuracy_mean = np.mean(accuracy_means)
        avg_sensitivity_mean = np.mean(sensitivity_means)
        avg_specificity_mean = np.mean(specificity_means)
        avg_significance_ratio_mean = np.mean(significance_ratio_means)

        avg_auc_std = np.mean(auc_stds)
        avg_accuracy_std = np.mean(accuracy_stds)
        avg_sensitivity_std = np.mean(sensitivity_stds)
        avg_specificity_std = np.mean(specificity_stds)
        avg_significance_ratio_std = np.mean(significance_ratio_stds)

        # Prepare directories
        outputs_dir = PROJECT_ROOT / 'outputs'
        results_dir = outputs_dir / 'analysis_results'
        results_file = results_dir / 'performance_metrics.txt'

        # Write results to file
        with open(results_file, 'a') as f:
            f.write(f"Hyperparameters: {args}\n")
            f.write("Overall Performance Averages (Mean ± Std):\n")
            f.write(f"Mean ROC AUC: {avg_auc_mean:.4f} ± {avg_auc_std:.4f}\n")
            f.write(f"Mean Accuracy: {avg_accuracy_mean:.4f} ± {avg_accuracy_std:.4f}\n")
            f.write(f"Mean Sensitivity: {avg_sensitivity_mean:.4f} ± {avg_sensitivity_std:.4f}\n")
            f.write(f"Mean Specificity: {avg_specificity_mean:.4f} ± {avg_specificity_std:.4f}\n")
            f.write(f"Mean Significance Ratio: {avg_significance_ratio_mean:.4f} ± {avg_significance_ratio_std:.4f}\n")
            f.write("\n\n\n\n")
    elif args.action == 'analyze':
        analyze(args)

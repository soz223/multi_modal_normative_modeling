#!/home/songlinzhao/anaconda3/envs/normodiff/bin/python3
"""Unified script for training, testing, and group analysis using the supervised cVAE model."""

import argparse
from pathlib import Path
import random as rn
import pandas as pd
from sklearn.preprocessing import RobustScaler
import numpy as np
import tensorflow as tf
import torch
from os.path import join, exists
from tqdm import tqdm
from utils import COLUMNS_HCP, COLUMNS_NAME, load_dataset, COLUMNS_NAME_SNP, COLUMNS_NAME_VBM, COLUMNS_3MODALITIES, cliff_delta
from utils_vae import plot_losses, MyDataset_labels, Logger, MyDataset, reconstruction_deviation_seperate_roi
from cVAE import cVAE, cVAE_multimodal
from scipy import stats
from sklearn.metrics import roc_curve, auc
from numpy import interp, linspace
import matplotlib.pyplot as plt
import joblib

PROJECT_ROOT = Path.cwd()
result_baseline = './result_baseline/'

# Main training function
def train(dataset_resourse, hz_para_list, combine, procedure, epochs):
    """Train the supervised cVAE model on the bootstrapped samples."""
    n_bootstrap = 10
    model_name = 'supervised_cvae'
    bootstrap_dir = PROJECT_ROOT / 'outputs' / 'bootstrap_analysis'
    ids_dir = bootstrap_dir / 'ids'
    model_dir = bootstrap_dir / model_name
    model_dir.mkdir(exist_ok=True)

    # Set random seed
    random_seed = 42
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)
    rn.seed(random_seed)

    if dataset_resourse == 'ADNI':
        if procedure.startswith('SE'):
            dataset_names = ['av45', 'vbm', 'fdg']
        elif procedure.startswith('UCA'):
            dataset_names = ['av45', 'vbm', 'fdg', '3modalities']
        else:
            raise ValueError('Unknown procedure: {}'.format(procedure))
    elif dataset_resourse == 'HCP':
        dataset_names = ['T1_volume', 'mean_T1_intensity', 'mean_FA', 'mean_MD', 'mean_L1', 'mean_L2', 'mean_L3', 'min_BOLD', '25_percentile_BOLD', '50_percentile_BOLD', '75_percentile_BOLD', 'max_BOLD']
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset_resourse))


    if combine is None:
        raise ValueError(f'Unknown procedure: {procedure}')


    modalities = len(dataset_names)
    for i_bootstrap in range(n_bootstrap):
        ids_filename = 'cleaned_bootstrap_{:03d}.csv'.format(i_bootstrap)
        ids_path = ids_dir / ids_filename
        bootstrap_model_dir = model_dir / '{:03d}'.format(i_bootstrap)
        bootstrap_model_dir.mkdir(exist_ok=True)
        generator_train_list = []
        input_dim_list = []

        # Loading data
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
            dataset_df = load_dataset(participants_path, ids_path, freesurfer_path)

            if dataset_resourse == 'ADNI':
                hc_label = 2
            elif dataset_resourse == 'HCP':
                hc_label = 1
            else:
                raise ValueError('Unknown dataset resource')

            dataset_df = dataset_df.loc[dataset_df['DIA'] == hc_label]
            train_data = dataset_df[columns_name].values

            scaler = RobustScaler()
            train_data = scaler.fit_transform(train_data)
            train_data = pd.DataFrame(train_data)

            train_covariates = dataset_df[['DIA', 'PTGENDER', 'AGE']]
            train_covariates.DIA[train_covariates.DIA == 0] = 0

            bin_labels = list(range(0, 27))
            AGE_bins_train, bin_edges = pd.qcut(train_covariates['AGE'].rank(method="first"), q=27, retbins=True, labels=bin_labels)
            one_hot_AGE = np.eye(27)[AGE_bins_train.values]

            PTGENDER_bins_train, bin_edges = pd.qcut(train_covariates['PTGENDER'].rank(method="first"), q=2, retbins=True, labels=list(range(0, 2)))
            one_hot_PTGENDER = np.eye(2)[PTGENDER_bins_train.values]

            batch_size = 256
            torch.manual_seed(42)
            use_cuda = torch.cuda.is_available()
            if use_cuda:
                torch.cuda.manual_seed(42)
            DEVICE = torch.device("cuda:1" if use_cuda else "cpu")

            input_dim = train_data.shape[1]
            one_hot_covariates_train = np.concatenate((one_hot_AGE, one_hot_PTGENDER), axis=1).astype('float32')
            c_dim = one_hot_covariates_train.shape[1]
            train_dataset = MyDataset_labels(train_data.to_numpy(), one_hot_covariates_train)
            generator_train = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=False)
            generator_train_list.append(generator_train)
            input_dim_list.append(input_dim)

        h_dim = hz_para_list[:-1]
        z_dim = hz_para_list[-1]
        global_step = 0
        n_epochs = epochs
        gamma = 0.98
        scale_fn = lambda x: gamma ** x
        base_lr = 0.0000001
        max_lr = 0.000005
        n_samples = train_data.shape[0]

        print('train model')
        # model = cVAE_multimodal(input_dim_list=input_dim_list, hidden_dim=h_dim, latent_dim=z_dim, c_dim=c_dim, learning_rate=0.0001, modalities=modalities, non_linear=True)
        model = cVAE_multimodal(input_dim_list=input_dim_list, hidden_dim=h_dim, latent_dim=z_dim, c_dim=c_dim, learning_rate=0.0001, modalities=modalities, non_linear=True)
        model.to(DEVICE)
        step_size = 2 * np.ceil(n_samples / batch_size)

        for epoch in range(n_epochs):
            for batch_idx, batch in enumerate(zip(*generator_train_list)):
                global_step += 1
                cycle = np.floor(1 + global_step / (2 * step_size))
                x_lr = np.abs(global_step / step_size - 2 * cycle + 1)
                clr = base_lr + (max_lr - base_lr) * max(0, 1 - x_lr) * scale_fn(cycle)
                model.optimizer = torch.optim.Adam(list(model.encoder_list.parameters()) + list(model.decoder_list.parameters()), lr=clr)
                data_curr_list = []
                cov_list = []
                for modal in range(modalities):
                    data_curr = batch[modal][0].to(DEVICE)
                    cov = batch[modal][1].to(DEVICE)
                    data_curr_list.append(data_curr)
                    cov_list.append(cov)

                fwd_rtn = model.forward_multimodal(data_curr_list, cov_list, combine)
                loss = model.loss_function_cvae_multimodal(data_curr_list, fwd_rtn)
                model.optimizer.zero_grad()
                loss['total'].backward()
                model.optimizer.step()

                if batch_idx == 0:
                    to_print = 'Train Epoch:' + str(epoch) + ' ' + 'Train batch: ' + str(batch_idx) + ' ' + ', '.join([k + ': ' + str(round(v.item(), 3)) for k, v in loss.items()])
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

# Main testing function
def test(dataset_resourse, procedure):
    """Make predictions using trained normative models."""
    # ----------------------------------------------------------------------------
    n_bootstrap = 10
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

# Main group analysis function
def group_analysis(hz_para_list, hc_label, disease_label, combine, dataset_resourse, procedure, epochs):
    """Perform the group analysis."""
    n_bootstrap = 10
    model_name = 'supervised_cvae'
    participants_path = PROJECT_ROOT / 'data' / 'y.csv'
    bootstrap_dir = PROJECT_ROOT / 'outputs' / 'bootstrap_analysis'
    model_dir = bootstrap_dir / model_name
    ids_dir = bootstrap_dir / 'ids'
    tpr_list = []
    auc_roc_list = []
    accuracy_list = []
    recall_list = []
    specificity_list = []
    effect_size_list = []
    significance_ratio_list = []

    if procedure.startswith('SE'):
        dataset_names = ['av45', 'vbm', 'fdg']
    elif procedure.startswith('UCA'):
        dataset_names = ['av45', 'vbm', 'fdg', '3modalities']
    else:
        raise ValueError('Unknown procedure: {}'.format(procedure))

    if combine is None:
        raise ValueError(f'Unknown procedure: {procedure}')


    for i_bootstrap in tqdm(range(n_bootstrap)):
        bootstrap_model_dir = model_dir / '{:03d}'.format(i_bootstrap)
        if dataset_resourse == 'ADNI':
            dataset_names = ['av45', 'fdg', 'vbm']
        elif dataset_resourse == 'HCP':
            dataset_names = ['T1_volume', 'mean_T1_intensity', 'mean_FA', 'mean_MD', 'mean_L1', 'mean_L2', 'mean_L3', 'min_BOLD', '25_percentile_BOLD', '50_percentile_BOLD', '75_percentile_BOLD', 'max_BOLD']
        else:
            raise ValueError('Unknown dataset resource')

        reconstruction_error_df_list = []
        for dataset_name in dataset_names:
            output_dataset_dir = bootstrap_model_dir / dataset_name
            output_dataset_dir.mkdir(exist_ok=True)
            analysis_dir = output_dataset_dir / '{:02d}_vs_{:02d}'.format(hc_label, disease_label)
            analysis_dir.mkdir(exist_ok=True)
            test_ids_filename = 'cleaned_bootstrap_test_{:03d}.csv'.format(i_bootstrap)
            ids_path = ids_dir / test_ids_filename
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

        print('reconstruction_error_df_averaged: ', reconstruction_error_df_averaged)

        roc_auc, accuracy, recall, specificity, significance_ratio = compute_classification_performance(reconstruction_error_df_averaged, clinical_df, disease_label, hc_label)
        auc_roc_list.append(roc_auc)
        accuracy_list.append(accuracy)
        recall_list.append(recall)
        specificity_list.append(specificity)
        significance_ratio_list.append(significance_ratio)

    comparison_dir = bootstrap_dir / dataset_name / ('{:02d}_vs_{:02d}'.format(hc_label, disease_label))
    comparison_dir.mkdir(exist_ok=True)

    auc_roc_list = np.array(auc_roc_list)
    accuracy_list = np.array(accuracy_list)
    recall_list = np.array(recall_list)
    specificity_list = np.array(specificity_list)
    significance_ratio_list = auc_roc_list / (1 - auc_roc_list)

    if hc_label == 2 and disease_label == 0:
        compare_name = 'HC_vs_AD'
    elif hc_label == 2 and disease_label == 1:
        compare_name = 'HC_vs_MCI'
    elif hc_label == 1 and disease_label == 0:
        compare_name = 'MCI_vs_AD'

    with open(result_baseline + 'mm.txt', 'a') as f:
        # f.write('Experiment settings: CVAE. {}. Dataset {} Procedure {} Epochs {}\n'.format('HC_vs_AD', 'SE-'+('MoE' if combine == 'moe' else 'PoE'), procedure, epochs))
        f.write(f'Experiment settings: CVAE. Data resource: {dataset_resourse}. Procedure: {procedure}. compare_name: {compare_name}. Epochs: {epochs}\n')
        f.write('ROC-AUC: $ {:0.2f} \pm {:0.2f} $ \n'.format(np.mean(auc_roc_list) * 100, np.std(auc_roc_list) * 100))
        f.write('Accuracy: $ {:0.2f} \pm {:0.2f} $ \n'.format(np.mean(accuracy_list) * 100, np.std(accuracy_list) * 100))
        f.write('Sensitivity: $ {:0.2f} \pm {:0.2f} $ \n'.format(np.mean(recall_list) * 100, np.std(recall_list) * 100))
        f.write('Specificity: $ {:0.2f} \pm {:0.2f} $ \n'.format(np.mean(specificity_list) * 100, np.std(specificity_list) * 100))
        f.write('Significance ratio: $ {:0.2f} \pm {:0.2f} $ \n'.format(np.mean(significance_ratio_list), np.std(significance_ratio_list)))
        f.write('hz_para_list: ' + str(hz_para_list) + '\n')
        f.write('\n\n\n')
    np.savetxt("cvae_auc_and_std.csv", np.concatenate((auc_roc_list, [np.std(auc_roc_list)])), delimiter=",")
    auc_roc_df = pd.DataFrame(columns=['ROC-AUC'], data=auc_roc_list)
    auc_roc_df.to_csv(comparison_dir / 'auc_rocs.csv', index=False)

# Utility function for computing classification performance
def compute_classification_performance(reconstruction_error_df, clinical_df, disease_label, hc_label):
    """Calculate the AUCs and accuracy of the normative model."""
    # Align indices
    # reconstruction_error_df = reconstruction_error_df.reindex(clinical_df.index)
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
                        help='How do we combine all modalities.',
                        type=str)
    parser.add_argument('-P', '--procedure',
                        dest='procedure',
                        help='Procedure to perform the analysis.',
                        type=str)
    parser.add_argument('-E', '--epochs',
                        dest='epochs',
                        help='Number of epochs to train the model.',
                        type=int)

    args = parser.parse_args()

    if args.hz_para_list is None:
        args.hz_para_list = [110, 110, 10]
    if args.combine is None:
        args.combine = 'moe'
    if args.procedure is None:
        args.procedure = 'SE-MoE'
    if args.dataset_resourse is None:
        args.dataset_resourse = 'ADNI'
    if args.epochs is None:
        args.epochs = 2

    train(args.dataset_resourse, args.hz_para_list, args.combine, args.procedure, args.epochs)
    test(args.dataset_resourse, args.procedure)
    group_analysis(args.hz_para_list, 2, 0, args.combine, args.dataset_resourse, args.procedure, args.epochs)
    group_analysis(args.hz_para_list, 2, 1, args.combine, args.dataset_resourse, args.procedure, args.epochs)
    group_analysis(args.hz_para_list, 1, 0, args.combine, args.dataset_resourse, args.procedure, args.epochs)

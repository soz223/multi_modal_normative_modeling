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
from sklearn.model_selection import KFold, train_test_split

from utils import COLUMNS_HCP, COLUMNS_NAME, load_dataset, COLUMNS_NAME_SNP, COLUMNS_NAME_VBM, COLUMNS_3MODALITIES 
from utils_vae import plot_losses, MyDataset_labels, Logger
import torch
from cVAE import cVAE, cVAE_multimodal_endtoend, mmJSD, DMVAE, WeightedDMVAE, mvtCAE, cVAE_multimodal_endtoend

from os.path import join
from utils_vae import plot_losses, Logger, MyDataset
import argparse

def generate_kfold_ids(HC_group, other_group, oversample_percentage=1, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    kfold_dir = PROJECT_ROOT / 'outputs' / 'kfold_analysis'
    kfold_dir.mkdir(parents=True, exist_ok=True)
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(HC_group)):
        train_ids = HC_group.iloc[train_idx]['IID']
        test_ids_hc = HC_group.iloc[test_idx]['IID']

        oversample_percentage = oversample_percentage
        oversample_size = int(len(train_ids) * (oversample_percentage))
        train_ids_oversampled = np.random.choice(train_ids, size=oversample_size, replace=True)
        train_ids = pd.Series(train_ids_oversampled)
        # name the train_ids column as IID
        train_ids = pd.DataFrame(train_ids)
        train_ids.columns = ['IID']

        
        # Combine test ids from HC group and the other group
        test_ids_other = other_group['IID']
        test_ids = pd.concat([test_ids_hc, test_ids_other])

        # Split test_ids into 80% test and 20% validation
        # test_ids, val_ids = train_test_split(test_ids, test_size=0.2, random_state=42)

        train_ids_path = kfold_dir / f'train_ids_{fold:03d}.csv'
        test_ids_path = kfold_dir / f'test_ids_{fold:03d}.csv'
        # val_ids_path = kfold_dir / f'val_ids_{fold:03d}.csv'
        
        train_ids.to_csv(train_ids_path, index=False)
        test_ids.to_csv(test_ids_path, index=False)
        # val_ids.to_csv(val_ids_path, index=False)
        
        # print(f'Fold {fold} - Train IDs saved to: {train_ids_path}, Test IDs saved to: {test_ids_path}, Validation IDs saved to: {val_ids_path}')


PROJECT_ROOT = Path.cwd()
def main(dataset_resourse, hz_para_list, combine, procedure, epochs, oversample_percentage=1, n_splits=5):
    """Train the normative method using k-fold cross-validation."""
    # ----------------------------------------------------------------------------
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    model_name = 'supervised_cvae'

    # ----------------------------------------------------------------------------
    output_dir = PROJECT_ROOT / 'outputs' 
    output_dir.mkdir(exist_ok=True)
    kfold_dir = output_dir / 'kfold_analysis'
    kfold_dir.mkdir(exist_ok=True)
    model_dir = kfold_dir / model_name
    model_dir.mkdir(exist_ok=True)

    # ----------------------------------------------------------------------------
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
    
    modalities = len(dataset_names)

    participants_path = PROJECT_ROOT / 'data' / dataset_resourse / 'y.csv'
    ids_df = pd.read_csv(participants_path)

    if dataset_resourse == 'ADNI':
        hc_label = 2
    elif dataset_resourse == 'HCP':
        hc_label = 1
    else:
        raise ValueError('Unknown dataset resource')
    
    label_of_all = ids_df['DIA'].values

    HC_group = ids_df[ids_df['DIA'] == hc_label]
    label_of_HC_group = HC_group['DIA'].values

    # other group is the group of who is not 2
    other_group = ids_df[ids_df['DIA'] != hc_label]
    label_of_other_group = other_group['DIA'].values

    # generate_kfold_ids(HC_group, other_group, oversample_percentage=oversample_percentage, n_splits=n_splits)

    for fold, (train_idx, test_idx) in enumerate(kf.split(HC_group)):

        train_ids_path = kfold_dir / 'train_ids_{:03d}.csv'.format(fold)
        test_ids_path = kfold_dir / 'test_ids_{:03d}.csv'.format(fold)

        
        fold_model_dir = model_dir / '{:03d}'.format(fold)
        fold_model_dir.mkdir(exist_ok=True)
        generator_train_list = []
        input_dim_list = []

        # ----------------------------------------------------------------------------
        # Loading data

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

            train_dataset_df = load_dataset(participants_path, train_ids_path, freesurfer_path)
            test_dataset_df = load_dataset(participants_path, test_ids_path, freesurfer_path)

            # # read train_dataset_df and test_dataset_df's DIA column, and give stastics
            # print("train_dataset_df.DIA.value_counts()", train_dataset_df.DIA.value_counts())
            # print("test_dataset_df.DIA.value_counts()", test_dataset_df.DIA.value_counts())
            
            # # check train_dataset_df.IID and test_dataset_df.IID. have overlap or not
            # print("train_dataset_df.IID.isin(test_dataset_df.IID).sum()", train_dataset_df.IID.isin(test_dataset_df.IID).sum())


            train_dataset_df = train_dataset_df.loc[train_dataset_df['DIA'] == hc_label]
            train_data = train_dataset_df[columns_name].values

            scaler = RobustScaler()
            train_data = scaler.fit_transform(train_data)
            train_data = pd.DataFrame(train_data)
            
            train_covariates = train_dataset_df[['DIA','PTGENDER', 'AGE']]
            train_covariates.DIA[train_covariates.DIA == 0] = 0
      
            bin_labels = list(range(0,27))                          
            AGE_bins_train, bin_edges = pd.qcut(train_covariates['AGE'].rank(method="first"), q=27, retbins=True, labels=bin_labels)
            
            one_hot_AGE = np.eye(27)[AGE_bins_train.values]
            
            PTGENDER_bins_train, bin_edges = pd.qcut(train_covariates['PTGENDER'].rank(method="first"), q=2, retbins=True, labels=list(range(0,2)) )
            
            one_hot_PTGENDER = np.eye(2)[PTGENDER_bins_train.values]

            batch_size = 256
            n_samples = train_data.shape[0]

            torch.manual_seed(42)
            use_cuda =  torch.cuda.is_available()
            if use_cuda:
                torch.cuda.manual_seed(42)
            DEVICE = torch.device("cuda:1" if use_cuda else "cpu")
            
            input_dim = train_data.shape[1]
            one_hot_covariates_train = np.concatenate((one_hot_AGE, one_hot_PTGENDER), axis=1).astype('float32')
            
            c_dim = one_hot_covariates_train.shape[1]

            # now we are going to consider the label here. old label is actually the covariates, now is adding the true label
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
        base_lr = 0.000001
        max_lr = 0.00005

        print('train model')
        model = cVAE_multimodal_endtoend(input_dim_list=input_dim_list, hidden_dim=h_dim, latent_dim=z_dim, c_dim=c_dim, learning_rate=0.0001, modalities=modalities, non_linear=True)
        model.to(DEVICE)
        
        step_size = 2 * np.ceil(n_samples / batch_size)
        
        for epoch in range(n_epochs): 
            for batch_idx, batch in enumerate(zip(*generator_train_list)):
                global_step = global_step + 1
                cycle = np.floor(1 + global_step / (2 * step_size))
                x_lr = np.abs(global_step / step_size - 2 * cycle + 1)
                clr = base_lr + (max_lr - base_lr) * max(0, 1 - x_lr) * scale_fn(cycle)
                model.optimizer1.lr = clr
                # print the learning rate of the model's optimizer
                data_curr_list = []
                cov_list = []
                for modal in range(modalities):
                    data_curr = batch[modal][0].to(DEVICE)
                    cov = batch[modal][1].to(DEVICE)
                    data_curr_list.append(data_curr)
                    cov_list.append(cov)

                fwd_rtn = model.forward_multimodal(data_curr_list, cov_list, combine)
                loss = model.loss_function_multimodal(data_curr_list, fwd_rtn)
                
                # model.optimizer1.zero_grad()
                model.optimizer1.zero_grad()
                loss['total'].backward()
                model.optimizer1.step()

                if batch_idx == 0:
                    to_print = 'Train Epoch:' + str(epoch) + ' ' + 'Train batch: ' + str(batch_idx) + ' '+ ', '.join([k + ': ' + str(round(v.item(), 3)) for k, v in loss.items()])
                    print(to_print)        
                    if epoch == 0:
                        log_keys = list(loss.keys())
                        logger = Logger()
                        logger.on_train_init(log_keys)
                    else:
                        logger.on_step_fi(loss)
        plot_losses(logger, fold_model_dir, 'training')
        model_path = join(fold_model_dir, 'cVAE_model.pkl')
        torch.save(model, model_path)


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
                        help='how do we combine all modalities.',
                        type=str)
    
    parser.add_argument('-P', '--procedure',
                        dest='procedure',
                        help='Procedure to perform the analysis.',
                        type=str)

    parser.add_argument('-E', '--epochs',
                        dest='epochs',
                        help='Number of epochs to train the model.',
                        type=int)
    
    parser.add_argument('-K', '--n_splits',
                        dest='n_splits',
                        help='Number of splits for k-fold cross-validation.',
                        type=int, default=5)

    parser.add_argument('-O', '--oversample_percentage',
                        dest='oversample_percentage',
                        help='Percentage of oversampling of the training data.',
                        type=float, default=1)

    args = parser.parse_args()

    if args.hz_para_list is None:
        args.hz_para_list = [110, 110, 10]
    if args.procedure is None:
        args.procedure = 'SE-MoE'
    if args.combine is None:
        args.combine = args.procedure.split('-')[1]
    if args.dataset_resourse is None:
        args.dataset_resourse = 'ADNI'
    if args.epochs is None:
        args.epochs = 200

    main(args.dataset_resourse, args.hz_para_list, args.combine, args.procedure, args.epochs, args.oversample_percentage, args.n_splits)

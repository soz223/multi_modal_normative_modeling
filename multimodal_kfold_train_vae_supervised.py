#!/home/songlinzhao/anaconda3/envs/normodiff/bin/python3
"""Script to train the deterministic supervised adversarial autoencoder."""
from pathlib import Path
import random as rn
import pandas as pd
from sklearn.preprocessing import RobustScaler, OneHotEncoder
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold

from utils import COLUMNS_HCP, COLUMNS_NAME, load_dataset, COLUMNS_NAME_SNP, COLUMNS_NAME_VBM, COLUMNS_3MODALITIES
from utils_vae import plot_losses, MyDataset_labels, Logger
import torch
from VAE import VAE_multimodal

from os.path import join
from utils_vae import plot_losses, Logger, MyDataset
import argparse

PROJECT_ROOT = Path.cwd()

def main(dataset_resourse, hz_para_list, combine, procedure, epochs, n_splits=5):
    """Train the normative method using k-fold cross-validation."""
    model_name = 'supervised_vae'

    output_dir = PROJECT_ROOT / 'outputs'
    output_dir.mkdir(exist_ok=True)
    kfold_dir = output_dir / 'kfold_analysis'
    kfold_dir.mkdir(exist_ok=True)
    model_dir = kfold_dir / model_name
    model_dir.mkdir(exist_ok=True)

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
    participants_path = PROJECT_ROOT / 'data' / dataset_resourse / 'y.csv'
    ids_df = pd.read_csv(participants_path)
    HC_group = ids_df[ids_df['DIA'] == 2]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(kf.split(HC_group)):

        train_ids_path = kfold_dir / f'train_ids_{fold:03d}.csv'
        test_ids_path = kfold_dir / f'test_ids_{fold:03d}.csv'

        train_ids = pd.read_csv(train_ids_path)
        test_ids = pd.read_csv(test_ids_path)


        fold_model_dir = model_dir / '{:03d}'.format(fold)
        fold_model_dir.mkdir(exist_ok=True)
        generator_train_list = []
        input_dim_list = []

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
                columns_name = [(lambda x: dataset_name + '_' + str(x))(y) for y in list(range(132))]

            freesurfer_path = PROJECT_ROOT / 'data' / dataset_resourse / (dataset_name + '.csv')

            train_dataset_df = load_dataset(participants_path, train_ids_path, freesurfer_path)
            test_dataset_df = load_dataset(participants_path, test_ids_path, freesurfer_path)

            if dataset_resourse == 'ADNI':
                hc_label = 2
            elif dataset_resourse == 'HCP':
                hc_label = 1
            else:
                raise ValueError('Unknown dataset resource')

            train_dataset_df = train_dataset_df.loc[train_dataset_df['DIA'] == hc_label]
            train_data = train_dataset_df[columns_name].values

            scaler = RobustScaler()
            train_data = scaler.fit_transform(train_data)
            train_data = pd.DataFrame(train_data)

            train_covariates = train_dataset_df[['DIA', 'PTGENDER', 'AGE']]
            train_covariates.DIA[train_covariates.DIA == 0] = 0

            bin_labels = list(range(0, 27))
            AGE_bins_train, bin_edges = pd.qcut(train_covariates['AGE'].rank(method="first"), q=27, retbins=True, labels=bin_labels)
            one_hot_AGE = np.eye(27)[AGE_bins_train.values]

            PTGENDER_bins_train, bin_edges = pd.qcut(train_covariates['PTGENDER'].rank(method="first"), q=2, retbins=True, labels=list(range(0, 2)))
            one_hot_PTGENDER = np.eye(2)[PTGENDER_bins_train.values]

            batch_size = 256
            n_samples = train_data.shape[0]

            torch.manual_seed(42)
            use_cuda = torch.cuda.is_available()
            if use_cuda:
                torch.cuda.manual_seed(42)
            DEVICE = torch.device("cuda:1" if use_cuda else "cpu")

            input_dim = train_data.shape[1]
            one_hot_covariates_train = np.concatenate((one_hot_AGE, one_hot_PTGENDER), axis=1).astype('float32')

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

        print('train model')
        model = VAE_multimodal(input_dim_list=input_dim_list, hidden_dim=h_dim, latent_dim=z_dim, learning_rate=0.0001, modalities=modalities, non_linear=True)
        model.to(DEVICE)

        step_size = 2 * np.ceil(n_samples / batch_size)

        for epoch in range(n_epochs):
            for batch_idx, batch in enumerate(zip(*generator_train_list)):
                global_step = global_step + 1
                cycle = np.floor(1 + global_step / (2 * step_size))
                x_lr = np.abs(global_step / step_size - 2 * cycle + 1)
                clr = base_lr + (max_lr - base_lr) * max(0, 1 - x_lr) * scale_fn(cycle)
                model.optimizer_multimodal = torch.optim.Adam(
                    [p for model in model.encoder_list for p in model.parameters()] +
                    [p for model in model.decoder_list for p in model.parameters()] +
                    list(model.alpha_m_list.parameters()),
                    lr=clr
                )
                data_curr_list = []
                for modal in range(modalities):
                    data_curr = batch[modal][0].to(DEVICE)
                    data_curr_list.append(data_curr)

                fwd_rtn = model.forward_multimodal(data_curr_list, combine)
                loss = model.loss_function_multimodal(data_curr_list, fwd_rtn)

                model.optimizer_multimodal.zero_grad()
                loss[0]['total'].backward()
                model.optimizer_multimodal.step()

                if batch_idx == 0:
                    to_print = 'Train Epoch:' + str(epoch) + ' ' + 'Train batch: ' + str(batch_idx) + ' ' + ', '.join([k + ': ' + str(round(v.item(), 3)) for k, v in loss[0].items()])
                    print(to_print)
                    if epoch == 0:
                        log_keys = list(loss[0].keys())
                        logger = Logger()
                        logger.on_train_init(log_keys)
                    else:
                        logger.on_step_fi(loss[0])

        plot_losses(logger, fold_model_dir, 'training')
        model_path = join(fold_model_dir, 'VAE_model.pkl')
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
        args.epochs = 200

    main(args.dataset_resourse, args.hz_para_list, args.combine, args.procedure, args.epochs, args.n_splits)

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

from utils import COLUMNS_HCP, COLUMNS_NAME, load_dataset, COLUMNS_NAME_SNP, COLUMNS_NAME_VBM, COLUMNS_3MODALITIES, COLUMNS_NAME_AAL116, get_column_name, get_datasets_name, get_hc_label
from utils import generate_kfold_ids
from utils_vae import plot_losses, MyDataset_labels, Logger
import torch
from cVAE import cVAE, cVAE_multimodal, mmJSD, DMVAE, WeightedDMVAE, mvtCAE, mmVAEPlus

from os.path import join
from utils_vae import plot_losses, Logger, MyDataset
import argparse

PROJECT_ROOT = Path.cwd()
def main(args):
    """Train the normative method using k-fold cross-validation."""
    # ----------------------------------------------------------------------------
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=42)
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

    dataset_names = get_datasets_name(args.dataset_resourse, args.procedure)

    modalities = len(dataset_names)

    participants_path = PROJECT_ROOT / 'data' / args.dataset_resourse / 'y.csv'
    ids_df = pd.read_csv(participants_path)

    hc_label = get_hc_label(args.dataset_resourse)
    disease_label = 0

    if args.training_class == 'nm':
        training_class_label = hc_label
    else:
        training_class_label = disease_label
    
    training_class_group = ids_df[ids_df['DIA'] == training_class_label]

    # other group is the group of who is not training class
    other_group = ids_df[ids_df['DIA'] != training_class_label]

    generate_kfold_ids(training_class_group, other_group, oversample_percentage=args.oversample_percentage, n_splits=args.n_splits)

    for fold, (train_idx, test_idx) in enumerate(kf.split(training_class_group)):

        train_ids_path = kfold_dir / 'train_ids_{:03d}.csv'.format(fold)
        test_ids_path = kfold_dir / 'test_ids_{:03d}.csv'.format(fold)

        
        fold_model_dir = model_dir / '{:03d}'.format(fold)
        fold_model_dir.mkdir(exist_ok=True)
        generator_train_list = []
        input_dim_list = []

        # ----------------------------------------------------------------------------
        # Loading data

        for dataset_name in dataset_names:
            columns_name = get_column_name(args.dataset_resourse, dataset_name)


            freesurfer_path = PROJECT_ROOT / 'data' / args.dataset_resourse / (dataset_name + '.csv')

            train_dataset_df = load_dataset(participants_path, train_ids_path, freesurfer_path)
            test_dataset_df = load_dataset(participants_path, test_ids_path, freesurfer_path)

            # train_dataset_df = train_dataset_df.loc[train_dataset_df['DIA'] == hc_label]
            train_data = train_dataset_df[columns_name].values

            # train_data = (np.true_divide(train_data, tiv))
            # tiv is the total intracranial volume, which is each row's sum of the freesurfer data
            # tiv = train_data.sum(axis=1)
            
            # train_data = (np.true_divide(train_data, tiv[:, None]))


            scaler = RobustScaler()
            train_data = scaler.fit_transform(train_data)
            train_data = pd.DataFrame(train_data)
            
            train_covariates = train_dataset_df[['DIA','PTGENDER', 'AGE']]
      
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
            train_dataset = MyDataset_labels(train_data.to_numpy(), one_hot_covariates_train)    
            
            generator_train = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=False)
            
            generator_train_list.append(generator_train)
            input_dim_list.append(input_dim)

        h_dim = args.hz_para_list[:-1]
        z_dim = args.hz_para_list[-1]

        global_step = 0
        n_epochs = args.epochs
        gamma = 0.98
        scale_fn = lambda x: gamma ** x
        base_lr = args.base_learning_rate
        max_lr = args.max_learning_rate

        print('train model')


        # Define available models
        model_dict = {
            'cVAE_multimodal': cVAE_multimodal,
            'mmJSD': mmJSD,
            'DMVAE': DMVAE,
            'WeightedDMVAE': WeightedDMVAE,
            'mvtCAE': mvtCAE,
            'mmVAEPlus': mmVAEPlus
        }

        # Initialize the selected model with error handling
        try:
            model = model_dict[args.model](
                input_dim_list=input_dim_list,
                hidden_dim=h_dim,
                latent_dim=z_dim,
                c_dim=c_dim,
                learning_rate=0.0001,
                modalities=modalities,
                non_linear=True
            )
        except KeyError:
            raise ValueError(f"Model '{args.model}' is not recognized. Available models are: {', '.join(model_dict.keys())}")

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

                fwd_rtn = model.forward_multimodal(data_curr_list, cov_list, args.combine)
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
        print('fold_model_dir:', fold_model_dir)
        print('file saved at ', model_path)

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
    parser.add_argument('-Model', '--model',
                        default='cVAE_multimodal',
                        dest='model',
                        help='Model to use for training the data.',
                        type=str)
    
    parser.add_argument('-SingleModality', '--single_modality',
                        dest='single_modality',
                        default=None,
                        help='Single modality to use for training the data.',
                        type=str)
    
    parser.add_argument('-Baselearningrate', '--base_learning_rate',
                        dest='base_learning_rate',
                        help='Base learning rate for the model.',
                        type=float, default=0.0001)
    
    parser.add_argument('-Maxlearningrate', '--max_learning_rate',
                        dest='max_learning_rate',
                        help='Max learning rate for the model.',
                        type=float, default=0.005)
    
    # an argument called training_class, which is the class to train the model, default is nm, which is the normative modeling,another option is dm, which is the disease modeling
    parser.add_argument('-TrainingClass', '--training_class',
                        dest='training_class',
                        default='nm',
                        help='Class to train the model.',
                        type=str)
    
    
    

    args = parser.parse_args()

    if args.hz_para_list is None:
        args.hz_para_list = [110, 110, 10]
    if args.procedure is None:
        args.procedure = 'SE-gPoE'
    if args.combine is None:
        args.combine = args.procedure.split('-')[1]
    if args.dataset_resourse is None:
        args.dataset_resourse = 'ADNI'
    if args.epochs is None:
        args.epochs = 200

    main(args)

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
from utils import generate_kfold_ids, generate_kfold_ids_endtoend
from utils_vae import plot_losses, MyDataset_labels, Logger, MyDataset_labels_endtoend
import torch
from cVAE import cVAE, cVAE_multimodal, mmJSD, DMVAE, WeightedDMVAE, mvtCAE, mmVAEPlus, cVAE_multimodal_endtoend

from os.path import join
from utils_vae import plot_losses, Logger, MyDataset
import argparse

PROJECT_ROOT = Path.cwd()




def evaluate(model, test_loader_list, modalities, DEVICE):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batches in zip(*test_loader_list):
            xes = []
            cs = []
            labels = []
            for modal in range(modalities):
                data = batches[modal][0].to(DEVICE)
                cov = batches[modal][1].to(DEVICE)
                label = batches[modal][2].to(DEVICE)
                xes.append(data)
                cs.append(cov)
                labels.append(label)
            labels = labels[0]  # Assuming labels are the same across modalities
            logits = model.predict(xes, cs)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    # Compute metrics
    from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score, confusion_matrix

    accuracy = accuracy_score(all_labels, all_preds)
    try:
        auroc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auroc = float('nan')
    recall = recall_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    specificity = tn / (tn + fp)
    metrics = {
        'accuracy': accuracy,
        'auroc': auroc,
        'sensitivity': recall,
        'specificity': specificity,
        'f1_score': f1
    }
    return metrics




def process_dataset(dataset_df, columns_name, scaler=None, fit_scaler=False, hc_label=None, n_bins_age=27, n_bins_gender=2):
    """
    处理数据集，包括特征缩放、协变量编码和标签转换。

    参数:
    - dataset_df: pandas DataFrame，包含数据的原始数据集。
    - columns_name: list，所需的特征列名称。
    - scaler: sklearn Scaler 对象，默认为 None。如果提供，将使用该 scaler。
    - fit_scaler: bool，是否在当前数据集上拟合 scaler。
    - hc_label: int，健康组的标签。
    - n_bins_age: int，年龄分箱的数量。
    - n_bins_gender: int，性别分箱的数量。

    返回:
    - 处理后的数据 (numpy.ndarray)
    - 协变量 (numpy.ndarray)
    - 标签 (numpy.ndarray)
    - scaler 对象
    """
    data = dataset_df[columns_name].values

    if scaler is None:
        scaler = RobustScaler()
    
    if fit_scaler:
        data = scaler.fit_transform(data)
    else:
        data = scaler.transform(data)
    
    data = pd.DataFrame(data)

    covariates = dataset_df[['DIA', 'PTGENDER', 'AGE']]

    # 年龄分箱
    bin_labels_age = list(range(n_bins_age))
    AGE_bins, _ = pd.qcut(covariates['AGE'].rank(method="first"), q=n_bins_age, retbins=True, labels=bin_labels_age)
    one_hot_AGE = np.eye(n_bins_age)[AGE_bins.astype(int).values]

    # 性别分箱
    bin_labels_gender = list(range(n_bins_gender))
    PTGENDER_bins, _ = pd.qcut(covariates['PTGENDER'].rank(method="first"), q=n_bins_gender, retbins=True, labels=bin_labels_gender)
    one_hot_PTGENDER = np.eye(n_bins_gender)[PTGENDER_bins.astype(int).values]

    one_hot_covariates = np.concatenate((one_hot_AGE, one_hot_PTGENDER), axis=1).astype('float32')

    # 标签转换：健康=0，疾病=1
    labels = covariates['DIA'].apply(lambda x: 0 if x == hc_label else 1).values

    return data.to_numpy(), one_hot_covariates, labels, scaler



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
    # hc_label = disease_label

    HC_group = ids_df[ids_df['DIA'] == hc_label]

    # other group is the group of who is not HC
    other_group = ids_df[ids_df['DIA'] != hc_label]

    all_metrics = []

    generate_kfold_ids_endtoend(HC_group, other_group, oversample_percentage=args.oversample_percentage, n_splits=args.n_splits)
    # 在主循环中使用 process_dataset 函数
    for fold, (train_idx, test_idx) in enumerate(kf.split(HC_group)):
        train_ids_path = kfold_dir / f'train_ids_{fold:03d}.csv'
        test_ids_path = kfold_dir / f'test_ids_{fold:03d}.csv'

        fold_model_dir = model_dir / f'{fold:03d}'
        fold_model_dir.mkdir(exist_ok=True)
        generator_train_list = []
        generator_test_list = []
        input_dim_list = []

        scaler = None  # 初始化 scaler

        for dataset_name in dataset_names:
            columns_name = get_column_name(args.dataset_resourse, dataset_name)
            freesurfer_path = PROJECT_ROOT / 'data' / args.dataset_resourse / f'{dataset_name}.csv'

            train_dataset_df = load_dataset(participants_path, train_ids_path, freesurfer_path)
            test_dataset_df = load_dataset(participants_path, test_ids_path, freesurfer_path)

            # 处理训练数据
            train_data, train_covariates, train_labels, scaler = process_dataset(
                dataset_df=train_dataset_df,
                columns_name=columns_name,
                scaler=scaler,
                fit_scaler=True,  # 仅在训练集上拟合 scaler
                hc_label=hc_label
            )

            batch_size = 256
            n_samples = train_data.shape[0]

            torch.manual_seed(42)
            use_cuda = torch.cuda.is_available()
            if use_cuda:
                torch.cuda.manual_seed(42)
            DEVICE = torch.device("cuda:1" if use_cuda else "cpu")

            input_dim = train_data.shape[1]
            one_hot_covariates_train = train_covariates

            c_dim = one_hot_covariates_train.shape[1]
            train_dataset = MyDataset_labels_endtoend(train_data, one_hot_covariates_train, train_labels)

            generator_train = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=False)
            generator_train_list.append(generator_train)
            input_dim_list.append(input_dim)

            # 处理测试数据
            test_data, test_covariates, test_labels, _ = process_dataset(
                dataset_df=test_dataset_df,
                columns_name=columns_name,
                scaler=scaler,  # 使用相同的 scaler
                fit_scaler=False,
                hc_label=hc_label
            )

            one_hot_covariates_test = test_covariates

            test_dataset = MyDataset_labels_endtoend(test_data, one_hot_covariates_test, test_labels)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            generator_test_list.append(test_loader)


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
            model = cVAE_multimodal_endtoend(
                input_dim_list=input_dim_list,
                hidden_dim=h_dim,
                latent_dim=z_dim,
                c_dim=c_dim,
                # learning_rate=learning_rate,
                modalities=modalities,
                non_linear=True,
                # classifier_layers=[128, 64, 32],
                classifier_layers=args.layers,
                dropout_rate=0.5,
                num_classes=2
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
                model.optimizer.lr = clr
                # print the learning rate of the model's optimizer
                data_curr_list = []
                cov_list = []
                label_list = []
                for modal in range(modalities):
                    data_curr = batch[modal][0].to(DEVICE)
                    cov = batch[modal][1].to(DEVICE)
                    label = batch[modal][2].to(DEVICE)
                    data_curr_list.append(data_curr)
                    cov_list.append(cov)
                    label_list.append(label)
                labels = label_list[0]

                fwd_rtn = model.forward(data_curr_list, cov_list)
                loss = model.loss_function(data_curr_list, fwd_rtn, labels, args.margin, args.weightcontrastive)
                
                # model.optimizer1.zero_grad()
                model.optimizer.zero_grad()
                loss['total_loss'].backward()
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
        plot_losses(logger, fold_model_dir, 'training')
        model_path = join(fold_model_dir, 'cVAE_model.pkl')
        torch.save(model, model_path)

        # Evaluate the model
        metrics = evaluate(model, generator_test_list, modalities, DEVICE)
        print(f"Fold {fold} metrics:")
        print(metrics)

        all_metrics.append(metrics)
    # metrics of all folds' mean
    all_metrics_df = pd.DataFrame(all_metrics)
    # all_metrics_df.to_csv(join(model_dir, 'all_metrics.csv'))
    print(all_metrics_df.mean())
    print(all_metrics_df.std())
    # save the mean metrics to results_endtoend.csv, with args as the first row
    results_path = join('./', 'results_endtoend.csv')
    with open(results_path, 'a') as f:
        f.write(str(args) + '\n')
        # all_metrics_df.mean().to_csv(f, header=False)
        # each metrics in the form of metric_name $mean \pm std$
        for metric in all_metrics_df.mean().index:
            # f.write(f'{all_metrics_df.mean()[metric]:.3f} ± {all_metrics_df.std()[metric]:.3f}\n')
            f.write(f'{metric} ${all_metrics_df.mean()[metric]:.3f} \pm {all_metrics_df.std()[metric]:.3f}$\n')
        f.write('\n\n\n')
        



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
    
    # learning_rate for mlp classifier
    parser.add_argument('-Learningrateclassifier', '--learning_rate_classifier',
                        dest='learning_rate_classifier',
                        help='Learning rate for the classifier.',
                        type=float, default=0.001)
    
    # margin for the contrastive loss
    parser.add_argument('-Margin', '--margin',
                        dest='margin',
                        help='Margin for the contrastive loss.',
                        type=float, default=1)
    
    # weightcontrastive for the contrastive loss, or alpha here
    parser.add_argument('-Weightcontrastive', '--weightcontrastive',
                        dest='weightcontrastive',
                        help='weight for the contrastive loss.',
                        type=float, default=1)
    
    # weight for the kl divergence loss
    parser.add_argument('-Weightkl', '--weight_kl',
                        dest='weight_kl',
                        help='Weight for the kl divergence loss.',
                        type=float, default=1)
    
    # weight for the reconstruction loss
    parser.add_argument('-Weightrec', '--weight_rec',
                        dest='weight_rec',
                        help='Weight for the reconstruction loss.',
                        type=float, default=1)
    

    # dropout rate for the mlp classifier
    parser.add_argument('-Dropout', '--dropout',
                        dest='dropout',
                        help='Dropout rate for the classifier.',
                        type=float, default=0.5)
    
    # layers for the mlp classifier
    parser.add_argument('-Layers', '--layers',
                        dest='layers',
                        nargs='+',
                        help='Layers for the classifier.',
                        default=[128, 64, 32],
                        type=int)
    
    
    

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

    if args.procedure.startswith('SingleModality'):
        if args.dataset_resourse == 'ADNI':
            args.single_modality = 'av45'
        elif args.dataset_resourse == 'HCP':
            args.single_modality = 'T1_volume'
        else:
            raise ValueError('Unknown dataset resource')
        

    main(args)

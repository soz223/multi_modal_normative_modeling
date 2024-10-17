#!/usr/bin/env python3
"""Script to perform the group analysis.

Creates the figures 3 and 4 from the paper

References:
    https://towardsdatascience.com/an-introduction-to-the-bootstrap-method-58bcb51b4d60
    https://machinelearningmastery.com/calculate-bootstrap-confidence-intervals-machine-learning-results-python/
    https://stats.stackexchange.com/questions/186337/average-roc-for-repeated-10-fold-cross-validation-with-probability-estimates
"""
import argparse
from pathlib import Path
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
from numpy import interp, linspace
from sklearn.model_selection import train_test_split


from utils import COLUMNS_HCP, COLUMNS_NAME, load_dataset, cliff_delta, COLUMNS_NAME_SNP, COLUMNS_NAME_VBM, COLUMNS_3MODALITIES, COLUMNS_NAME_AAL116
from sklearn.model_selection import KFold

PROJECT_ROOT = Path.cwd()
result_baseline = './result_baseline/'




def compute_classification_thresholds(reconstruction_error_df, clinical_df, disease_label, hc_label):
    """ Calculate the AUCs and accuracy of the normative model."""
    error_hc = reconstruction_error_df.loc[clinical_df['DIA'] == hc_label]['Reconstruction error']
    error_patient = reconstruction_error_df.loc[clinical_df['DIA'] == disease_label]['Reconstruction error']

    labels = list(np.zeros_like(error_hc)) + list(np.ones_like(error_patient))
    predictions = list(error_hc) + list(error_patient)
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)

    # Find the optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # Compute accuracy using the optimal threshold
    predicted_labels = [1 if e > optimal_threshold else 0 for e in predictions]

    accuracy = (np.array(predicted_labels) == np.array(labels)).mean()

    return roc_auc, accuracy, optimal_threshold
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve
import numpy as np

def find_best_threshold_by_f1(labels, predictions):
    thresholds = np.linspace(0, 1, 100)
    best_threshold = 0
    best_f1 = 0
    
    for threshold in thresholds:
        predicted_labels = (predictions >= threshold).astype(int)
        f1 = f1_score(labels, predicted_labels)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            
    return best_threshold, best_f1

def find_best_threshold_by_pr(labels, predictions):
    precision, recall, thresholds = precision_recall_curve(labels, predictions)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    best_threshold = thresholds[np.argmax(f1_scores)]
    return best_threshold

def find_best_threshold_by_cost(labels, predictions, cost_fn, cost_fp):
    thresholds = np.linspace(0, 1, 100)
    best_threshold = 0
    best_cost = float('inf')
    
    for threshold in thresholds:
        predicted_labels = (predictions >= threshold).astype(int)
        fp = np.sum((predicted_labels == 1) & (labels == 0))
        fn = np.sum((predicted_labels == 0) & (labels == 1))
        total_cost = fp * cost_fp + fn * cost_fn
        if total_cost < best_cost:
            best_cost = total_cost
            best_threshold = threshold
            
    return best_threshold, best_cost

def find_best_threshold_by_eer(labels, predictions):
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer_threshold

def compute_classification_performance(reconstruction_error_df, clinical_df, disease_label, hc_label, optimal_threshold=None, method='roc'):
    """Calculate the AUCs and accuracy of the normative model."""

    # print('reconstruction_error_df:', reconstruction_error_df)
    # print('clinical_df:', clinical_df)


    error_hc = reconstruction_error_df.loc[clinical_df['DIA'] == hc_label]['Reconstruction error']
    error_patient = reconstruction_error_df.loc[clinical_df['DIA'] == disease_label]['Reconstruction error']

    labels = list(np.zeros_like(error_hc)) + list(np.ones_like(error_patient))
    predictions = list(error_hc) + list(error_patient)

    fpr, tpr, thresholds = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)

    print('thresholds:', thresholds)

    if optimal_threshold is None:
        if method == 'roc':
            # Find the optimal threshold using Youden's J statistic
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
        elif method == 'f1':
            optimal_threshold, _ = find_best_threshold_by_f1(labels, predictions)
        elif method == 'pr':
            optimal_threshold = find_best_threshold_by_pr(labels, predictions)
        elif method == 'cost':
            optimal_threshold, _ = find_best_threshold_by_cost(labels, predictions, cost_fn=1, cost_fp=1)
        elif method == 'eer':
            optimal_threshold = find_best_threshold_by_eer(labels, predictions)
        else:
            raise ValueError("Unknown method for finding optimal threshold")

    predicted_labels = (np.array(predictions) >= optimal_threshold).astype(int)
    accuracy = (np.array(predicted_labels) == np.array(labels)).mean()

    TP = np.sum((predicted_labels == 1) & (np.array(labels) == 1))
    FN = np.sum((predicted_labels == 0) & (np.array(labels) == 1))
    TN = np.sum((predicted_labels == 0) & (np.array(labels) == 0))
    FP = np.sum((predicted_labels == 1) & (np.array(labels) == 0))

    recall = TP / (TP + FN)
    specificity = TN / (TN + FP)

    significance_ratio = roc_auc / (1 - roc_auc)

    return roc_auc, accuracy, recall, specificity, significance_ratio


def main(hz_para_list, hc_label, disease_label, combine, dataset_resourse, procedure, epochs, oversample_percentage, method='roc', n_splits=5):
    """Perform the group analysis."""
    model_name = 'supervised_cvae'

    participants_path = PROJECT_ROOT / 'data' / dataset_resourse / 'y.csv'
    outputs_dir = PROJECT_ROOT / 'outputs'
    kfold_dir = outputs_dir / 'kfold_analysis'
    model_dir = kfold_dir / model_name
    ids_dir = kfold_dir / 'ids'

    auc_roc_list = []
    accuracy_list = []
    sensitivity_list = []
    specificity_list = []
    significance_ratio_list = []

    if dataset_resourse == 'ADNI':
        if procedure.startswith('SingleModality'):
            dataset_names = ['av45']
        elif procedure.startswith('SE'):
            dataset_names = ['av45', 'vbm', 'fdg']
        elif procedure.startswith('UCA'):
            dataset_names = ['av45', 'vbm', 'fdg', '3modalities']
        else:
            raise ValueError('Unknown procedure: {}'.format(procedure))
    elif dataset_resourse == 'HCP':
        dataset_names = ['T1_volume', 'mean_T1_intensity', 'mean_FA', 'mean_MD', 'mean_L1', 'mean_L2', 'mean_L3', 'min_BOLD', '25_percentile_BOLD', '50_percentile_BOLD', '75_percentile_BOLD', 'max_BOLD']
    elif dataset_resourse == 'ADHD':
        dataset_names = ['sMRI', 'fMRI']
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset_resourse))

    if combine is None:
        raise ValueError(f'Unknown procedure: {procedure}')

    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    ids_df = pd.read_csv(participants_path)
    HC_group = ids_df[ids_df['DIA'] == hc_label]

    for fold, (train_idx, test_idx) in enumerate(kf.split(HC_group)):

        train_ids_path = kfold_dir / 'train_ids_{:03d}.csv'.format(fold)
        test_ids_path = kfold_dir / 'test_ids_{:03d}.csv'.format(fold)
        
        fold_model_dir = model_dir / '{:03d}'.format(fold)
        fold_model_dir.mkdir(exist_ok=True)
        
        reconstruction_error_df_list = []

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
            elif dataset_resourse == 'ADHD':
                columns_name = COLUMNS_NAME_AAL116

            freesurfer_path = PROJECT_ROOT / 'data' / dataset_resourse / (dataset_name + '.csv')

            # what does this do is, it loads the dataset and then sets the index to participant_id, and only filter those from the test_ids
            test_dataset_df = load_dataset(participants_path, test_ids_path, freesurfer_path)
            
            test_dataset_df = test_dataset_df.set_index('participant_id')

            normalized_df = pd.read_csv(fold_model_dir / dataset_name / 'normalized_{}.csv'.format(dataset_name), index_col='participant_id')
            reconstruction_df = pd.read_csv(fold_model_dir / dataset_name / 'reconstruction_{}.csv'.format(dataset_name), index_col='participant_id')
            reconstruction_error_df = pd.read_csv(fold_model_dir / dataset_name / 'reconstruction_error_{}.csv'.format(dataset_name), index_col='participant_id')
            reconstruction_error_df_list.append(reconstruction_error_df)

        reconstruction_error_df_averaged = reconstruction_error_df_list[0]
        for i in range(1, len(reconstruction_error_df_list)):
            reconstruction_error_df_averaged += reconstruction_error_df_list[i]
            # print('reconstruction_error_df_averaged:', reconstruction_error_df_averaged)
        reconstruction_error_df_averaged /= len(reconstruction_error_df_list)

        # print('reconstruction_error_df_averaged:', reconstruction_error_df_averaged)

        # print('reconstruction_error_df_list:', reconstruction_error_df_list[0].shape)
        # print('reconstruction_error_df_list:', reconstruction_error_df_list[1].shape)
        


        test_ids = test_dataset_df.index

        # print('test_ids:', test_ids)

        # # seperate 20% of test data for validation, randomly, using 42 as seed
        # test_dataset_df, validation_dataset_df = train_test_split(test_dataset_df, test_size=0, random_state=42)

        # reconstruction_error_df_averaged_test = reconstruction_error_df_averaged.loc[test_dataset_df.index]
        reconstruction_error_df_averaged_test = reconstruction_error_df_averaged

        # reconstruction_error_df_averaged_validation = reconstruction_error_df_averaged.loc[validation_dataset_df.index]
        # roc_auc, accuracy, optimal_threshold = compute_classification_thresholds(reconstruction_error_df_averaged_validation, validation_dataset_df, disease_label, hc_label)

        # print('optimal_threshold:', optimal_threshold)

        print('reconstruction_error_df_averaged_test:', reconstruction_error_df_averaged_test)

        # roc_auc, accuracy, recall, specificity, significance_ratio = compute_classification_performance(reconstruction_error_df_averaged, test_dataset_df, disease_label, hc_label)
        roc_auc, accuracy, recall, specificity, significance_ratio = compute_classification_performance(reconstruction_error_df_averaged_test, test_dataset_df, disease_label, hc_label, method='roc')
        auc_roc_list.append(roc_auc)
        accuracy_list.append(accuracy)
        sensitivity_list.append(recall)
        specificity_list.append(specificity)
        significance_ratio_list.append(significance_ratio)



    (kfold_dir / dataset_name).mkdir(exist_ok=True)
    comparison_dir = kfold_dir / dataset_name / ('{:02d}_vs_{:02d}'.format(hc_label, disease_label))
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

    
    with open(result_baseline + 'result_multimodal.txt', 'a') as f:
        # f.write('Experiment settings: CVAE. {}. Procedure {} Epochs {} Oversample percentage {}\n'.format(compare_name, 'SE-'+('MoE' if combine == 'moe' else 'PoE'), procedure, epochs, oversample_percentage))
        f.write('Experiment settings: CVAE. {}. Procedure {} Epochs {} Oversample percentage {}\n args.Model {}\n'.format(compare_name, procedure, epochs, oversample_percentage, args.model))
        f.write('ROC-AUC: $ {:0.2f} \pm {:0.2f} $ \n'.format(np.mean(auc_roc_list) * 100, np.std(auc_roc_list) * 100))
        f.write('Accuracy: $ {:0.2f} \pm {:0.2f} $ \n'.format(np.mean(accuracy_list) * 100, np.std(accuracy_list) * 100))
        f.write('Sensitivity: $ {:0.2f} \pm {:0.2f} $ \n'.format(np.mean(sensitivity_list) * 100, np.std(sensitivity_list) * 100))
        f.write('Specificity: $ {:0.2f} \pm {:0.2f} $ \n'.format(np.mean(specificity_list) * 100, np.std(specificity_list) * 100))
        f.write('Significance ratio: $ {:0.2f} \pm {:0.2f} $ \n'.format(np.mean(significance_ratio_list), np.std(significance_ratio_list)))
        f.write('hz_para_list: ' + str(hz_para_list) + '\n')
        # for i in range(len(auc_roc_list)):
        #     f.write('Fold {}: ROC-AUC: {:0.2f} Accuracy: {:0.2f} Sensitivity: {:0.2f} Specificity: {:0.2f} Significance ratio: {:0.2f}\n'.format(i, auc_roc_list[i] * 100, accuracy_list[i] * 100, sensitivity_list[i] * 100, specificity_list[i] * 100, significance_ratio_list[i]))
        f.write('\n\n\n')  
    np.savetxt("cvae_auc_and_std.csv", np.concatenate((auc_roc_list, [np.std(auc_roc_list)])), delimiter=",")
    auc_roc_df = pd.DataFrame(columns=['ROC-AUC'], data=auc_roc_list)
    auc_roc_df.to_csv(comparison_dir / 'auc_rocs.csv', index=False)

    return np.mean(auc_roc_list), np.std(auc_roc_list), \
           np.mean(accuracy_list), np.std(accuracy_list), \
           np.mean(sensitivity_list), np.std(sensitivity_list), \
           np.mean(specificity_list), np.std(specificity_list), \
           np.mean(significance_ratio_list), np.std(significance_ratio_list)

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


    args = parser.parse_args()

    if args.hz_para_list is None:
        args.hz_para_list = [110, 110, 10]
    if args.procedure is None:
        args.procedure = 'SE-MoE'
    if args.dataset_resourse is None:
        args.dataset_resourse = 'ADNI'
    if args.combine is None:
        args.combine = args.procedure.split('-')[1]
    if args.epochs is None:
        args.epochs = 200

    if args.dataset_resourse == 'ADNI':
        hc_patient_comb_list = [[2, 0], [2, 1], [1, 0]]
    elif args.dataset_resourse == 'HCP':
        hc_patient_comb_list = [[1, 0]]
    elif args.dataset_resourse == 'ADHD':
        hc_patient_comb_list = [[1, 0]]

    mean_auc_roc_list = []
    std_auc_roc_list = []
    mean_accuracy_list = []
    std_accuracy_list = []
    mean_recall_list = []
    std_recall_list = []
    mean_specificity_list = []
    std_specificity_list = []
    mean_significance_ratio_list = []
    std_significance_ratio_list = []

    # find_threshold_methods = ['roc', 'f1', 'pr', 'cost', 'eer']

    for hc_patient_comb in hc_patient_comb_list:
        mean_auc_roc, std_auc_roc, mean_accuracy, std_accuracy, mean_recall, std_recall, mean_specificity, std_specificity, mean_significance_ratio, std_significance_ratio = main(args.hz_para_list, hc_patient_comb[0], hc_patient_comb[1], args.combine, args.dataset_resourse, args.procedure, args.epochs, args.oversample_percentage, args.n_splits)

        mean_auc_roc_list.append(mean_auc_roc)
        std_auc_roc_list.append(std_auc_roc)
        mean_accuracy_list.append(mean_accuracy)
        std_accuracy_list.append(std_accuracy)
        mean_recall_list.append(mean_recall)
        std_recall_list.append(std_recall)
        mean_specificity_list.append(mean_specificity)
        std_specificity_list.append(std_specificity)
        mean_significance_ratio_list.append(mean_significance_ratio)
        std_significance_ratio_list.append(std_significance_ratio)

    with open(os.path.join(result_baseline, 'result_4.txt'), 'a') as f:
        f.write('Experiment settings: CVAE. {}. Procedure {} Epochs {} Oversample percentage {}\n'.format('HC vs AD, HC vs MCI, MCI vs AD', args.procedure, args.epochs, args.oversample_percentage))
        f.write('ROC-AUC: $ {:0.2f} \pm {:0.2f} $ \n'.format(np.mean(mean_auc_roc_list) * 100, np.mean(std_auc_roc_list) * 100))
        f.write('Accuracy: $ {:0.2f} \pm {:0.2f} $ \n'.format(np.mean(mean_accuracy_list) * 100, np.mean(std_accuracy_list) * 100))
        f.write('Sensitivity: $ {:0.2f} \pm {:0.2f} $ \n'.format(np.mean(mean_recall_list) * 100, np.mean(std_recall_list) * 100))
        f.write('Specificity: $ {:0.2f} \pm {:0.2f} $ \n'.format(np.mean(mean_specificity_list) * 100, np.mean(std_specificity_list) * 100))
        f.write('Significance ratio: $ {:0.2f} \pm {:0.2f} $ \n'.format(np.mean(mean_significance_ratio_list), np.mean(std_significance_ratio_list)))
        f.write('hz_para_list: ' + str(args.hz_para_list) + '\n')
        f.write('\n\n\n')

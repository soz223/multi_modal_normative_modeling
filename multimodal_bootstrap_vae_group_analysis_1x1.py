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

from utils import COLUMNS_HCP, COLUMNS_NAME, load_dataset, cliff_delta, COLUMNS_NAME_SNP, COLUMNS_NAME_VBM, COLUMNS_3MODALITIES
import argparse

PROJECT_ROOT = Path.cwd()
result_baseline = './result_baseline/'

def compute_brain_regions_deviations(diff_df, clinical_df, disease_label, hc_label):
    """ Calculate the Cliff's delta effect size between groups."""
    # Initialize an empty DataFrame to store the results
    region_df = pd.DataFrame(columns=['regions', 'pvalue', 'effect_size'])

    # Select data based on the disease label and the healthy control label
    diff_hc = diff_df.loc[clinical_df['DIA'] == disease_label]
    diff_patient = diff_df.loc[clinical_df['DIA'] == hc_label]

    # List to hold data frames for each region, which will be concatenated later
    df_list = []

    for region in COLUMNS_NAME:
        _, pvalue = stats.mannwhitneyu(diff_hc[region], diff_patient[region])
        effect_size = cliff_delta(diff_hc[region].values, diff_patient[region].values)

        # Append the result as a new DataFrame to df_list
        df_list.append(pd.DataFrame({'regions': [region], 'pvalue': [pvalue], 'effect_size': [effect_size]}))

    # Concatenate all DataFrames in df_list into region_df
    region_df = pd.concat(df_list, ignore_index=True)

    return region_df

def compute_classification_performance(reconstruction_error_df, clinical_df, disease_label, hc_label):
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
    accuracy_in_hc = (np.array(predicted_labels)[np.array(labels) == 0] == 0).mean()
    accuracy_in_ad = (np.array(predicted_labels)[np.array(labels) == 1] == 1).mean()

    TP = np.sum((np.array(predicted_labels) == 1) & (np.array(labels) == 1))
    FN = np.sum((np.array(predicted_labels) == 0) & (np.array(labels) == 1))
    TN = np.sum((np.array(predicted_labels) == 0) & (np.array(labels) == 0))
    FP = np.sum((np.array(predicted_labels) == 1) & (np.array(labels) == 0))

    print('TP:', TP)
    print('FN:', FN)
    print('TN:', TN)
    print('FP:', FP)

    recall = TP / (TP + FN)
    specificity = TN / (TN + FP)

    significance_ratio = roc_auc / (1 - roc_auc)

    fixed_points = 100
    base_fpr = linspace(0, 1, fixed_points)
    interp_tpr = interp(base_fpr, fpr, tpr)
    interp_tpr[0] = 0.0  # Ensuring that the curve starts at 0

    sensitivity = recall

    return roc_auc, accuracy, sensitivity, specificity, significance_ratio

def main(hz_para_list, hc_label, disease_label, combine, dataset_resourse, procedure, epochs):
    """Perform the group analysis."""
    # ----------------------------------------------------------------------------
    n_bootstrap = 10

    model_name = 'supervised_vae'

    participants_path = PROJECT_ROOT / 'data' / 'y.csv'

    # ----------------------------------------------------------------------------
    bootstrap_dir = PROJECT_ROOT / 'outputs' / 'bootstrap_analysis'
    model_dir = bootstrap_dir / model_name
    ids_dir = bootstrap_dir / 'ids'

    tpr_list = []
    auc_roc_list = []
    accuracy_list = []
    recall_list = []
    specificity_list = []
    effect_size_list = []

    auc_roc_list = []
    accuracy_list = []
    sensitivity_list = []
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
                dataset_name = dataset_names[i_dataset]
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

            normalized_df = pd.read_csv(output_dataset_dir / 'normalized_{}.csv'.format(dataset_name), index_col='participant_id')
            reconstruction_df = pd.read_csv(output_dataset_dir / 'reconstruction_{}.csv'.format(dataset_name), index_col='participant_id')
            reconstruction_error_df = pd.read_csv(output_dataset_dir / 'reconstruction_error_{}.csv'.format(dataset_name), index_col='participant_id')
            reconstruction_error_df_list.append(reconstruction_error_df)

            print('reconstruction_error_df path:', output_dataset_dir / 'reconstruction_error_{}.csv'.format(dataset_name))

        reconstruction_error_df_averaged = reconstruction_error_df_list[0]
        for i in range(1, len(reconstruction_error_df_list)):
            reconstruction_error_df_averaged += reconstruction_error_df_list[i]
        reconstruction_error_df_averaged /= len(reconstruction_error_df_list)

        roc_auc, accuracy, recall, specificity, significance_ratio = compute_classification_performance(reconstruction_error_df_averaged, clinical_df, disease_label, hc_label)
        auc_roc_list.append(roc_auc)
        accuracy_list.append(accuracy)
        sensitivity_list.append(recall)
        specificity_list.append(specificity)
        significance_ratio_list.append(significance_ratio)

    (bootstrap_dir / dataset_name).mkdir(exist_ok=True)
    comparison_dir = bootstrap_dir / dataset_name / '{:02d}_vs_{:02d}'.format(hc_label, disease_label)
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

    print('so far so good')
    with open(result_baseline + 'result_multimodal.txt', 'a') as f:
        f.write('Experiment settings: VAE. {}. Dataset {} Procedure {} Epochs {}\n'.format(compare_name, 'SE-'+('MoE' if combine == 'moe' else 'PoE'), procedure, epochs))
        f.write('ROC-AUC: $ {:0.2f} \pm {:0.2f} $ \n'.format(np.mean(auc_roc_list) * 100, np.std(auc_roc_list) * 100))
        f.write('Accuracy: $ {:0.2f} \pm {:0.2f} $ \n'.format(np.mean(accuracy_list) * 100, np.std(accuracy_list) * 100))
        f.write('Sensitivity: $ {:0.2f} \pm {:0.2f} $ \n'.format(np.mean(sensitivity_list) * 100, np.std(sensitivity_list) * 100))
        f.write('Specificity: $ {:0.2f} \pm {:0.2f} $ \n'.format(np.mean(specificity_list) * 100, np.std(specificity_list) * 100))
        f.write('Significance ratio: $ {:0.2f} \pm {:0.2f} $ \n'.format(np.mean(significance_ratio_list), np.std(significance_ratio_list)))
        f.write('hz_para_list: ' + str(hz_para_list) + '\n')
        f.write('\n\n\n')
    np.savetxt("vae_auc_and_std.csv", np.concatenate((auc_roc_list, [np.std(auc_roc_list)])), delimiter=",")
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

    hc_patient_comb_list = [[2, 0], [2, 1], [1, 0]]

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

    for hc_patient_comb in hc_patient_comb_list:
        mean_auc_roc, std_auc_roc, mean_accuracy, std_accuracy, mean_recall, std_recall, mean_specificity, std_specificity, mean_significance_ratio, std_significance_ratio = main(args.hz_para_list, hc_patient_comb[0], hc_patient_comb[1], args.combine, args.dataset_resourse, args.procedure, args.epochs)

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
        f.write('Experiment settings: VAE. {}. Dataset {} Procedure {} Epochs {}\n'.format('HC_vs_AD', 'SE-'+('MoE' if args.combine == 'moe' else 'PoE'), args.procedure, args.epochs))
        f.write('ROC-AUC: $ {:0.2f} \pm {:0.2f} $ \n'.format(np.mean(mean_auc_roc_list) * 100, np.mean(std_auc_roc_list) * 100))
        f.write('Accuracy: $ {:0.2f} \pm {:0.2f} $ \n'.format(np.mean(mean_accuracy_list) * 100, np.mean(std_accuracy_list) * 100))
        f.write('Sensitivity: $ {:0.2f} \pm {:0.2f} $ \n'.format(np.mean(mean_recall_list) * 100, np.mean(std_recall_list) * 100))
        f.write('Specificity: $ {:0.2f} \pm {:0.2f} $ \n'.format(np.mean(mean_specificity_list) * 100, np.mean(std_specificity_list) * 100))
        f.write('Significance ratio: $ {:0.2f} \pm {:0.2f} $ \n'.format(np.mean(mean_significance_ratio_list), np.mean(std_significance_ratio_list)))
        f.write('hz_para_list: ' + str(args.hz_para_list) + '\n')
        f.write('\n\n\n')

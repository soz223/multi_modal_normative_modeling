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
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import os
from utils import COLUMNS_HCP, COLUMNS_NAME, load_dataset, cliff_delta, COLUMNS_NAME_SNP, COLUMNS_NAME_VBM
from scipy import interp
from numpy import linspace

PROJECT_ROOT = Path.cwd()
result_focal_dir = './result_focal'

def compute_brain_regions_deviations(diff_df, clinical_df, hc_label, disease_label):
    """ Calculate the Cliff's delta effect size between groups."""
    region_df = pd.DataFrame(columns=['regions', 'pvalue', 'effect_size'])

    diff_hc = diff_df.loc[clinical_df['DIA'] == disease_label]
    diff_patient = diff_df.loc[clinical_df['DIA'] == hc_label]

    for region in COLUMNS_NAME:
        _, pvalue = stats.mannwhitneyu(diff_hc[region], diff_patient[region])
        effect_size = cliff_delta(diff_hc[region].values, diff_patient[region].values)

        region_df = region_df.append({'regions': region, 'pvalue': pvalue, 'effect_size': effect_size},
                                     ignore_index=True)

    return region_df

def compute_classification_performance(reconstruction_error_df, clinical_df, hc_label, disease_label):
    """ Calculate the AUCs and accuracy of the normative model."""
    error_hc = reconstruction_error_df.loc[clinical_df['DIA'] == hc_label]['Reconstruction error']
    error_patient = reconstruction_error_df.loc[clinical_df['DIA'] == disease_label]['Reconstruction error']

    labels = list(np.zeros_like(error_hc)) + list(np.ones_like(error_patient))
    predictions = list(error_hc) + list(error_patient)

    fpr, tpr, thresholds = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)
    

    # =============================================================================
    # # only maximize sensitivity

    # # Find the threshold where TPR is above the tpr_threshold
    # # This will give us the threshold corresponding to the desired sensitivity
    # tpr_threshold = 0.9
    # sensitivity_indices = np.where(tpr >= tpr_threshold)
    # optimal_idx = sensitivity_indices[0][0] if sensitivity_indices[0].size else 0
    # optimal_threshold = thresholds[optimal_idx]

    # # Compute accuracy using the optimal threshold
    # predicted_labels = np.array(predictions > optimal_threshold, dtype=int)




# =============================================================================

    # Youden’s index optimization criterion (for both sensitivity and specificity)

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

    recall = TP / (TP + FN)
    specificity = TN / (TN + FP)

    # significance ratio
    significance_ratio = recall / (1 - specificity) 


    fixed_points = 100
    base_fpr = linspace(0, 1, fixed_points)
    interp_tpr = interp(base_fpr, fpr, tpr)
    interp_tpr[0] = 0.0  # Ensuring that the curve starts at 0

    print('shape of tpr', tpr.shape)
    print('shape of predictions', np.array(predictions).shape)
    return roc_auc, interp_tpr, accuracy, accuracy_in_hc, accuracy_in_ad, recall, specificity, significance_ratio

def main(dataset_name, comb_label, hz_para_list, hc_label, disease_label):
    """Perform the group analysis."""
    # ----------------------------------------------------------------------------
    n_bootstrap = 10

    model_name = 'supervised_cvae_age_gender'

    participants_path = PROJECT_ROOT / 'data' / 'y.csv'
    freesurfer_path = PROJECT_ROOT / 'data' / (dataset_name + '.csv')



    # ----------------------------------------------------------------------------
    bootstrap_dir = PROJECT_ROOT / 'outputs' / 'bootstrap_analysis'
    model_dir = bootstrap_dir / model_name
    # ids_path = PROJECT_ROOT / 'outputs' / (dataset_name + '_homogeneous_ids.csv')

    # # ----------------------------------------------------------------------------
    # clinical_df = load_dataset(participants_path, ids_path, freesurfer_path)
    # clinical_df = clinical_df.set_index('participant_id')
    ids_dir = bootstrap_dir / 'ids'

    tpr_list = []
    auc_roc_list = []
    effect_size_list = []
    accuracy_list = []
    accuracy_in_ad_list = []
    accuracy_in_hc_list = []
    recall_list = []
    specificity_list = []
    significance_ratio_list = []


    for i_bootstrap in tqdm(range(n_bootstrap)):
        bootstrap_model_dir = model_dir / '{:03d}'.format(i_bootstrap)

        output_dataset_dir = bootstrap_model_dir / dataset_name
        output_dataset_dir.mkdir(exist_ok=True)

        analysis_dir = output_dataset_dir / '{:02d}_vs_{:02d}'.format(hc_label, disease_label)
        analysis_dir.mkdir(exist_ok=True)

        test_ids_filename = 'cleaned_bootstrap_test_{:03d}.csv'.format(i_bootstrap)
        ids_path = ids_dir / test_ids_filename

        # ----------------------------------------------------------------------------
        clinical_df = load_dataset(participants_path, ids_path, freesurfer_path)
        clinical_df = clinical_df.set_index('participant_id')

        # ----------------------------------------------------------------------------
        # normalized_df = pd.read_csv(output_dataset_dir / 'normalized.csv', index_col='participant_id')
        # reconstruction_df = pd.read_csv(output_dataset_dir / 'reconstruction.csv', index_col='participant_id')
        output_dataset_dir = bootstrap_model_dir / 'multimodal'
        output_dataset_dir.mkdir(exist_ok=True)
        reconstruction_error_df = pd.read_csv(output_dataset_dir / 'reconstruction_error.csv',
                                              index_col='participant_id')

        # ----------------------------------------------------------------------------
        # Compute effect size of the brain regions for the bootstrap iteration
        # diff_df = np.abs(normalized_df - reconstruction_df)
        # region_df = compute_brain_regions_deviations(diff_df, clinical_df, hc_label, disease_label)
        # effect_size_list.append(region_df['effect_size'].values)
        # region_df.to_csv(analysis_dir / 'regions_analysis.csv', index=False)

        # ----------------------------------------------------------------------------
        # Compute ROC-AUC for the bootstrap iteration
        roc_auc, tpr, accuracy, accuracy_in_hc, accuracy_in_ad, recall, specificity, significance_ratio = compute_classification_performance(reconstruction_error_df, clinical_df, hc_label, disease_label)
        auc_roc_list.append(roc_auc)
        tpr_list.append(tpr)
        accuracy_list.append(accuracy)
        accuracy_in_hc_list.append(accuracy_in_hc)
        accuracy_in_ad_list.append(accuracy_in_ad)
        recall_list.append(recall)
        specificity_list.append(specificity)
        significance_ratio_list.append(significance_ratio)
        

    (bootstrap_dir / dataset_name).mkdir(exist_ok=True)
    comparison_dir = bootstrap_dir / dataset_name / ('{:02d}_vs_{:02d}'.format(hc_label, disease_label))
    comparison_dir.mkdir(exist_ok=True)

    
    print('auc_roc_list\n', auc_roc_list)
    

    print('auc_roc: mean and std\n', np.mean(auc_roc_list), np.std(auc_roc_list))
    print('accuracy_list\n', accuracy_list)
    print('accuracy: mean and std\n', np.mean(accuracy_list), np.std(accuracy_list))
    print('accuracy_in_hc_list\n', accuracy_in_hc_list)
    print('accuracy_in_hc: mean and std\n', np.mean(accuracy_in_hc_list), np.std(accuracy_in_hc_list))
    print('accuracy_in_ad_list\n', accuracy_in_ad_list)
    print('accuracy_in_ad: mean and std\n', np.mean(accuracy_in_ad_list), np.std(accuracy_in_ad_list))
    print('recall: mean and std\n', np.mean(recall_list), np.std(recall_list))
    print('specificity: mean and std\n', np.mean(specificity_list), np.std(specificity_list))
    print('\n\n\n')

    if hc_label == 2 and disease_label == 0:
        compare_name = 'HC_vs_AD'
    elif hc_label == 2 and disease_label == 1:
        compare_name = 'HC_vs_MCI'
    elif hc_label == 1 and disease_label == 0:
        compare_name = 'MCI_vs_AD'


    # save auc roc and accuracy to txt file in result_focal folder
    with open(os.path.join(result_focal_dir, 'result_multimodal.txt'), 'a') as f:
        # f.write('Experiment settings: FAAE. {}. Dataset {}\n'.format(compare_name, 'Multimodal'))
        # f.write('auc_roc: mean and std\n' + str(np.mean(auc_roc_list)) + ' ' + str(np.std(auc_roc_list)) + '\n')
        # f.write('accuracy: mean and std\n' + str(np.mean(accuracy_list)) + ' ' + str(np.std(accuracy_list)) + '\n')
        # f.write('sensitivity: mean and std\n' + str(np.mean(recall_list)) + ' ' + str(np.std(recall_list)) + '\n')
        # f.write('specificity: mean and std\n' + str(np.mean(specificity_list)) + ' ' + str(np.std(specificity_list)) + '\n')
        # f.write('significance_ratio: mean and std\n' + str(np.mean(significance_ratio_list)) + ' ' + str(np.std(significance_ratio_list)) + '\n')
        # f.write('hz_para_list: ' + str(hz_para_list) + '\n')
        # f.write('\n\n\n')
        f.write('Experiment settings: FAAE. {}. Dataset {}\n'.format(compare_name, 'Multimodal'))
        f.write('ROC-AUC: $ {:0.2f} \pm {:0.2f} $ \n'.format(np.mean(auc_roc_list) * 100, np.std(auc_roc_list) * 100))
        f.write('Accuracy: $ {:0.2f} \pm {:0.2f} $ \n'.format(np.mean(accuracy_list) * 100, np.std(accuracy_list) * 100))
        f.write('Sensitivity: $ {:0.2f} \pm {:0.2f} $ \n'.format(np.mean(recall_list) * 100, np.std(recall_list) * 100))
        f.write('Specificity: $ {:0.2f} \pm {:0.2f} $ \n'.format(np.mean(specificity_list) * 100, np.std(specificity_list) * 100))
        f.write('Significance ratio: $ {:0.2f} \pm {:0.2f} $ \n'.format(np.mean(significance_ratio_list), np.std(significance_ratio_list)))
        f.write('hz_para_list: ' + str(hz_para_list) + '\n')
        f.write('\n\n\n')

    return np.mean(auc_roc_list), np.std(auc_roc_list), np.mean(accuracy_list), np.std(accuracy_list), np.mean(accuracy_in_hc_list), np.std(accuracy_in_hc_list), np.mean(accuracy_in_ad_list), np.std(accuracy_in_ad_list), np.mean(recall_list), np.std(recall_list), np.mean(specificity_list), np.std(specificity_list), np.mean(significance_ratio_list), np.std(significance_ratio_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--dataset_name',
                        dest='dataset_name',
                        help='Dataset name to perform group analysis.')
    parser.add_argument('-L', '--comb_label',
                        dest='comb_label',
                        help='Combination label to perform group analysis.',
                        type=int)
    parser.add_argument('-H', '--hz_para_list',
                        dest='hz_para_list',
                        nargs='+',
                        help='List of paras to perform the analysis.',
                        type=int)
    args = parser.parse_args()
    if args.dataset_name == 'snp':
        COLUMNS_NAME = COLUMNS_NAME_SNP
    elif args.dataset_name == 'vbm':
        COLUMNS_NAME = COLUMNS_NAME_VBM

    # main(args.dataset_name, args.comb_label, args.hz_para_list)
    mean_auc_roc_list = []
    std_auc_roc_list = []
    mean_accuracy_list = []
    std_accuracy_list = []
    mean_accuracy_in_hc_list = []
    std_accuracy_in_hc_list = []
    mean_accuracy_in_ad_list = []
    std_accuracy_in_ad_list = []
    mean_recall_list = []
    std_recall_list = []
    mean_specificity_list = []
    std_specificity_list = []
    mean_significance_ratio_list = []
    std_significance_ratio_list = []

    hc_patient_comb_list = [[2, 0], [2, 1], [1, 0]]
    
    for hc_patient_comb in hc_patient_comb_list:

        mean_auc_roc, std_auc_roc, \
        mean_accuracy, std_accuracy, \
        mean_accuracy_in_hc, std_accuracy_in_hc, \
        mean_accuracy_in_ad, std_accuracy_in_ad, \
        mean_recall, std_recall, \
        mean_specificity, std_specificity, \
        mean_significance_ratio, std_significance_ratio = main(args.dataset_name, args.comb_label, args.hz_para_list, hc_patient_comb[0], hc_patient_comb[1])


        mean_auc_roc_list.append(mean_auc_roc)
        std_auc_roc_list.append(std_auc_roc)
        mean_accuracy_list.append(mean_accuracy)
        std_accuracy_list.append(std_accuracy)
        mean_accuracy_in_hc_list.append(mean_accuracy_in_hc)
        std_accuracy_in_hc_list.append(std_accuracy_in_hc)
        mean_accuracy_in_ad_list.append(mean_accuracy_in_ad)
        std_accuracy_in_ad_list.append(std_accuracy_in_ad)
        mean_recall_list.append(mean_recall)
        std_recall_list.append(std_recall)
        mean_specificity_list.append(mean_specificity)
        std_specificity_list.append(std_specificity)
        mean_significance_ratio_list.append(mean_significance_ratio)
        std_significance_ratio_list.append(std_significance_ratio)

    # with open(os.path.join(result_focal_dir, 'result_multimodal.txt'), 'a') as f:
    #     f.write('Experiment settings: FAAE. Average Report. Dataset Multimodal\n')
    #     f.write('auc_roc: mean and std\n' + str(np.mean(mean_auc_roc_list)) + ' ' + str(np.mean(std_auc_roc_list)) + '\n')
    #     f.write('accuracy: mean and std\n' + str(np.mean(mean_accuracy_list)) + ' ' + str(np.mean(std_accuracy_list)) + '\n')
    #     f.write('senstivity: mean and std\n' + str(np.mean(mean_recall_list)) + ' ' + str(np.mean(std_recall_list)) + '\n')
    #     f.write('specificity: mean and std\n' + str(np.mean(mean_specificity_list)) + ' ' + str(np.mean(std_specificity_list)) + '\n')
    #     f.write('significance_ratio: mean and std\n' + str(np.mean(mean_significance_ratio_list)) + ' ' + str(np.mean(std_significance_ratio_list)) + '\n')
    #     f.write('hz_para_list: ' + str(args.hz_para_list) + '\n')
    #     f.write('\n\n\n')



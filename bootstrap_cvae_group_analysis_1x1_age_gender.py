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
from utils import COLUMNS_NAME, load_dataset, cliff_delta
from scipy import interp
from numpy import linspace

PROJECT_ROOT = Path.cwd()
result_focal_dir = './result_focal'

def compute_brain_regions_deviations(diff_df, clinical_df, disease_label, hc_label=1):
    """ Calculate the Cliff's delta effect size between groups."""
    region_df = pd.DataFrame(columns=['regions', 'pvalue', 'effect_size'])

    diff_hc = diff_df.loc[clinical_df['Diagnosis'] == disease_label]
    diff_patient = diff_df.loc[clinical_df['Diagnosis'] == hc_label]

    for region in COLUMNS_NAME:
        _, pvalue = stats.mannwhitneyu(diff_hc[region], diff_patient[region])
        effect_size = cliff_delta(diff_hc[region].values, diff_patient[region].values)

        region_df = region_df.append({'regions': region, 'pvalue': pvalue, 'effect_size': effect_size},
                                     ignore_index=True)

    return region_df

def compute_classification_performance(reconstruction_error_df, clinical_df, disease_label, hc_label=1):
    """ Calculate the AUCs and accuracy of the normative model."""
    error_hc = reconstruction_error_df.loc[clinical_df['Diagnosis'] == hc_label]['Reconstruction error']
    error_patient = reconstruction_error_df.loc[clinical_df['Diagnosis'] == disease_label]['Reconstruction error']

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


    fixed_points = 100
    base_fpr = linspace(0, 1, fixed_points)
    interp_tpr = interp(base_fpr, fpr, tpr)
    interp_tpr[0] = 0.0  # Ensuring that the curve starts at 0

    print('shape of tpr', tpr.shape)
    print('shape of predictions', np.array(predictions).shape)
    return roc_auc, interp_tpr, accuracy, accuracy_in_hc, accuracy_in_ad, recall, specificity

def main(dataset_name, comb_label, hz_para_list):
    """Perform the group analysis."""
    # ----------------------------------------------------------------------------
    n_bootstrap = 10

    dataset_name = 'ADNI'
    disease_label = 0

    model_name = 'supervised_cvae_age_gender'

    participants_path = PROJECT_ROOT / 'data' / dataset_name / 'participants.tsv'
    freesurfer_path = PROJECT_ROOT / 'data' / dataset_name / 'freesurferData.csv'

    hc_label = 1

    # ----------------------------------------------------------------------------
    bootstrap_dir = PROJECT_ROOT / 'outputs' / 'bootstrap_analysis'
    model_dir = bootstrap_dir / model_name
    ids_path = PROJECT_ROOT / 'outputs' / (dataset_name + '_homogeneous_ids.csv')

    # ----------------------------------------------------------------------------
    clinical_df = load_dataset(participants_path, ids_path, freesurfer_path)
    clinical_df = clinical_df.set_index('participant_id')

    tpr_list = []
    auc_roc_list = []
    effect_size_list = []
    accuracy_list = []
    accuracy_in_ad_list = []
    accuracy_in_hc_list = []
    recall_list = []
    specificity_list = []

    for i_bootstrap in tqdm(range(n_bootstrap)):
        bootstrap_model_dir = model_dir / '{:03d}'.format(i_bootstrap)

        output_dataset_dir = bootstrap_model_dir / dataset_name
        output_dataset_dir.mkdir(exist_ok=True)

        analysis_dir = output_dataset_dir / '{:02d}_vs_{:02d}'.format(hc_label, disease_label)
        analysis_dir.mkdir(exist_ok=True)

        # ----------------------------------------------------------------------------
        normalized_df = pd.read_csv(output_dataset_dir / 'normalized.csv', index_col='participant_id')
        reconstruction_df = pd.read_csv(output_dataset_dir / 'reconstruction.csv', index_col='participant_id')
        reconstruction_error_df = pd.read_csv(output_dataset_dir / 'reconstruction_error.csv',
                                              index_col='participant_id')

        # ----------------------------------------------------------------------------
        # Compute effect size of the brain regions for the bootstrap iteration
        diff_df = np.abs(normalized_df - reconstruction_df)
        region_df = compute_brain_regions_deviations(diff_df, clinical_df, disease_label)
        effect_size_list.append(region_df['effect_size'].values)
        region_df.to_csv(analysis_dir / 'regions_analysis.csv', index=False)

        # ----------------------------------------------------------------------------
        # Compute AUC-ROC for the bootstrap iteration
        roc_auc, tpr, accuracy, accuracy_in_hc, accuracy_in_ad, recall, specificity = compute_classification_performance(reconstruction_error_df, clinical_df, disease_label)
        auc_roc_list.append(roc_auc)
        tpr_list.append(tpr)
        accuracy_list.append(accuracy)
        accuracy_in_hc_list.append(accuracy_in_hc)
        accuracy_in_ad_list.append(accuracy_in_ad)
        recall_list.append(recall)
        specificity_list.append(specificity)
        

    (bootstrap_dir / dataset_name).mkdir(exist_ok=True)
    comparison_dir = bootstrap_dir / dataset_name / ('{:02d}_vs_{:02d}'.format(hc_label, disease_label))
    comparison_dir.mkdir(exist_ok=True)

    # save auc roc and accuracy to txt file in result_focal folder
    print('auc_roc_list\n', auc_roc_list)
    # to save auc_roc_list to csv file

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

    # save auc roc and accuracy to txt file in result_focal folder
    with open(os.path.join(result_focal_dir, 'result_focal_main_2.txt'), 'a') as f:
        f.write('auc_roc: mean and std\n' + str(np.mean(auc_roc_list)) + ' ' + str(np.std(auc_roc_list)) + '\n')
        f.write('accuracy: mean and std\n' + str(np.mean(accuracy_list)) + ' ' + str(np.std(accuracy_list)) + '\n')
        # f.write('accuracy_in_hc: mean and std\n' + str(np.mean(accuracy_in_hc_list)) + ' ' + str(np.std(accuracy_in_hc_list)) + '\n')
        # f.write('accuracy_in_ad: mean and std\n' + str(np.mean(accuracy_in_ad_list)) + ' ' + str(np.std(accuracy_in_ad_list)) + '\n')
        f.write('sensitivity: mean and std\n' + str(np.mean(recall_list)) + ' ' + str(np.std(recall_list)) + '\n')
        f.write('specificity: mean and std\n' + str(np.mean(specificity_list)) + ' ' + str(np.std(specificity_list)) + '\n')
        f.write('\n\n\n')

    # ----------------------------------------------------------------------------
    # Save regions effect sizes
    effect_size_df = pd.DataFrame(columns=COLUMNS_NAME, data=np.array(effect_size_list))
    effect_size_df.to_csv(comparison_dir / 'effect_size.csv')

    # Save AUC bootstrap values
# =============================================================================
#     if os.path.exists(comparison_dir/'auc_rocs_mean_std_{:02d}.csv'.format(comb_label)):
#         auc_rocs_mean_std_list = []
#         auc_rocs_mean_std_list.append([comb_label,hz_para_list[0], hz_para_list[1], hz_para_list[2],np.mean(auc_roc_list),np.std(auc_roc_list)])
#         auc_rocs_COL_NAME = ['Comb_Type', 'H_DIM1','H_DIM2','Z_DIM','Mean','Std']
#         auc_rocs_mean_std_df = pd.DataFrame(columns=auc_rocs_COL_NAME, data=np.array(auc_rocs_mean_std_list))
#         auc_rocs_mean_std_df.to_csv(comparison_dir/'cvae_auc_rocs_mean_std_{:02d}.csv'.format(comb_label),mode='a',header=False)   
#             
#     else:
#         auc_rocs_mean_std_list = []
#         auc_rocs_mean_std_list.append([comb_label,hz_para_list[0], hz_para_list[1], hz_para_list[2],np.mean(auc_roc_list),np.std(auc_roc_list)])
#         auc_rocs_COL_NAME = ['Comb_Type', 'H_DIM1','H_DIM2','Z_DIM','Mean','Std']
#         auc_rocs_mean_std_df = pd.DataFrame(columns=auc_rocs_COL_NAME, data=np.array(auc_rocs_mean_std_list))
#         auc_rocs_mean_std_df.to_csv(comparison_dir/'cvae_auc_rocs_mean_std_{:02d}.csv'.format(comb_label))   
# =============================================================================
    
    auc_roc_list = np.array(auc_roc_list)
    np.savetxt("cvae_age_gender_auc_and_std.csv", np.concatenate((auc_roc_list, [np.std(auc_roc_list)])), delimiter=",")
    auc_roc_df = pd.DataFrame(columns=['AUC-ROC'], data=auc_roc_list)
    auc_roc_df.to_csv(comparison_dir / 'auc_rocs.csv', index=False)

    # ----------------------------------------------------------------------------
    # Create Figure 3 of the paper
    
    # randomly delete 3 values in tpr_list[0] and tpr_list[1]
    

    tpr_list = np.array(tpr_list)
    print('tpr_list\n', tpr_list)
    mean_tprs = np.mean(tpr_list, axis=0)


    tprs_upper = np.percentile(tpr_list, 97.5, axis=0)
    tprs_lower = np.percentile(tpr_list, 2.5, axis=0)
    #plt.figure(figsize=(16, 20))
    plt.plot(np.linspace(0, 1, 100),
             mean_tprs,
             'b', lw=2,
             label='ROC curve (AUC = {:0.3f} ; 95% CI [{:0.3f}, {:0.3f}])'.format(np.mean(auc_roc_list),
                                                                                  np.percentile(auc_roc_list, 2.5),
                                                                                  np.percentile(auc_roc_list, 97.5)))
    plt.fill_between(np.linspace(0, 1, 100),
                     tprs_lower, tprs_upper,
                     color='grey', alpha=0.2)

    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig(comparison_dir / 'AUC-ROC.pdf', format='pdf')
    plt.show()
    #plt.close()
    #plt.clf()

    # --------------------------------------------------------------------------------------------
    # Create figure for supplementary materials
    effect_size_df = effect_size_df.reindex(abs(effect_size_df.mean()).sort_values().index, axis=1)

    plt.figure(figsize=(16, 20))
    plt.hlines(range(100),
               abs(np.percentile(effect_size_df, 2.5, axis=0)),
               abs(np.percentile(effect_size_df, 97.5, axis=0)))

    plt.plot(abs(effect_size_df.mean().values), range(100), 's', color='k')
    plt.axvline(0, ls='--')
    plt.yticks(np.arange(100), effect_size_df.columns)
    plt.xlabel('Effect size')
    plt.ylabel('Brain regions')
    plt.tight_layout()
    plt.savefig(comparison_dir / 'Regions.pdf', format='pdf')
    plt.show()
    #plt.close()
    #plt.clf()

    # --------------------------------------------------------------------------------------------
    # Create Figure 4 of the paper
    effect_size_sig_df = effect_size_df.reindex(abs(effect_size_df.mean()).sort_values().index, axis=1)
    lower_bound = np.percentile(effect_size_sig_df, 2.5, axis=0)
    higher_bound = np.percentile(effect_size_sig_df, 97.5, axis=0)

    for i, column in enumerate(effect_size_sig_df.columns):
        if (lower_bound[i] < 0) & (higher_bound[i] > 0):
            effect_size_sig_df = effect_size_sig_df.drop(columns=column)

    n_regions = len(effect_size_sig_df.columns)

    plt.figure()
    plt.hlines(range(n_regions),
               abs(np.percentile(effect_size_sig_df, 2.5, axis=0)),
               abs(np.percentile(effect_size_sig_df, 97.5, axis=0)))

    plt.plot(abs(effect_size_sig_df.mean().values), range(n_regions), 's', color='k')
    plt.axvline(0, ls='--')
    plt.yticks(np.arange(n_regions), effect_size_sig_df.columns)
    plt.xlabel('Effect size')
    plt.ylabel('Brain regions')
    plt.tight_layout()
    plt.savefig(comparison_dir / 'Significant_regions.pdf', format='pdf')
    plt.show()
    #plt.close()
    #plt.clf()
    save = np.concatenate(([effect_size_sig_df.columns],
                           [abs(effect_size_sig_df.mean().values)],
                           [abs(np.percentile(effect_size_sig_df, 2.5, axis=0))],
                           [abs(np.percentile(effect_size_sig_df, 97.5, axis=0))]))
    np.savetxt("cvae_effect_size.csv", save, fmt='%s', delimiter=",")
    
    # print success information
    print('Group analysis successfully performed for {} vs {}.'.format(hc_label, disease_label))

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

    main(args.dataset_name, args.comb_label, args.hz_para_list)

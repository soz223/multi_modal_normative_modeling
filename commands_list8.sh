#!/bin/bash


# dataset_names = ['T1_volume', 'mean_T1_intensity', 'mean_FA', 'mean_MD', 'mean_L1', 'mean_L2', 'mean_L3', 'min_BOLD', '25_percentile_BOLD', '50_percentile_BOLD', '75_percentile_BOLD', 'max_BOLD']

D_VALUES=(T1_volume mean_T1_intensity mean_FA mean_MD mean_L1 mean_L2 mean_L3 min_BOLD 25_percentile_BOLD 50_percentile_BOLD 75_percentile_BOLD max_BOLD)

# D_VALUES=(75_percentile_BOLD max_BOLD)

# ./bootstrap_train_ae_supervised.py -R HCP -D 
# ./bootstrap_test_ae_supervised.py -R HCP -D 
# ./bootstrap_ae_group_analysis_1x1.py  -R HCP -D 

for d in ${D_VALUES[@]}; do
    
    echo $d
    ./bootstrap_train_ae_supervised.py -R HCP -D $d
    ./bootstrap_test_ae_supervised.py -R HCP -D $d
    ./bootstrap_ae_group_analysis_1x1.py  -R HCP -D $d
done
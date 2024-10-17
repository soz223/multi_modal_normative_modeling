#!/bin/bash


# # Define an array containing all datasets

# datasets=("av45" "fdg" "vbm" "3modalities")
# datasets=("av45" "fdg" "vbm" "3modalities")


P_VALUES=("SE-MoE" "SE-PoE" "SE-gPoE" "UCA-MoE" "UCA-PoE" "UCA-gPoE")

E_VALUES=(1300)




for p in "${P_VALUES[@]}"
do
    for e in "${E_VALUES[@]}"
    do
        echo "Processing dataset: $p" 
        python ./mmcvae.py -P "$p" -E "$e"
    done

done







for p in "${P_VALUES[@]}"
do
    for e in "${E_VALUES[@]}"
    do
        echo "Processing dataset: $p" 
        python ./mmvae.py -P "$p" -E "$e"
    done

done


# for dataset in "${datasets[@]}"
# do
#     echo "Processing dataset: $dataset"
#     # Execute the training script
#     ./bootstrap_train_cvae_supervised.py -D "$dataset"  

#     # Execute the testing script
#     ./bootstrap_test_cvae_supervised.py -D "$dataset" 

#     # Execute the analysis script
#     ./bootstrap_cvae_group_analysis_1x1.py -D "$dataset" 

# done  




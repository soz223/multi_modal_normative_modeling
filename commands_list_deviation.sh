#!/bin/bash



# our proposed model
E_VALUES=(800)
P_VALUES=("UCA-gPoE" "SM-av45" "SM-fdg" "SM-vbm")
MODEL_VALUES=("cVAE_multimodal")
K_VALUES=(10)
R_VALUES=("ADNI")
TrainingClass_VALUES=("dm")

for e in "${E_VALUES[@]}"; do
    for p in "${P_VALUES[@]}"; do
        for m in "${MODEL_VALUES[@]}"; do
            for k in "${K_VALUES[@]}"; do
                ./multimodal_kfold_train_cvae_supervised.py -P $p -E $e  -K $k -R $R_VALUES -TrainingClass $TrainingClass_VALUES
                ./multimodal_kfold_test_cvae_supervised.py -P $p -K $k -R $R_VALUES 
                # ./multimodal_kfold_cvae_group_analysis_1x1.py -P $p -E $e -K $k 
            done
        done
    done
done


E_VALUES=(800)
P_VALUES=("UCA-gPoE" "SM-fMRI" "SM-sMRI")
MODEL_VALUES=("cVAE_multimodal")
K_VALUES=(10)
R_VALUES=("ADHD")

for e in "${E_VALUES[@]}"; do
    for p in "${P_VALUES[@]}"; do
        for m in "${MODEL_VALUES[@]}"; do
            for k in "${K_VALUES[@]}"; do
                ./multimodal_kfold_train_cvae_supervised.py -P $p -E $e  -K $k -R $R_VALUES -TrainingClass $TrainingClass_VALUES
                ./multimodal_kfold_test_cvae_supervised.py -P $p -K $k -R $R_VALUES
                # ./multimodal_kfold_cvae_group_analysis_1x1.py -P $p -E $e -K $k 
            done
        done
    done
done



# E_VALUES=(800)
# P_VALUES=("UCA-gPoE" "SM-fMRI" "SM-T1w_sMRI" "SM-T2w_sMRI")
# # MODEL_VALUES=("cVAE_multimodal")
# K_VALUES=(10)
# R_VALUES=("HCPimage")

# for e in "${E_VALUES[@]}"; do
#     for p in "${P_VALUES[@]}"; do
#         for m in "${MODEL_VALUES[@]}"; do
#             for k in "${K_VALUES[@]}"; do
#                 ./multimodal_kfold_train_cvae_supervised.py -P $p -E $e  -K $k -R $R_VALUES
#                 ./multimodal_kfold_test_cvae_supervised.py -P $p -K $k -R $R_VALUES
#                 # ./multimodal_kfold_cvae_group_analysis_1x1.py -P $p -E $e -K $k 
#             done
#         done
#     done
# done

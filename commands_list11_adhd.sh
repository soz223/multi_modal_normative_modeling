#!/bin/bash



# our proposed model
# E_VALUES=(50 100 200 400 800 1000 2000)
# E_VALUES=(50 200 1000)
E_VALUES=(50 500 1000)
# P_VALUES=("SE-MoE" "SE-PoE" "SE-gPoE")
# P_VALUES=("SE-MoE")
# P_VALUES=("SM-sMRI" "SM-fMRI")
P_VALUES=("SM-sMRI" "SM-fMRI" "SE-MoE" "SE-PoE" "SE-gPoE")

Baselearningrate_values=(0.001 0.00001 0.000001)
Maxlearningrate_values=(0.1)
MODEL_VALUES=("cVAE_multimodal")

hz_para_list_values=("110 110 10" "110 110 20" "110 110 30" "110 110 40" "110 110 50" "110 110 60" "110 110 70" "110 110 80" "110 110 90" "110 110 100" "1024 512 256 32" "20 10" "10 5" "100 5" "110 5" "110 10")
# hz_para_list_values=("110 5" "110 10" "110 20" "110 30" "110 40" "110 50" "110 60" "110 70" "110 80" "110 90" "110 100" "1024 512 256 32" "20 10" "10 5" "100 5" "110 5" "110 10")
# hz_para_list_values=("110 110 10")
K_VALUES=(10)

for e in "${E_VALUES[@]}"; do
        for m in "${MODEL_VALUES[@]}"; do
            for Baselearningrate in "${Baselearningrate_values[@]}"; do
                for Maxlearningrate in "${Maxlearningrate_values[@]}"; do
                    for k in "${K_VALUES[@]}"; do
                        for hz_para_list in "${hz_para_list_values[@]}"; do
                            for p in "${P_VALUES[@]}"; do

                            ./multimodal_kfold_train_cvae_supervised.py -P $p -E $e -Model $m -K $k -Baselearningrate $Baselearningrate -Maxlearningrate $Maxlearningrate -R ADHD -H $hz_para_list
                            ./multimodal_kfold_test_cvae_supervised.py -P $p -K $k -R ADHD -H $hz_para_list
                            ./multimodal_kfold_cvae_group_analysis_1x1.py -P $p -E $e -Model $m -K $k -R ADHD -H $hz_para_list
                        done
                    done
                done
            done
        done
    done
done


# # E_VALUES=(50)
# P_VALUES=( "SE-MoE" "SE-PoE" "SE-gPoE")

# for e in "${E_VALUES[@]}"; do
#     for p in "${P_VALUES[@]}"; do
#         for m in "${MODEL_VALUES[@]}"; do
#             for Baselearningrate in "${Baselearningrate_values[@]}"; do
#                 for Maxlearningrate in "${Maxlearningrate_values[@]}"; do
#                     for k in "${K_VALUES[@]}"; do
#                         for hz_para_list in "${hz_para_list_values[@]}"; do
#                             ./multimodal_kfold_train_cvae_supervised.py -P $p -E $e -Model $m -K $k -Baselearningrate $Baselearningrate -Maxlearningrate $Maxlearningrate -R ADHD -H $hz_para_list
#                             ./multimodal_kfold_test_cvae_supervised.py -P $p -K $k -R ADHD -H $hz_para_list
#                             ./multimodal_kfold_cvae_group_analysis_1x1.py -P $p -E $e -Model $m -K $k -R ADHD -H $hz_para_list
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done



MODEL_VALUES=("mmJSD" "DMVAE" "WeightedDMVAE" "mvtCAE")
P_VALUES=("SE-PoE")

for e in "${E_VALUES[@]}"; do
    for p in "${P_VALUES[@]}"; do
        for m in "${MODEL_VALUES[@]}"; do
            for Baselearningrate in "${Baselearningrate_values[@]}"; do
                for Maxlearningrate in "${Maxlearningrate_values[@]}"; do
                    for k in "${K_VALUES[@]}"; do
                        for hz_para_list in "${hz_para_list_values[@]}"; do
                            ./multimodal_kfold_train_cvae_supervised.py -P $p -E $e -Model $m -K $k -Baselearningrate $Baselearningrate -Maxlearningrate $Maxlearningrate -R ADHD -H $hz_para_list
                            ./multimodal_kfold_test_cvae_supervised.py -P $p -K $k -R ADHD -H $hz_para_list
                            ./multimodal_kfold_cvae_group_analysis_1x1.py -P $p -E $e -Model $m -K $k -R ADHD -H $hz_para_list
                        done
                    done
                done
            done
        done
    done
done
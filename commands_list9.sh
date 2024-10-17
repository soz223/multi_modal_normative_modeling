#!/bin/bash



# our proposed model
E_VALUES=(800)
P_VALUES=("UCA-gPoE")
MODEL_VALUES=("cVAE_multimodal")

for e in "${E_VALUES[@]}"; do
    for p in "${P_VALUES[@]}"; do
        for m in "${MODEL_VALUES[@]}"; do
            ./multimodal_kfold_train_cvae_supervised.py -P $p -E $e -Model $m -K 10
            ./multimodal_kfold_test_cvae_supervised.py -P $p -K 10
            ./multimodal_kfold_cvae_group_analysis_1x1.py -P $p -E $e -Model $m -K 10
        done
    done
done




# mmVAE
E_VALUES=(800)



# MoE-normVAE, PoE-normVAE, gPoE-normVAE
E_VALUES=(50)
P_VALUES=("SE-MoE" "SE-PoE" "SE-gPoE")
MODEL_VALUES=("cVAE_multimodal")

for e in "${E_VALUES[@]}"; do
    for p in "${P_VALUES[@]}"; do
        for m in "${MODEL_VALUES[@]}"; do
            ./multimodal_kfold_train_cvae_supervised.py -P $p -E $e -Model $m -K 10
            ./multimodal_kfold_test_cvae_supervised.py -P $p -K 10
            ./multimodal_kfold_cvae_group_analysis_1x1.py -P $p -E $e -Model $m -K 10
        done
    done
done





# mmJSD, DMVAE, WeightedDMVAE, mvtCAE
E_VALUES=(50)

MODEL_VALUES=("mmJSD" "DMVAE" "WeightedDMVAE" "mvtCAE")
P_VALUES=("SE-PoE")
# MODEL_VALUES=("mmVAEPlus")

for e in "${E_VALUES[@]}"; do
    for p in "${P_VALUES[@]}"; do
        for m in "${MODEL_VALUES[@]}"; do
            ./multimodal_kfold_train_cvae_supervised.py -P $p -E $e -Model $m -K 10
            ./multimodal_kfold_test_cvae_supervised.py -P $p -K 10
            ./multimodal_kfold_cvae_group_analysis_1x1.py -P $p -E $e -Model $m -K 10
        done
    done
done
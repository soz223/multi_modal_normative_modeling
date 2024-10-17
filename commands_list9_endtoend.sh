#!/bin/bash


E_VALUES=(1000)
P_VALUES=("UCA-gPoE")
MODEL_VALUES=("cVAE_multimodal")





TestE_values=(50 60 70 80 90 100 200 300 400 500 600 700 800 900 1000)
TestLR_values=(0.000001 0.00001 0.0001 0.0005 0.001)
# TestE_values=(50)
# TestLR_values=(0.001)
MLP_values=("mlp")
./multimodal_kfold_cvae_freezeAE_endtoend.py train -R ADNI -P UCA-gPoE -E 1000
for e in "${E_VALUES[@]}"; do

    for p in "${P_VALUES[@]}"; do
        # ./multimodal_kfold_cvae_freezeAE_endtoend.py -P $p -E $e 
        for TestE in "${TestE_values[@]}"; do
            for TestLR in "${TestLR_values[@]}"; do
                for MLP in "${MLP_values[@]}"; do
                    ./multimodal_kfold_cvae_freezeAE_endtoend.py test -R ADNI -TestE $TestE -TestLR $TestLR -MLP $MLP -P $p -E $e
                    # ./multimodal_kfold_cvae_freezeAE_endtoend.py analyze -R ADNI -TestE $TestE -TestLR $TestLR -MLP $MLP -P $p -E $e
                done
            done
        done
    done

done


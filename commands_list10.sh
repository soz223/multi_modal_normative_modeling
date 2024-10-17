#!/bin/bash




# experiment on single modality on VAE
./bootstrap_create_ids.py -R ADNI
E_VALUES=(100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500)


D_VALUES=(3modalities)

# for d in ${D_VALUES[@]}; do
    
#     echo $d
#     ./bootstrap_train_vae_supervised.py -R ADNI -D $d
#     ./bootstrap_test_vae_supervised.py -R ADNI -D $d
#     ./bootstrap_vae_group_analysis_1x1.py  -R ADNI -D $d
# done

for d in ${D_VALUES[@]}; do
    for e in "${E_VALUES[@]}"
    do
        ./bootstrap_train_cvae_supervised.py -R ADNI -D $d -E $e
        ./bootstrap_test_cvae_supervised.py -R ADNI -D $d 
        ./bootstrap_cvae_group_analysis_1x1.py  -R ADNI -D $d -E $e
    done
done



for d in ${D_VALUES[@]}; do
    for e in "${E_VALUES[@]}"
    do
        ./bootstrap_train_vae_supervised.py -R ADNI -D $d -E $e
        ./bootstrap_test_vae_supervised.py -R ADNI -D $d 
        ./bootstrap_vae_group_analysis_1x1.py  -R ADNI -D $d -E $e
    done
done
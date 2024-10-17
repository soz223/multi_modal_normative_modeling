#!/bin/bash

# This script is used to run the multimodal commands in sequence.



# P_VALUES=SE-MoE SE-PoE UCA-MoE UCA-PoE

# E_VALUES is 10 20 30 40 50 60 70 80 90 100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 3000 4000 5000 6000 7000 8000

# E_VALUES=(2 3 5 8 10 20 30 40 50 60 70 80 90 100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 3000 4000 5000 6000 7000 8000)

# E_VALUES=(1100 1200 1300 1400)

# E_VALUES = 1100 1120 1140 1160 ... 1400
# E_VALUES=(1100 1120 1140 1160 1180 1200 1220 1240 1260 1280 1300 1320 1340 1360 1380 1400)

E_VALUES=(200)

P_VALUES=("SE-MoE" "SE-PoE" "UCA-MoE" "UCA-PoE")
# P_VALUES=("SE-MoE")

# for h in "${H_VALUES[@]}"
# do
#     for c in "${C_VALUES[@]}"
#     do
        
        
        

#             ./multimodal_bootstrap_train_cvae_supervised.py -H $h -C $c

#             ./multimodal_bootstrap_test_cvae_supervised.py 

#             ./multimodal_bootstrap_cvae_group_analysis_1x1.py -H $h -C $c

        
#     done
# done

for e in "${E_VALUES[@]}"
do 
    for p in "${P_VALUES[@]}"
        do
            ./multimodal_bootstrap_train_cvae_supervised.py -P $p -E $e

            ./multimodal_bootstrap_test_cvae_supervised.py -P $p

            ./multimodal_bootstrap_cvae_group_analysis_1x1.py -P $p -E $e
        done
done
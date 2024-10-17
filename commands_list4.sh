H_VALUES=("110 110 10")

# # for ae, aae, vae, cvae, only for av45, do the train test and analysis

# # ae code
# for h in "${H_VALUES[@]}"
# do
#     for dataset in "3modalities"
#     do
#         echo "Processing dataset: $dataset"

#         ./all_feature_bootstrap_train_ae_supervised.py -D "$dataset" -H $h -B 0.0001 -M 0.005

#         ./all_feature_bootstrap_test_ae_supervised.py -D "$dataset"

#         ./all_feature_bootstrap_ae_group_analysis_1x1.py -D "$dataset" -L 0 -H $h

#     done
# done



# H_VALUES=("110 110 10")

# for ae, aae, vae, cvae, only for av45, do the train test and analysis
H_VALUES=("110 110 10" "100 100 100 10" "90 90 10" "90 90 90 10" "90 90 90 90 90 10" "100 100 10" "300 300 30" "200 200 10" "64 32 16 10" "20 10" "10 10" "20 20" "50 25 12 10" "10 10 10" "20 20 20" "20 10 10" "10 10 5" "1024 512 10" "1000 10" "512 10" "256 10" "1024 512 256 10" "128 64 32" "256 128 64 32 16" "512 256 128 64 32 16" "1024 512 256 128 64 32 16" "2048 10" "10 10" "20 10" "50 10" "20 20" "110 110 110")

datasets=("av45" "fdg" "vbm" "3modalities")

# ae code
for h in "${H_VALUES[@]}"
do
    # for dataset in "av45 fdg vbm 3modalities"
    for dataset in "${datasets[@]}"
    do
        echo "Processing dataset: $dataset" "with h: $h"

        ./all_feature_bootstrap_train_cvae_supervised.py -D "$dataset" -H $h

        ./all_feature_bootstrap_test_cvae_supervised.py -D "$dataset"

        ./all_feature_bootstrap_cvae_group_analysis_1x1.py -D "$dataset" -H $h

    done
done
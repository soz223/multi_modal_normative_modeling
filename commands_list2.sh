


# # Define an array containing all datasets

datasets=("3modalities")
# A_VALUES="0.0001 0.0002 0.0003 0.0004 0.0005 0.0006 0.0007 0.0008 0.001 0.0025 0.003 0.005 0.0075 0.01 0.015 0.02 0.025 0.03 0.035 0.04 0.05 0.07 0.2 0.25 0.5 0.6 0.7 0.75 0.8 0.9 1"


# A_VALUES="0.01 0.02 0.05 0.1 0.2 0.5 1"
# G_VALUES="1 5 7.5 10 12.5 15 17.5 20"
# H_VALUES=("64 32 16 10" "20 10" "10 10" "20 20" "50 25 12 10" "10 10 10" "20 20 20" "20 10 10" "10 10 5")
# H_VALUES=("90 90 90 10" "100 100 10" "300 300 30" "200 200 10" "20 10" "10 10" "20 20" "10 10 10" "20 20 20" "20 10 10")
# H_VALUES=("100 100 100 10" "90 90 10" "90 90 90 10" "90 90 90 90 90 10" "100 100 10" "300 300 30" "200 200 10" "64 32 16 10" "20 10" "10 10" "20 20" "50 25 12 10" "10 10 10" "20 20 20" "20 10 10" "10 10 5" "1024 512 10" "1000 10" "512 10" "256 10" "1024 512 256 10" "128 64 32" "256 128 64 32 16" "512 256 128 64 32 16" "1024 512 256 128 64 32 16" "2048 10" "10 10" "20 10" "50 10" "20 20" "110 110 110")
# H_VALUES=("110 110 110" "10 10" "10 10 5" "20 20" "10 10 10" "20 10 10" "20 10")
H_VALUES=("110 110 10" "280 280 27")

A_VALUES="0 0.2"
G_VALUES="1 10 15 30"
# H_VALUES=("110 110 10")

# Loop through each dataset in the array
    # grid search the -A and -G parameters by looping through all possible combinations
for a in $A_VALUES
do
    for g in $G_VALUES
    do
        for h in "${H_VALUES[@]}"
        do
            for dataset in "${datasets[@]}"
            do
                echo "Processing dataset: $dataset"
                echo "Processing dataset: $dataset, A: $a, G: $g"
                # Execute the training script
                ./all_feature_bootstrap_train_cvae_supervised_age_gender.py -D "$dataset"  -A $a -G $g -R 0 -L 1 -H $h

                # Execute the testing script
                ./all_feature_bootstrap_test_cvae_supervised_age_gender.py -D "$dataset" -L 1

                # Execute the analysis script
                ./all_feature_bootstrap_cvae_group_analysis_1x1_age_gender.py -D "$dataset" -H $h

            done  
        done      
    done
done















































# # # Define an array containing all datasets

# datasets=("3modalities")
# # A_VALUES="0.0001 0.0002 0.0003 0.0004 0.0005 0.0006 0.0007 0.0008 0.001 0.0025 0.003 0.005 0.0075 0.01 0.015 0.02 0.025 0.03 0.035 0.04 0.05 0.07 0.2 0.25 0.5 0.6 0.7 0.75 0.8 0.9 1"


# A_VALUES="0.01 0.02 0.05 0.1 0.2 0.5 1"
# G_VALUES="1 5 7.5 10 12.5 15 17.5 20"
# # H_VALUES=("64 32 16 10" "20 10" "10 10" "20 20" "50 25 12 10" "10 10 10" "20 20 20" "20 10 10" "10 10 5")
# # H_VALUES=("90 90 90 10" "100 100 10" "300 300 30" "200 200 10" "20 10" "10 10" "20 20" "10 10 10" "20 20 20" "20 10 10")
# # H_VALUES=("100 100 100 10" "90 90 10" "90 90 90 10" "90 90 90 90 90 10" "100 100 10" "300 300 30" "200 200 10" "64 32 16 10" "20 10" "10 10" "20 20" "50 25 12 10" "10 10 10" "20 20 20" "20 10 10" "10 10 5" "1024 512 10" "1000 10" "512 10" "256 10" "1024 512 256 10" "128 64 32" "256 128 64 32 16" "512 256 128 64 32 16" "1024 512 256 128 64 32 16" "2048 10" "10 10" "20 10" "50 10" "20 20" "110 110 110")
# H_VALUES=("110 110 110" "10 10" "10 10 5" "20 20" "10 10 10" "20 10 10" "20 10")


# # A_VALUES="0"
# # G_VALUES="1"
# # H_VALUES=("110 110 10")

# # Loop through each dataset in the array
#     # grid search the -A and -G parameters by looping through all possible combinations
# for a in $A_VALUES
# do
#     for g in $G_VALUES
#     do
#         for h in "${H_VALUES[@]}"
#         do
#             for dataset in "${datasets[@]}"
#             do
#                 echo "Processing dataset: $dataset"
#                 echo "Processing dataset: $dataset, A: $a, G: $g"
#                 # Execute the training script
#                 ./all_feature_bootstrap_train_cvae_supervised_age_gender.py -D "$dataset"  -A $a -G $g -R 0 -L 1 -H $h

#                 # Execute the testing script
#                 ./all_feature_bootstrap_test_cvae_supervised_age_gender.py -D "$dataset" -L 1

#                 # Execute the analysis script
#                 ./all_feature_bootstrap_cvae_group_analysis_1x1_age_gender.py -D "$dataset" -H $h

#             done  
#         done      
#     done
# done





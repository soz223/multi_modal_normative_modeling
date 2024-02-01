./clean_data.py 

./bootstrap_create_ids.py


for dataset in "av45" "fdg" "vbm" "snp"
do
    echo "Processing dataset: $dataset"

    ./bootstrap_train_cvae_supervised.py -D "$dataset" 

    ./bootstrap_test_cvae_supervised.py -D "$dataset"

    ./bootstrap_cvae_group_analysis_1x1.py -D "$dataset" -L 0

done



# for dataset in "av45" "fdg" "vbm" "snp"
# do
#     echo "Processing dataset: $dataset"

#     ./bootstrap_train_ae_supervised.py -D "$dataset" 

#     ./bootstrap_test_ae_supervised.py -D "$dataset"

#     ./bootstrap_ae_group_analysis_1x1.py -D "$dataset" -L 0

# done



# # Define an array containing all datasets
# datasets=("av45" "fdg" "vbm" "snp")

# # Loop through each dataset in the array
# for dataset in "${datasets[@]}"
# do
#     echo "Processing dataset: $dataset"

#     # Execute the training script
#     ./bootstrap_train_cvae_supervised_age_gender.py -D "$dataset" -A 0 -G 1 -R 0

#     # Execute the testing script
#     ./bootstrap_test_cvae_supervised_age_gender.py -D "$dataset"

#     # Execute the analysis script
#     ./bootstrap_cvae_group_analysis_1x1_age_gender.py -D "$dataset"
# done


# # Loop through each dataset in the array
# for dataset in "${datasets[@]}"
# do
#     echo "Processing dataset: $dataset"

#     # Execute the training script
#     ./bootstrap_train_cvae_supervised_age_gender.py -D "$dataset"  -A 0.2 -G 15 -R 0

#     # Execute the testing script
#     ./bootstrap_test_cvae_supervised_age_gender.py -D "$dataset"

#     # Execute the analysis script
#     ./bootstrap_cvae_group_analysis_1x1_age_gender.py -D "$dataset"
# done

# # Train normative model
# ./bootstrap_train_ae_supervised.py

# # Calculate deviations on clinical data
# ./bootstrap_test_ae_supervised.py 

# # Perform statistical analysis
# ./bootstrap_ae_group_analysis_1x1.py -L 0

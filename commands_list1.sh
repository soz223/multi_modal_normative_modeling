# ./clean_data.py 

# ./bootstrap_create_ids.py

H_VALUES=("110 110 10")

# for ae, aae, vae, cvae, only for av45, do the train test and analysis

# ae code
for h in "${H_VALUES[@]}"
do
    for dataset in "av45" "fdg" "vbm" "snp"
    do
        echo "Processing dataset: $dataset"

        ./bootstrap_train_ae_supervised.py -D "$dataset" -H $h -B 0.0001 -M 0.005

        ./bootstrap_test_ae_supervised.py -D "$dataset"

        ./bootstrap_ae_group_analysis_1x1.py -D "$dataset" -L 0 -H $h

    done
done

# # aae code
# for dataset in "av45"
# do
#     echo "Processing dataset: $dataset"

#     ./bootstrap_train_aae_supervised_new.py -D "$dataset" 

#     ./bootstrap_test_aae_supervised_new.py -D "$dataset"

#     ./bootstrap_aae_new_group_analysis_1x1.py -D "$dataset" -L 0
# done

# # vae code
# for dataset in "av45"
# do
#     echo "Processing dataset: $dataset"

#     ./bootstrap_train_vae_supervised.py -D "$dataset" 

#     ./bootstrap_test_vae_supervised.py -D "$dataset"

#     ./bootstrap_vae_group_analysis_1x1.py -D "$dataset" -L 0

# done


# # cvae code
# for dataset in "av45"
# do
#     echo "Processing dataset: $dataset"

#     ./bootstrap_train_cvae_supervised.py -D "$dataset" 

#     ./bootstrap_test_cvae_supervised.py -D "$dataset"

#     ./bootstrap_cvae_group_analysis_1x1.py -D "$dataset" -L 0

# done














































# H_VALUES=("110 110 10")
# # H_VALUES=("90 90 90 10" "100 100 10" "300 300 30" "200 200 10" "20 10" "10 10" "20 20" "10 10 10" "20 20 20" "20 10 10")
# for h in "${H_VALUES[@]}"
# do
#     for dataset in "av45" "fdg" "vbm" "snp"
#     do
#         echo "Processing dataset: $dataset"

#         ./bootstrap_train_ae_supervised.py -D "$dataset" -H $h -B 0.0001 -M 0.005

#         ./bootstrap_test_ae_supervised.py -D "$dataset"

#         ./bootstrap_ae_group_analysis_1x1.py -D "$dataset" -L 0 -H $h

#     done
# done

# # aae code
# for dataset in "av45" "fdg" "vbm" "snp"
# do
#     echo "Processing dataset: $dataset"

#     ./bootstrap_train_aae_supervised_new.py -D "$dataset" 

#     ./bootstrap_test_aae_supervised_new.py -D "$dataset"

#     ./bootstrap_aae_new_group_analysis_1x1.py -D "$dataset" -L 0
# done

# # vae code
# for dataset in "av45" "fdg" "vbm" "snp"
# do
#     echo "Processing dataset: $dataset"

#     ./bootstrap_train_vae_supervised.py -D "$dataset" 

#     ./bootstrap_test_vae_supervised.py -D "$dataset"

#     ./bootstrap_vae_group_analysis_1x1.py -D "$dataset" -L 0

# done


# # cvae code
# for dataset in "av45" "fdg" "vbm" "snp"
# do
#     echo "Processing dataset: $dataset"

#     ./bootstrap_train_cvae_supervised.py -D "$dataset" 

#     ./bootstrap_test_cvae_supervised.py -D "$dataset"

#     ./bootstrap_cvae_group_analysis_1x1.py -D "$dataset" -L 0

# done
















































# # Define an array containing all datasets

# datasets=("av45" "fdg" "vbm" "snp")


# # A_VALUES="0.0001 0.0002 0.0003 0.0004 0.0005 0.0006 0.0007 0.0008 0.001 0.0025 0.003 0.005 0.0075 0.01 0.015 0.02 0.025 0.03 0.035 0.04 0.05 0.07 0.2 0.25 0.5 0.6 0.7 0.75 0.8 0.9 1"


# A_VALUES="0.01 0.1 0.25 0.5 0.7 1"
# G_VALUES="1 5 10 20 30"
# # H_VALUES=("64 32 16 10" "20 10" "10 10" "20 20" "50 25 12 10" "10 10 10" "20 20 20" "20 10 10" "10 10 5")
# H_VALUES=("90 90 90 10" "100 100 10" "300 300 30" "200 200 10" "20 10" "10 10" "20 20" "10 10 10" "20 20 20" "20 10 10")

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
#                 ./bootstrap_train_cvae_supervised_age_gender.py -D "$dataset"  -A $a -G $g -R 0 -L 1 -H $h

#                 # Execute the testing script
#                 ./bootstrap_test_cvae_supervised_age_gender.py -D "$dataset" -L 1

#                 # Execute the analysis script
#                 ./bootstrap_cvae_group_analysis_1x1_age_gender.py -D "$dataset" -H $h

#             done  
#         done      
#     done
# done






# # Loop through each dataset in the array
# for dataset in "${datasets[@]}"
# do
#     echo "Processing dataset: $dataset"

#     # Define an array containing different combinations of H values
#     H_VALUES=("64 32 16 10" "70 35 20 10" "80 40 20 10" "50 25 12 10" "90 45 22 10" "60 30 15 10""90 90 10" "514 114 10" "1000 100 10" "90 80 70 60 50 40 30 20 10" "20 10" "50 10" "90 90 90 5" "20 20 20 20" "20 20 20" "20 10" "10 10" "20 20" "45 25 15")

#     # Loop through each combination of H values
#     for h in "${H_VALUES[@]}"
#     do
#         echo "Processing dataset: $dataset, H: $h"

#         # Execute the training script
#         ./bootstrap_train_cvae_supervised_age_gender.py -D "$dataset" -A 0 -G 1 -R 0 -L 1 -H $h

#         # Execute the testing script
#         ./bootstrap_test_cvae_supervised_age_gender.py -D "$dataset" -L 1

#         # Execute the analysis script
#         ./bootstrap_cvae_group_analysis_1x1_age_gender.py -D "$dataset" -H $h
#     done
# done


# # Loop through each dataset in the array
# for dataset in "${datasets[@]}"
# do

    
#     echo "Processing dataset: $dataset"

#     # Execute the training script
#     ./bootstrap_train_cvae_supervised_age_gender.py -D "$dataset" -A 0 -G 1 -R 0 -L 1 -H 64 32 16 10

#     # Execute the testing script
#     ./bootstrap_test_cvae_supervised_age_gender.py -D "$dataset" -L 1

#     # Execute the analysis script
#     ./bootstrap_cvae_group_analysis_1x1_age_gender.py -D "$dataset" -H 64 32 16 10
# done





# # Loop through each dataset in the array
# for dataset in "${datasets[@]}"
# do
#     echo "Processing dataset: $dataset"

#     # Execute the training script
#     ./bootstrap_train_cvae_supervised_age_gender.py -D "$dataset" -A 0 -G 1 -R 0 -L 1

#     # Execute the testing script
#     ./bootstrap_test_cvae_supervised_age_gender.py -D "$dataset" -L 1

#     # Execute the analysis script
#     ./bootstrap_cvae_group_analysis_1x1_age_gender.py -D "$dataset"
# done


# # Loop through each dataset in the array
# for dataset in "${datasets[@]}"
# do
#     echo "Processing dataset: $dataset"

#     # Execute the training script
#     ./bootstrap_train_cvae_supervised_age_gender.py -D "$dataset"  -A 0.2 -G 15 -R 0 -L 1

#     # Execute the testing script
#     ./bootstrap_test_cvae_supervised_age_gender.py -D "$dataset" -L 1

#     # Execute the analysis script
#     ./bootstrap_cvae_group_analysis_1x1_age_gender.py -D "$dataset"
# done



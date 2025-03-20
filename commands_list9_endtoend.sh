#!/bin/bash

# Define hyperparameters and their possible values

# Margin for contrastive loss
margin_values=(0.5 1 2)

# Weight for contrastive loss
weightcontrastive_values=(0.1 1 5)

# Weight for KL divergence loss
weight_kl_values=(0.1 1 5)

# Weight for reconstruction loss
weight_rec_values=(0.1 1 5)

# Dropout rates
dropout_values=(0 0.2 0.5)

# MLP classifier hidden layers
layers_values=("128 64" "256 128 64" "128 64 32 16" "256 128 64 32 16" "110 110 10" "110 10")

# Hidden layer dimensions for encoder/decoder
hz_para_list_values=("110 110 10" "100 100 100 10" "90 90 10" "90 90 90 10" "90 90 90 90 90 10" "100 100 10" "300 300 30" "200 200 10" "64 32 16 10" "20 10" "10 10" "20 20" "50 25 12 10" "10 10 10" "20 20 20" "20 10 10" "10 10 5" "1024 512 10" "1000 10" "512 10" "256 10" "1024 512 256 10" "128 64 32" "256 128 64 32 16" "512 256 128 64 32 16" "1024 512 256 128 64 32 16" "2048 10" "10 10" "20 10" "50 10" "20 20" "110 110 110")

# Epochs
epochs_values=(10 20 30 40 50 100)

# Other fixed parameters
dataset_resourse="ADNI"

# Create results directory (not strictly necessary if not saving logs)
results_dir="./grid_search_results"
mkdir -p "${results_dir}"

# Total number of combinations
total_combinations=0

# Loop over all hyperparameter combinations
for epochs in "${epochs_values[@]}"; do
  for margin in "${margin_values[@]}"; do
    for weightcontrastive in "${weightcontrastive_values[@]}"; do
      for weight_kl in "${weight_kl_values[@]}"; do
        for weight_rec in "${weight_rec_values[@]}"; do
          for dropout in "${dropout_values[@]}"; do
            for layers in "${layers_values[@]}"; do
              for hz_para_list in "${hz_para_list_values[@]}"; do
                total_combinations=$((total_combinations + 1))

                # Print the parameters
                echo "Running experiment with parameters:"
                echo "Epochs: ${epochs}"
                echo "Margin: ${margin}"
                echo "Weight Contrastive: ${weightcontrastive}"
                echo "Weight KL: ${weight_kl}"
                echo "Weight Rec: ${weight_rec}"
                echo "Dropout: ${dropout}"
                echo "Layers: ${layers}"
                echo "Hz Para List: ${hz_para_list}"
                echo "---------------------------------------"

                # Run main program without logging to individual files
                python multimodal_kfold_train_cvae_supervised_endtoend2.py \
                  -R "${dataset_resourse}" \
                  -E "${epochs}" \
                  -Margin "${margin}" \
                  -Weightcontrastive "${weightcontrastive}" \
                  -Weightkl "${weight_kl}" \
                  -Weightrec "${weight_rec}" \
                  -Dropout "${dropout}" \
                  -Layers ${layers} \
                  -H ${hz_para_list}

                # Note: Ensure that your main program appends results to a single CSV file,
                # for example, './results_endtoend.csv'.

              done
            done
          done
        done
      done
    done
  done
done

echo "Total combinations run: $total_combinations"

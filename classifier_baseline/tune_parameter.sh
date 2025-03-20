#!/bin/bash

# Define the ranges for each hyperparameter
NUM_EPOCHS_LIST=(2000 3000 4000 5000 10000)
INITIAL_LR_LIST=(1e-4 5e-4 1e-3 5e-3 1e-2)
MIN_LR_LIST=(1e-9 1e-8 1e-7 1e-6)
HIDDEN_LAYERS_LIST=(
    # "128 64 32 16"
    # "128 64 32"
    "512 256 128"
    "512 256 128 64 32 16"
    "256 128 64"
    "64 32 16"
    "512 256"
)

DROPOUT_LIST=(0.0 0.1 0.2 0.3 0.4 0.5)


# Loop through each combination of hyperparameters
for num_epochs in "${NUM_EPOCHS_LIST[@]}"; do
  for initial_lr in "${INITIAL_LR_LIST[@]}"; do
    for min_lr in "${MIN_LR_LIST[@]}"; do
      for hidden_layers in "${HIDDEN_LAYERS_LIST[@]}"; do
        for dropout in "${DROPOUT_LIST[@]}"; do
          
          # Print the current combination of hyperparameters
          echo "Training with num_epochs=${num_epochs}, initial_lr=${initial_lr}, min_lr=${min_lr}, hidden_layers=${hidden_layers}, dropout=${dropout}"

          # Run the training script with the current combination of hyperparameters
          python classifier.py \
            --fmri_path /home/songlinzhao/multi_modal_normative_modeling/data/HCPimage/fMRI.csv \
            --labels_path /home/songlinzhao/multi_modal_normative_modeling/data/HCPimage/y.csv \
            --num_epochs ${num_epochs} \
            --initial_lr ${initial_lr} \
            --min_lr ${min_lr} \
            --hidden_layers ${hidden_layers} \
            --dropout ${dropout} \
            --checkpoint_path "checkpoints/model_${num_epochs}_${initial_lr}_${min_lr}_${dropout}.pth" \
            --log_level INFO

          echo "Finished training with num_epochs=${num_epochs}, initial_lr=${initial_lr}, min_lr=${min_lr}, hidden_layers=${hidden_layers}, dropout=${dropout}"
          
        done
      done
    done
  done
done

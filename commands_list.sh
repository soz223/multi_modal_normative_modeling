conda init bash

conda activate rop

./clean_data.py 

./bootstrap_create_ids.py

# Train normative model
./bootstrap_train_ae_supervised.py

# Calculate deviations on clinical data
./bootstrap_test_ae_supervised.py -D "ADNI"

# Perform statistical analysis
./bootstrap_ae_group_analysis_1x1.py -D "ADNI" -L 0

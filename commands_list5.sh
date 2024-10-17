H_VALUES=("100 100 100 10" "90 90 10" "90 90 90 10" "90 90 90 90 90 10" "100 100 10" "300 300 30" "200 200 10" "64 32 16 10" "20 10" "10 10" "20 20" "50 25 12 10" "10 10 10" "20 20 20" "20 10 10" "10 10 5" "1024 512 10" "1000 10" "512 10" "256 10" "1024 512 256 10" "128 64 32" "256 128 64 32 16" "512 256 128 64 32 16" "1024 512 256 128 64 32 16" "2048 10" "10 10" "20 10" "50 10" "20 20" "110 110 110")
A_VALUES="0.01 0.02 0.05 0.1 0.2 0.5 1"
G_VALUES="1 5 7.5 10 12.5 15 17.5 20"


# ./multimodal_bootstrap_train_cvae_supervised_age_gender_effect_each.py -H 110 110 10 -D av45 -A 0.2 -G 15 -R 0
# ./multimodal_bootstrap_test_cvae_supervised_age_gender_effect_each.py -D av45
# ./multimodal_bootstrap_cvae_group_analysis_1x1_age_gender_effect_each.py -D av45 -H 110 110 10


for h in "${H_VALUES[@]}"
do
    for a in $A_VALUES
    do
        for g in $G_VALUES
        do
            for dataset in "av45"
            do
                echo "Processing dataset: $dataset"
                echo "Processing dataset: $dataset, A: $a, G: $g"
                # Execute the training script
                ./multimodal_bootstrap_train_cvae_supervised_age_gender_effect_each.py -D "$dataset" -H $h -A $a -G $g -R 0
                ./multimodal_bootstrap_test_cvae_supervised_age_gender_effect_each.py -D "$dataset"
                ./multimodal_bootstrap_cvae_group_analysis_1x1_age_gender_effect_each.py -D "$dataset" -H $h
            done
        done
    done
done

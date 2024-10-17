H_VALUES=("100 100 100 10" "90 90 10" "90 90 90 10" "90 90 90 90 90 10" "100 100 10" "300 300 30" "200 200 10" "64 32 16 10" "20 10" "10 10" "20 20" "50 25 12 10" "10 10 10" "20 20 20" "20 10 10" "10 10 5" "1024 512 10" "1000 10" "512 10" "256 10" "1024 512 256 10" "128 64 32" "256 128 64 32 16" "512 256 128 64 32 16" "1024 512 256 128 64 32 16" "2048 10" "10 10" "20 10" "50 10" "20 20" "110 110 110")





for h in "${H_VALUES[@]}"
do
    for dataset in "av45"
    do
        echo "Processing dataset: $dataset"

        ./multimodal_bootstrap_train_ae_supervised.py -D "$dataset" -H $h -B 0.0001 -M 0.005

        ./multimodal_bootstrap_test_ae_supervised.py -D "$dataset"

        ./multimodal_bootstrap_ae_group_analysis_1x1.py -D "$dataset" -L 0 -H $h

    done
done




for h in "${H_VALUES[@]}"
do
    for dataset in "av45"
    do
        echo "Processing dataset: $dataset"

        ./multimodal_bootstrap_train_ae_supervised_effect_each.py -D "$dataset" -H $h -B 0.0001 -M 0.005

        ./multimodal_bootstrap_test_ae_supervised_effect_each.py -D "$dataset"

        ./multimodal_bootstrap_ae_group_analysis_1x1_effect_each.py -D "$dataset" -L 0 -H $h

    done
done

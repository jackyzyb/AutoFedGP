#!/bin/bash

source ~/anaconda3/bin/activate ; conda activate fl_data_dist
cd ..

# real world dataset 
# ColoredMNIST
for seed in 1 2 3 4 5; do
    for i in 0 1 2; do
        python main_domain_bed_offline.py --data_dir=/shared/rsaas/enyij2/domainbed/data --dataset ColoredMNIST \
        --uda_holdout_fraction 0.001  --test_envs $i --iter_idx fl_finetune_offline_$seed --trial_seed $seed --seed $seed \
        --load_trained_model --model_path experiments/fl_domainbed/ColoredMNIST
    done
done

# PACS
for i in 0 1 2 3; do
    python main_domain_bed_offline.py --data_dir=/shared/rsaas/enyij2/domainbed/data --dataset PACS \
    --uda_holdout_fraction 0.15  --test_envs $i --iter_idx fl_finetune_offline \
    --load_trained_model --model_path experiments/fl_domainbed/PACS
done

# VLCS
for seed in 1 2 3 4 5; do
    for i in 0 1 2 3; do
        python main_domain_bed_offline.py --data_dir=/shared/rsaas/enyij2/domainbed/data --dataset VLCS \
        --uda_holdout_fraction 0.05  --test_envs $i --iter_idx fl_finetune_offline_$seed --trial_seed $seed --seed $seed \
        --load_trained_model --model_path experiments/fl_domainbed/VLCS
    done
done

# TerraIncognita
for seed in 1 2 3 4 5; do
    for i in 0 1 2 3; do
        python main_domain_bed_offline.py --data_dir=/shared/rsaas/enyij2/domainbed/data --dataset TerraIncognita \
        --uda_holdout_fraction 0.05  --test_envs $i --iter_idx fl_finetune_offline_$seed --trial_seed $seed --seed $seed \
        --load_trained_model --model_path experiments/fl_domainbed/TerraIncognita
    done
done
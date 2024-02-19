#!/bin/bash
nvidia-smi

source ~/anaconda3/bin/activate ; conda activate fl_data_dist
cd ..

# real world dataset 
# ColorMNIST
for seed in 1 2 3 4 5; do
    for i in 0 1 2; do
        python main_domain_bed.py --data_dir=/shared/rsaas/enyij2/domainbed/data --dataset ColoredMNIST --uda_holdout_fraction 0.001  --test_envs $i --iter_idx fedavg_$seed --trial_seed $seed --seed $seed
    done
done

# PACS
for i in 0 1 2 3; do
    python main_domain_bed.py --data_dir=/shared/rsaas/enyij2/domainbed/data --dataset PACS --uda_holdout_fraction 0.10  --test_envs $i --iter_idx fedavg
done

# OfficeHome
for i in 0 1 2 3; do
    python main_domain_bed.py --data_dir=/shared/rsaas/enyij2/domainbed/data --dataset OfficeHome  --uda_holdout_fraction 0.01 --test_envs $i --iter_idx fedavg
done

# VLCS
for seed in 1 2 3 4 5; do
    for i in 0 1 2 3; do
        python main_domain_bed.py --data_dir=/shared/rsaas/enyij2/domainbed/data --dataset VLCS  --uda_holdout_fraction 0.01 --test_envs $i --iter_idx fedavg_$seed --trial_seed $seed --seed $seed
    done 
done

# TerraIncognita
for seed in 1 2 3 4 5; do
    for i in 0 1 2 3; do
        python main_domain_bed.py --data_dir=/shared/rsaas/enyij2/domainbed/data --dataset TerraIncognita  --uda_holdout_fraction 0.05 --test_envs $i --iter_idx fedavg_$seed --trial_seed $seed --seed $seed
    done 
done
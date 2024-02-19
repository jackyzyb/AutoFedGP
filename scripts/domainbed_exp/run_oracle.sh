#!/bin/bash

source ~/anaconda3/bin/activate ; conda activate fl_data_dist
cd ..

# real world dataset 
# ColorMNIST
for seed in 1 2 3 4 5; do
    for i in 0 1 2; do
        python oracle_domain_bed.py --data_dir=/shared/rsaas/enyij2/domainbed/data --dataset ColoredMNIST  --test_envs $i --iter_idx oracle_$seed --trial_seed $seed --seed $seed
    done
done

# PACS
for i in 0 1 2 3; do
    python oracle_domain_bed.py --data_dir=/shared/rsaas/enyij2/domainbed/data --dataset PACS --test_envs $i --iter_idx oracle
done

# OfficeHome
for i in 0 1 2 3; do
    python oracle_domain_bed.py --data_dir=/shared/rsaas/enyij2/domainbed/data --dataset OfficeHome --test_envs $i --iter_idx oracle
done

# VLCS
for seed in 1 2 3 4 5; do
    for i in 0 1 2 3; do
        python oracle_domain_bed.py --data_dir=/shared/rsaas/enyij2/domainbed/data --dataset VLCS --test_envs $i --iter_idx oracle_$seed --trial_seed $seed --seed $seed
    done 
done

# TerraIncognita
for seed in 1 2 3 4 5; do
    for i in 0 1 2 3; do
        python oracle_domain_bed.py --data_dir=/shared/rsaas/enyij2/domainbed/data --dataset TerraIncognita --test_envs $i --iter_idx oracle_$seed --trial_seed $seed --seed $seed
    done 
done

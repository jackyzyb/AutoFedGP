#!/bin/bash

source ~/anaconda3/bin/activate ; conda activate fl_data_dist
cd ..

# real world dataset 

# ColorMNIST
for seed in 1 2 3 4 5; do
    for i in 0 1 2; do
        python main_domain_bed_auto_beta.py --data_dir=/data/enyij2/domainbed/data --dataset ColoredMNIST --uda_holdout_fraction 0.001 --use_sim --proj_w 0.5  --test_envs $i --iter_idx fl_gp_auto_proj_$seed --target_batch_size 1 --trial_seed $seed --seed $seed
    done
done

# PACS
for i in 0 1 2 3; do
    python main_domain_bed_auto_beta.py --data_dir=/data/enyij2/domainbed/data --dataset PACS --uda_holdout_fraction 0.15 --use_sim --proj_w 0.5 --test_envs $i --iter_idx fl_gp_auto --target_batch_size 4
done

# VLCS
for seed in 1 2 3 4 5; do
    for i in 0 1 2 3; do
        python main_domain_bed_auto_beta.py --data_dir=/data/enyij2/domainbed/data --dataset VLCS  --uda_holdout_fraction 0.05 --use_sim --proj_w 0.5 --test_envs $i --iter_idx fl_gp_auto_proj_$seed --target_batch_size 4 --trial_seed $seed --seed $seed
    done 
done

# OfficeHome
for i in 0 1 2 3; do
    python main_domain_bed_auto_beta.py --data_dir=/data/enyij2/domainbed/data --dataset OfficeHome  --uda_holdout_fraction 0.15 --test_envs $i --use_sim --proj_w 0.5 --iter_idx fl_gp_auto --target_batch_size 128 --auto_lr_ratio 0.5
done

# TerraIncognita
for seed in 1 2 3 4 5; do
    for i in 0 1 2 3; do
        python main_domain_bed_auto_beta.py --data_dir=/data/enyij2/domainbed/data --dataset TerraIncognita  --uda_holdout_fraction 0.05 --use_sim --proj_w 0.5 --test_envs $i --iter_idx fl_gp_auto_proj_$seed --target_batch_size 8 --trial_seed $seed --seed $seed
    done 
done 

# DomainNet
for i in 0 1 2 3 4 5; do
    python main_domain_bed_auto_beta.py --data_dir=/data/enyij2/domainbed/data --dataset DomainNet  --uda_holdout_fraction 0.15 --use_sim --proj_w 0.5 --test_envs $i --iter_idx fl_gp_auto_resnet50 --target_batch_size 64 --num_global_epochs 20
done
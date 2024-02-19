#!/bin/bash

source ~/anaconda3/bin/activate ; conda activate fl_data_dist
cd ..

# femnist

# label shift
for eta in 0 0.05 0.1 0.15 0.3 0.45; do
    python main_noniid_auto_beta.py --dataset fmnist --n_target_samples 100  --n_parties 10 --exp_dir fl_noniid/fmnist/gp_auto --use_sim --proj_w 0.5 --partition imbalance-class-$eta --iter_idx imbalance_$eta
done

# covariate shift target
for noise in 0 0.2 0.4 0.6 0.8; do
    python main_noniid_auto_beta.py --dataset fmnist --n_target_samples 100  --n_parties 10 --exp_dir fl_noniid/fmnist/gp_auto --use_sim --proj_w 0.5  --partition homo --noise $noise --iter_idx noise_$noise --target_batch_size 4
done

# cifar10

# covariate shift
for noise in 0 0.2 0.4 0.6 0.8; do
    python main_noniid_auto_beta.py --dataset cifar10 --n_target_samples 100  --n_parties 10 --exp_dir fl_noniid/cifar10/gp_auto  --use_sim --proj_w 0.5 --partition noniid-#label4 --noise $noise --iter_idx noise_$noise --source_lr 0.005 --target_lr 0.0025
done
#!/bin/bash

source ~/anaconda3/bin/activate ; conda activate fl_data_dist
cd ..

# femnist

# label shift target

# without freezing
for eta in 0 0.05 0.1 0.15 0.3 0.45; do
    python main_noniid_finetune_offline.py --dataset fmnist --n_target_samples 100  --n_parties 10 --exp_dir fl_noniid/fmnist/finetune_offline  --partition imbalance-class-$eta --iter_idx imbalance_$eta --load_trained_model --model_path experiments/fl_noniid/fmnist/base/server_checkpoint_imbalance_$eta.pt
done

# freezing
for eta in 0 0.05 0.1 0.15 0.3 0.45; do
    python main_noniid_finetune_offline.py --dataset fmnist --n_target_samples 100  --n_parties 10 --exp_dir fl_noniid/fmnist/finetune_offline  --partition imbalance-class-$eta --iter_idx imbalance_freeze_$eta --load_trained_model --model_path experiments/fl_noniid/fmnist/base/server_checkpoint_imbalance_$eta.pt --freeze
done

# covariate shift target

# without freezing
for noise in 0 0.2 0.4 0.6 0.8; do
    python main_noniid_finetune_offline.py --dataset fmnist --n_target_samples 100  --n_parties 10 --exp_dir fl_noniid/fmnist/finetune_offline  --partition homo --noise $noise --iter_idx noise_$noise --load_trained_model --model_path experiments/fl_noniid/fmnist/base/server_checkpoint_noise_$noise.pt
done

# freezing
for noise in 0 0.2 0.4 0.6 0.8; do
    python main_noniid_finetune_offline.py --dataset fmnist --n_target_samples 100  --n_parties 10 --exp_dir fl_noniid/fmnist/finetune_offline  --partition homo --noise $noise --iter_idx noise_freeze_$noise --load_trained_model --model_path experiments/fl_noniid/fmnist/base/server_checkpoint_noise_$noise.pt --freeze
done

# cifar10

# covariate shift

# without freezing
for noise in 0 0.2 0.4 0.6 0.8; do
    python main_noniid_finetune_offline.py --dataset cifar10 --partition noniid-#label4 --n_target_samples 100  --n_parties 10 --exp_dir fl_noniid/cifar10/finetune_offline --noise $noise --iter_idx noise_$noise --load_trained_model --model_path experiments/fl_noniid/cifar10/base/server_checkpoint_noise_$noise.pt --source_lr 0.005 --target_lr 0.0025
done

# freezing
for noise in 0 0.2 0.4 0.6 0.8; do
    python main_noniid_finetune_offline.py --dataset cifar10 --partition noniid-#label4 --n_target_samples 100  --n_parties 10 --exp_dir fl_noniid/cifar10/finetune_offline --noise $noise --iter_idx noise_freeze_$noise --load_trained_model --model_path experiments/fl_noniid/cifar10/base/server_checkpoint_noise_$noise.pt --source_lr 0.005 --target_lr 0.0025 --freeze
done
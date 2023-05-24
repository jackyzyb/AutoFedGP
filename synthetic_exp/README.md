# Synthetic Experiment for Federated Auto-weighted Domain Adaptation 

To run the synthetic experiment, simply do `python train_test_model_auto_beta.py`.
In the `train_test_model_auto_beta.py`: if `auto_beta = True` then FedGP and FedDA will use the auto-chosen beta in their training, otherwise FedGP and FedDA will use beta=0.5 all the time; if `is_estimation = True` then FedGP and FedDA will use the estimated d and sigma for their auto-chosen beta in their training, otherwise FedGP and FedDA will use ground-truth d and sigma for their auto-chosen beta.

To see the result, simply do `python plot_auto_beta.py`. Make sure the settings are the same as `train_test_model_auto_beta.py`.
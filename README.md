# Principled Federated Domain Adaptation: Gradient Projection and Auto-Weighting

This is the repository for our paper ["Principled Federated Domain Adaptation: Gradient Projection and Auto-Weighting"](https://openreview.net/forum?id=6J3ehSUrMU)

Federated Domain Adaptation (FDA) describes the federated learning (FL) setting where source clients and a server work collaboratively to improve the performance of a target client where limited data is available. The domain shift between the source and target domains, coupled with limited data of the target client, makes FDA a challenging problem, e.g., common techniques such as federated averaging and fine-tuning fail due to domain shift and data scarcity. 
To theoretically understand the problem, we introduce new metrics that characterize the FDA setting and a theoretical framework with novel theorems for analyzing the performance of server aggregation rules. Further, we propose a novel lightweight aggregation rule, Federated Gradient Projection (*FedGP*), which significantly improves the target performance with domain shift and data scarcity. Moreover, our theory suggests an *auto-weighting scheme* that finds the optimal combinations of the source and target gradients. This scheme improves both *FedGP* and a simpler heuristic aggregation rule. Extensive experiments verify the theoretical insights and illustrate the effectiveness of the proposed methods in practice.

## Table of Contents
1. [Installation](#installation)
2. [FedDA and FedGP](#fedda-and-fedgp)
3. [Auto-weighted versions of FedDA and FedGP](#auto-weighted-versions-of-fedda-and-fedgp)
4. [Other baselines](#other-baselines)
5. [Contributors](#contributors)
6. [How to cite?](#how-to-cite)
7. [Credits](#credits)

## Installation
To install requirements, one can run:

`pip install -r requirements.txt`

For the synthetic experiment used in the paper, please refer to [synthetic_exp](https://github.com/jackyzyb/AutoFedGP/tree/main/synthetic_exp).

## FedDA and FedGP
We provide scripts to run *FedDA* and *FedGP* experiments, for both the non-iid and domainbed datasets. One can run `run_fedda_0.5.sh` and `run_fedgp_0.5.sh` in `scripts/noniid_exp` and `scripts/domainbed_exp` folders.

## Auto-weighted versions of FedDA and FedGP
Additionally, we provide scripts to run *FedDA* and *FedGP* experiments of their auto versions, for both the non-iid and domainbed datasets. One can run `run_fedda_auto.sh` and `run_fedgp_auto.sh` in `scripts/noniid_exp` and `scripts/domainbed_exp` folders.

## Other baselines
One can run `run_fedavg.sh`, `run_target_only.sh`, or `run_oracle.sh` in `scripts/noniid_exp` and `scripts/domainbed_exp` folders, to reproduce results of source-only, target-only, and upper bound baselines.

## Contributors
- Enyi Jiang - enyij2@illinois.edu
- Yibo Jacky Zhang - yiboz@stanford.edu
- Sanmi Koyejo - sanmi@cs.stanford.edu

## How to cite?
Thanks for your interest in our work. If you find it useful, please cite our paper as follows. 

```
@inproceedings{
jiang2024principled,
title={Principled Federated Domain Adaptation: Gradient Projection and Auto-Weighting},
author={Enyi Jiang and Yibo Jacky Zhang and Sanmi Koyejo},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=6J3ehSUrMU}
}
```

## Credits
Parts of the code in this repo is based on

- https://github.com/nicchiou/domain_adapt_cxr

- https://github.com/uiuc-federated-learning/ml-fault-injector

- https://github.com/Xtra-Computing/NIID-Bench

- https://github.com/facebookresearch/DomainBed

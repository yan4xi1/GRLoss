## Title
Boosting Single Positive Multi-label Classification with Generalized Robust Loss

## Abstract
Multi-label learning (MLL) requires comprehensive multi-semantic annotations that is hard to fully obtain, thus resulting in missing labels scenarios. In this paper, we investigate Single Positive Multi-label Learning (SPML), where each image is associated with merely one positive label. Existing SPML methods only focus on designing losses using mechanisms such as hard pseudo-labeling and robust losses, mostly leading to unacceptable false negatives. To address this issue, we first propose a generalized loss framework based on expected risk minimization of SPML, and point out that current loss functions can be seamlessly converted into our framework. In particular, we design the generalized robust loss from our framework, which enjoys flexible coordination between false positives and false negatives, and can additionally deal with the imbalance between positive and negative samples. Extensive experiments show that our approach can significantly improve SPML performance and our approach outperforms the vast majority of methods on all the four benchmarks.


## Dataset Preparation
See the `README.md` file in the `data` directory for instructions on downloading and setting up the datasets.

## Model Training & Evaluation
You can train and evaluate the models by
```
python main.py --dataset [dataset] \
               --lr [learning_rate] \
               --linear_init [linear_init] \
               --beta [beta] \
               --alpha [alpha] \
               --q2q3 [q2q3] \
```
linear_init represents the number of epochs for linear initialization, with a default value of 0.Beta is a list [($w^{(0)}$), ($w^{(T)}$), ($b^{(0)}$), ($b^{(T)}$)], for example [0, 2, -2, -2]. Alpha is a list [($\mu^{(0)}$), ($\sigma^{(0)}$), ($\mu^{(T)}$), ($\sigma^{(T)}$)], for example [0.5, 2, 0.8, 0.5]. q2q3 is a list [($q_2$), ($q_3$)], for example [0.01, 1].

## How to cite

## Acknowledgements
Our code is heavily built upon [Multi-Label Learning from Single Positive Labels](https://github.com/elijahcole/single-positive-multi-label).

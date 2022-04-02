# Dataset Estimation

## Tabular datasets

The folder contains the arranged tabular datasets from Automl and Kaggle.

Each dataset is represented as a '.pkl', which contains the name, id, categorical feature names, numerical feature names, labels and features. Specifically, the categorical features are all relabeled from 0 to the number categories. 

The subfolder name represents the source of the datasets 

## Curves

The curves folder contains the training curves with different pilot size. 

Where each 'csv' contains the names, sources of the dataset. 'all_x' represents the real test set performance when training with x data points. 'small_x' represents the $s_x$as defined in paper.




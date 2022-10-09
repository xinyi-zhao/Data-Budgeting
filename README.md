# Dataset Budgeting for Machine Learning 

Data is the fuel powering AI and creates tremendous value for many domains. However, collecting datasets for AI is a time-consuming, expensive, and complicated endeavor. For practitioners, data investment remains to be a leap of faith in practice. In this work, we study the data budgeting problem and formulate it as two sub-problems: predicting (1) what is the saturating performance if given enough data, and (2) how many data points are needed to reach near the saturating performance. Different from traditional dataset-independent methods like PowerLaw, we proposed a learning method to solve data budgeting problems. To support and systematically evaluate the learning-based method for data budgeting, we curate a large collection of 383 tabular ML datasets, along with their data vs performance curves. Our empirical evaluation shows that it is possible to perform data budgeting given a small pilot study dataset with as few as $50$ data points.

## Tabular datasets

The folder contains the arranged tabular datasets from Automl and Kaggle.

Each dataset is represented as a '.pkl', which contains the name, id, categorical feature names, numerical feature names, labels and features. Specifically, the categorical features are all relabeled from 0 to the number categories. 

The subfolder name represents the source of the datasets 

## Curves

The curves folder contains the training curves with different pilot size. 

Where each 'csv' contains the names, sources of the dataset. 'all_x' represents the real test set performance when training with x data points. 'small_x' represents the $s_x$as defined in paper.




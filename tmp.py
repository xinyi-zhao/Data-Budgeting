import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import powerlaw
import seaborn
from sklearn.cluster import KMeans,AgglomerativeClustering
#from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier,MLPRegressor
import warnings
import math
from scipy.optimize import curve_fit
import torch
import os
import torch.nn as nn
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score,mean_squared_error
import difflib
from utils.parse_cmd import parse_cmd
warnings.filterwarnings("ignore")

df = pd.read_csv('dataset_data_100.csv')
drop_keys = []
for i in range(90):
    drop_keys.append('small2_'+str(i))
    drop_keys.append('small3_'+str(i))
    drop_keys.append('small2_var_'+str(i))
    drop_keys.append('small3_var_'+str(i))
df.drop(drop_keys,axis = 1)
df.to_csv('dataset_data_100_new.csv')
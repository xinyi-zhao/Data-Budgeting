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

cluster_label =  pkl.load(open('utils/cluster_information.pkl','rb'))
def read_dataset_and_save_basic_value(target_value, train_alg, use_file):
    df = pd.read_csv('../arrange/'+use_file[0])
    drop_keys = []
    for i in range(90):
        drop_keys.append('small2_'+str(i))
        drop_keys.append('small3_'+str(i))
        drop_keys.append('small2_var_'+str(i))
        drop_keys.append('small3_var_'+str(i))
    df.drop(drop_keys,axis = 1)
    if('dataset_data_noised.csv' in use_file):
        df2 = pd.read_csv('../arrange/dataset_data_noised.csv')
        df = df.append(df2)
    df_pre = df
    target_keys = []
    target_keys_2 = []
    target_key_all = []
    for i in range(0,pilot_length - 10):
        target_keys.append('small_'+str(i))
        target_keys_2.append('small_var_'+str(i))
    for i in range(2000):
        target_key_all.append('all_'+str(i))
    pre_save = {}
    pre_save_all = {}
    for index, row in df_pre.iterrows():
        try:
            pre_save[row['name']+str(row['dataset_num'])+row['noised']+row['Evaluation Method']] = row[target_keys].tolist()
            pre_save[row['name']+str(row['dataset_num'])+row['noised']+row['Evaluation Method']+'_var'] = row[target_keys_2].tolist()
            pre_save_all[row['name']+str(row['dataset_num'])+row['noised']+row['Evaluation Method']] = row[target_key_all].tolist()
        except:
            ii = 1
    df = df.loc[df['Evaluation Method'].isin([target_value])]
    df = df.loc[df['dataset_type'].isin(['clf','multi_clf'])]
    df = df.loc[df['Learning Algorithm'].isin([train_alg])]
    print('finish import datasets')
    return df,pre_save,pre_save_all

def string_similar(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).quick_ratio()
def modify(s):
    if(len(s)<=4 or s[0]!='B' or s[1]!='N' or s[2]!='G'):
        return s
    return s[4:-1]

def get_meta_data(X,y, task_type,n_cnt, c_cnt, meta_value):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=min(1000, len(X) - 2000), random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=0)
    X_tmp, X_train, y_tmp, y_train = train_test_split(X_train, y_train, test_size=100, random_state=0)
    ret = []
    coe_v = [0,0,0,0,0]
    num1 = 0
    num2 = 0
    for i in range(X_train.shape[1]):
        coe_v.append(r2_score(X_train[:,i], y_train))
        arr_appear = dict((a, X_train[:,i].tolist().count(a)) for a in X_train[:,i])
        #coe_v.append(1.0*max(arr_appear.values())/50.0)
        if(max(arr_appear.values())< num * 0.95):
            if(i < n_cnt):
                num1 += 1
            else:
                num2 += 1
    if('feature_cnt' in meta_value):
        ret.extend([num1,num2])
    if('max_coev' in meta_value):
        coe_v.sort(reverse=True)
        coe_v = coe_v[0:5]
        ret.extend(coev)
    if('label_ratio' in meta_value):
        coe_v = []
        arr_appear = dict((a, y_train.tolist().count(a)) for a in y_train)
        coe_v.append(len(arr_appear.keys()))
        coe_v.append(1.0*min(arr_appear.values())/100.0)
        coe_v.append(1.0*max(arr_appear.values())/100.0)
        ret.extend(coev)
    return ret
def get_train_test(df, pre_save, pre_save_all, _r, use_value, meta_value, use_num,stop_ratio = 0.95):
    chosen_label = np.array(range(100))
    np.random.seed(_r)
    np.random.shuffle(chosen_label)
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    print(df.shape)
    print(use_value)
    for index,row in df.iterrows():
        v = []
        for value_name in use_value:
            for num in use_num:
                v.append(pre_save[row['name']+str(row['dataset_num'])+row['noised']+value_name][num])
        #print(len(v))
        #print(v)
        for i in range(len(v)):
            if(math.isnan(v[i])):
                v[i] = 0
        if(meta_value != []):
            try:
                data =  pkl.load(open(os.path.join('../auto_maker/'+row['File_dir'], '{}'.format(row['File_name'])), 'rb')) 
            except:
                try:
                    data =  pkl.load(open(os.path.join('../auto_maker/'+row['File_dir'], '{}_kaggle.pkl'.format(row['File_name'])), 'rb'))
                except:
                    data =  pkl.load(open(os.path.join('../auto_maker/'+row['File_dir'], '{}_kaggle'.format(row['File_name'])), 'rb')) 
            
            X = None
            if(data['X_num'].size > 0 and data['X_cat'].size>0):
                X = np.concatenate([data['X_num'], data['X_cat']], -1)
            elif(data['X_num'].size > 0):
                X = data['X_num']
            else:
                X = data['X_cat']
            y = data['y']
            task_type = row['dataset_type']
            if(task_type == 'multi_clf'):
                xx, y = np.unique(y, return_inverse=True)
            X = (X - np.mean(X, 0)) / np.clip(np.std(X, 0), 1e-12, None)
            X, y = X[:3000], y[:3000]
            if(row['File_dir'][-3:]=='reg'):
                y_median = np.median(np.array(y))
                y = (y > y_median).astype(int)
            v.extend(get_meta_data(X,y,task_type,row['numerical_cnt'],row['categorical_cnt'], meta_value))
        vy = row['all_1990']
        if(chosen_label[cluster_label[row['name']]]<80):
            X_train.append(v)
            y_train.append(vy)
        else:
            X_test.append(v)
            y_test.append(vy)
        
    print(len(y_train), len(y_test))
    return np.array(X_train),np.array(X_test),np.array(y_train),np.array(y_test)

def deal_str(x):
    return str(round(x,5))

def main(args):
    pilot_length = args.pilot_length

    df, pre_save, pre_save_all = read_dataset_and_save_basic_value(args.target_value, args.train_alg, args.use_file)
    a = []
    a_mse = []
    result_coef = []
    result_int = []
    for _ in range(args.split_times):
        print('######run_time:'+str(_)+'############' )
        X_train, X_test, y_train, y_test = get_train_test(df,pre_save, pre_save_all,  _, args.use_value, args.meta_value, args.use_num)
        print(X_train.shape)
        print(y_train.shape)
        if(args.method == 'LR'):
            LR = LinearRegression()
        elif(args.method == '2-layer NN'):
            LR = MLPRegressor(hidden_layer_sizes=(50,50))
        elif(args.method == 'RF'):
            LR = RandomForestRegressor()
        elif(args.method == '3-layer NN'):
            LR = MLPRegressor(hidden_layer_sizes=(50,50,20))
        LR.fit(X_train,y_train)
        predict_results=LR.predict(X_test)
        a.append(r2_score(y_test,predict_results))
        a_mse.append(math.sqrt(mean_squared_error(y_test,predict_results)))
        result_coef.append(LR.coef_)
        result_int.append(LR.intercept_)
    print(np.array(a).mean(),np.array(a).var())
    print(np.array(a_mse).mean(),np.array(a_mse).var())
    print(np.array(result_coef).mean(axis=0))
    print(np.array(result_int).mean())
    plt.plot(a,label =  'r2')
    plt.plot(a_mse,label = 'rmse')
    plt.xlabel('test_times')
    plt.legend()
    plt.title('r2 = '+deal_str(np.array(a).mean())+ '  var =' +deal_str(np.array(a).var()) + '\n rmse = '+deal_str(np.array(a_mse).mean())+ '  var =' +deal_str(np.array(a_mse).var()))
    Save_dir = 'result_fig/final_result_prediction/'+args.target_value+'/'
    plt.savefig(Save_dir + args.method+ '_'.join(args.use_value)+'_'.join(args.meta_value))
    
    plt.close()

if __name__ == "__main__":
    pilot_length = 100
    args = parse_cmd()
    main(args)
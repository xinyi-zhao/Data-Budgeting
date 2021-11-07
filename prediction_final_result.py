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
    df = pd.read_csv('../arrange/dataset_data_100.csv')
   # drop_keys = []
    #df.drop(drop_keys,axis = 1)
    #print(df['noised'])
    if('dataset_data_noised.csv' in use_file):
        df2 = pd.read_csv('../arrange/dataset_data_noised.csv')
        df = df.append(df2)
    #print(df['noised'])
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
    df = df.loc[df['Learning Algorithm'].isin([train_alg])]
    df_pre = df_pre.loc[df_pre['Learning Algorithm'].isin([train_alg])]
    for index, row in df_pre.iterrows():
        try:
            pre_save[row['name']+str(row['dataset_num'])+row['noised']+row['Evaluation Method']] = row[target_keys].tolist()
            pre_save[row['name']+str(row['dataset_num'])+row['noised']+row['Evaluation Method']+'_var'] = row[target_keys_2]
            for i in range(len(pre_save[row['name']+str(row['dataset_num'])+row['noised']+row['Evaluation Method']+'_var'])):
                pre_save[row['name']+str(row['dataset_num'])+row['noised']+row['Evaluation Method']+'_var'][i] = math.sqrt(pre_save[row['name']+str(row['dataset_num'])+row['noised']+row['Evaluation Method']+'_var'][i])
            pre_save_all[row['name']+str(row['dataset_num'])+row['noised']+row['Evaluation Method']] = row[target_key_all].tolist()
        except:
            ii = 1
    df = df.loc[df['Evaluation Method'].isin([target_value])]
    df = df.loc[df['dataset_type'].isin(['clf','multi_clf'])]
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
        coe_v.append(abs(r2_score(X_train[:,i], y_train)))
        arr_appear = dict((a, X_train[:,i].tolist().count(a)) for a in X_train[:,i])
        #coe_v.append(1.0*max(arr_appear.values())/50.0)
        if(max(arr_appear.values())< X_train.shape[0] * 0.95):
            if(i < n_cnt):
                num1 += 1
            else:
                num2 += 1
    if('feature_cnt' in meta_value):
        ret.extend([num1,num2])
    if('max_coev' in meta_value):
        coe_v.sort(reverse=True)
        coe_v = coe_v[0:5]
        ret.extend(coe_v)
    if('label_ratio' in meta_value):
        coe_v = []
        arr_appear = dict((a, y_train.tolist().count(a)) for a in y_train)
        coe_v.append(len(arr_appear.keys()))
        coe_v.append(1.0*min(arr_appear.values())/100.0)
        coe_v.append(1.0*max(arr_appear.values())/100.0)
        ret.extend(coe_v)
    return ret
def get_train_test(args,df, pre_save, pre_save_all, _r, use_value, meta_value, use_num,use_auto_ml, auto_ml_result = None):
    chosen_label = np.array(range(100))
    np.random.seed(_r)
    np.random.shuffle(chosen_label)
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    row_info = []
    noised_test = []
    #print(df.shape)
    #print(use_value)
    tmp1 = []
    tmp2 = []
    for index,row in df.iterrows():
        v = []
        if(row['name'] + str(row['dataset_num']) + row['noised'] not in auto_ml_result):
            continue
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

        if(use_auto_ml):
            try:
                #print(row['name'] + str(row['dataset_num']) + row['noised'] and )
               # print(auto_ml_result[row['name'] + str(row['dataset_num']) + row['noised']][-1])
                if(args.target_value == 'f1_score'):
                    vy = auto_ml_result[row['name'] + str(row['dataset_num']) + row['noised']][-1]
                else:
                    vy = auto_ml_result[row['name'] + str(row['dataset_num']) + row['noised']][-2]
                #print(vy)
                tmp1.append(row['all_1990'])
                tmp2.append(vy)
                if(chosen_label[cluster_label[row['name']]]<80):
                    X_train.append(v)
                    y_train.append(vy)
                else:
                    X_test.append(v)
                    y_test.append(vy)
                    row_info.append([row['name'],row['dataset_num'],row['dataset_type'], row['numerical_cnt'],row['categorical_cnt'],v[-3],v[-2],v[-1]])
                    if(row['noised']!='no'):
                        noised_test.append(1)
                    else:
                        noised_test.append(0)
            except:
                 ii = 1
            #print(r2_score(tmp1,tmp2))
        else:
            if(chosen_label[cluster_label[row['name']]]<80):
                #if(row['noised'] !='no'):
                #    continue
                X_train.append(v)
                y_train.append(vy)
            else:
                X_test.append(v)
                y_test.append(vy)
                row_info.append([row['name'],row['dataset_num'],row['dataset_type'],row['numerical_cnt'],row['categorical_cnt']])
                #print(row['noised'])
                if(row['noised']!='no'):
                    noised_test.append(1)
                else:
                    noised_test.append(0)
            
    #print(r2_score(tmp1,tmp2))
    #print(len(y_train), len(y_test))
    #print(row_info)
    return np.array(X_train),np.array(X_test),np.array(y_train),np.array(y_test), noised_test,row_info

def deal_str(x):
    return str(round(x,5))

def main(args):
    pilot_length = args.pilot_length
    auto_ml_result = None
    if(args.use_auto_ml):
        if(args.target_value == 'accuracy'):
            auto_ml_result = pkl.load(open('../arrange/try_auto_ml_result_clean_accuracy.pkl','rb'))
            print(len(auto_ml_result.keys()))
            auto_ml_result_noised = pkl.load(open('../arrange/try_auto_ml_result_noise_accuracy.pkl','rb'))
            print(len(auto_ml_result_noised.keys()))
            auto_ml_result=dict(auto_ml_result, **auto_ml_result_noised)
        elif(args.target_value == 'f1_score'):
            auto_ml_result = pkl.load(open('../arrange/try_auto_ml_result_clean_f1_score.pkl','rb'))
            print(len(auto_ml_result.keys()))
            auto_ml_result_noised = pkl.load(open('../arrange/try_auto_ml_result_noise_f1_score.pkl','rb'))
            print(len(auto_ml_result_noised.keys()))
            auto_ml_result=dict(auto_ml_result, **auto_ml_result_noised)
    print(len(auto_ml_result.keys()))
    df, pre_save, pre_save_all = read_dataset_and_save_basic_value(args.target_value, args.train_alg, args.use_file)
    a = []
    a_mse = []
    a2 = []
    a2_mse = []
    result_coef = []
    result_int = []
    plt.scatter(0,0,s=5,c='b',label = 'multi_clf')
    plt.scatter(0,0,s=5,c='r',label = 'binary_clf')
    for _ in range(args.split_times):
        #print('######run_time:'+str(_)+'############' )
        X_train, X_test, y_train, y_test, test_noise,row_info = get_train_test(args, df,pre_save, pre_save_all,  _, args.use_value, args.meta_value, args.use_num,args.use_auto_ml,auto_ml_result)
       # print(X_train.shape)
       # print(y_train.shape)
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
       # print(X_test)
       # print(y_test)
        a_mse.append(math.sqrt(mean_squared_error(y_test,predict_results)))

        clean_o = []
        clean_t = []
        #print(test_noise)
        #print(row_info)
        
        for i in range(len(y_test)):
            if(test_noise[i] == 0):
                clean_o.append(predict_results[i])
                clean_t.append(y_test[i])
            if(row_info[i][-1] =='multi_clf'):
                plt.scatter(y_test[i],predict_results[i],s=5,c='b')
            else:
                plt.scatter(y_test[i],predict_results[i],s=5,c='r')
            if(abs(y_test[i]-predict_results[i])>0.3):
                print(y_test[i],predict_results[i], row_info[i])
        a2.append(r2_score(clean_t,clean_o))
        a2_mse.append(math.sqrt(mean_squared_error(clean_t,clean_o)))
        result_coef.append(LR.coef_)
        result_int.append(LR.intercept_)
    print(args.use_num)
    print(np.array(a).mean(),np.array(a).var())
    print(np.array(a_mse).mean(),np.array(a_mse).var())

    print(np.array(a2).mean(),np.array(a2).var())
    print(np.array(a2_mse).mean(),np.array(a2_mse).var())

    print(np.array(result_coef).mean(axis=0))
    print(np.array(result_int).mean())
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('AutoML Result')
    plt.ylabel('prediction Result')
    #plt.plot(a,label =  'r2')
    #plt.plot(a_mse,label = 'rmse')
    #plt.xlabel('test_times')
    plt.legend()
    plt.title('r2 = '+deal_str(np.array(a).mean())+ '  var =' +deal_str(np.array(a).var()) + '\n rmse = '+deal_str(np.array(a_mse).mean())+ '  var =' +deal_str(np.array(a_mse).var()))
    Save_dir = 'result_fig/final_result_prediction/'+args.target_value+'/'
    plt.savefig(Save_dir + args.method+ '_'.join(str(args.use_num))+'_'.join(args.meta_value))
    
    plt.close()
    return np.array(a).mean()

if __name__ == "__main__":
    pilot_length = 100
    args = parse_cmd()
    main(args)
    if(False):
        result1 = []
        result2 = []
        for i in range(1,18):
            args.use_num = [i*5]
            result1.append('small_'+str(i*5))
            result2.append(main(args))
        fig=plt.figure(figsize=(10,6))
        plt.plot(result1,result2)
        plt.xticks(result1, result1,color='blue',rotation=60)
        plt.xlabel('Which point we use')
        plt.ylabel('R2')
        plt.title('Use one small_x point to predict the final result')
        plt.savefig('Finding_result/'+args.save_fig)
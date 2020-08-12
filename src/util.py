import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np

def reading_data(PATH,num_line):
    paths = os.listdir(PATH)
    print(paths)
    for path in paths:
        if path[-3:] == 'csv':
            print(path)
            df = pd.read_csv(PATH+path)
            print(df.shape)
            display(df.head(num_line))
            
def simple_preprocess(train,test,ID,TARGET):
    
    """
    int →　そのまま（もしカテゴリなら指定）
    float →　そのまま
    floatだけどintと欠損値　→　とりあえずそのまま（もしカテゴリにしたいなら指定）
    object →　基本的にはtrainもtestもlabelencodingしてからカテゴリに変更してる 
    
    """
    
    #事前準備，ラベルエンコーダと，train,testを一緒にする(train,testの見分けは付くように)
    le = LabelEncoder()
    
    train["adversarial"] = 0
    test["adversarial"] = 1
    df = pd.concat([train,test],sort=False)
    
    objects = pd.DataFrame()
    col_obs = []
    col_fls = []
    col_ins = []
    col_fl_int = []
    
    #boolenの型を変える
    boolen_col = df.select_dtypes(bool).columns
    df[boolen_col] = df[boolen_col].astype(int)
    
    #1列ずつ対処を変えていく
    print("object型のユニーク数：",end=' ')
    
    for col in df.drop([ID,TARGET],axis=1).columns:
        #型がobjectのものはユニーク数を出してから，labelencodeして数値に変更,かつcategoryに
        if df[col].dtype == object:
            if len(df[col].unique())!=len(df):            
                col_obs.append(col)
                print(col,len(df[col].unique()),end=' ')
                df[col] = df[col].fillna("0")
                df[col] = le.fit_transform(df[col])
                df[col] = df[col].astype("category")
            else:
                print(col,'はユニーク')
            
        #floatのものは，intなのかもしれないのでその見極めをする．純粋にfloatならそのまま
        elif df[col].dtype == float:
            
            if len(df) == df[col].fillna(0).apply(lambda x:x.is_integer()).sum():#すべてintegerならfloatintへ
                col_fl_int.append(col)
            else:
                col_fls.append(col)
        
        #intのものは，カテゴリなのか数値なのか怪しい
        elif train[col].dtype == "int64" or train[col].dtype == "int32":
            col_ins.append(col)

    print("")
    print("----------")
    #処理の確認フェーズ
    print("shape:",train.shape)
    print("objects:",len(col_obs))
    display(df[col_obs].head(3))
    
    print("floats:",len(col_fls))
    display(df[col_fls].head(3))
    
    print("ints:",len(col_ins))
    display(df[col_ins].head(3))
    
    print("float to int",len(col_fl_int))
    display(df[col_fl_int].head(3))
    
    print('bool型の列数',len(boolen_col))
    
    print("all:",len(col_obs)+len(col_fls)+len(col_ins)+len(col_fl_int)+len(boolen_col))
    

    #int型とfloatint型はどんなものでどのくらいのユニーク数なのかは確認
    col_ins.extend(col_fl_int)
    print("int型とfloatだけどint型のユニーク数")
    for col in col_ins:
        print(col,len(df[col].unique()),end=' ')
    
    return df,col_ins


def plot_importance(feature_importance, ax=None, height=0.2,
                    xlim=None, ylim=None, title='Feature importance',
                    xlabel='Feature importance', ylabel='Features',
                    importance_type='split', max_num_features=None,
                    ignore_zero=True, figsize=None, grid=True,
                    precision=None, **kwargs):

    importance=feature_importance.values.copy()
    feature_name=feature_importance.index.values.copy()
    
    tuples = sorted(zip(feature_name, importance), key=lambda x: x[1])
    if ignore_zero:
        tuples = [x for x in tuples if x[1] > 0]
    if max_num_features is not None and max_num_features > 0:
        tuples = tuples[-max_num_features:]
    labels, values = zip(*tuples)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    ylocs = np.arange(len(values))
    ax.barh(ylocs, values, align='center', height=height, **kwargs)

    for x, y in zip(values, ylocs):
        ax.text(x + 1, y,
                _float2str(x, precision) if importance_type == 'gain' else x,
                va='center')

    ax.set_yticks(ylocs)
    ax.set_yticklabels(labels)

    if xlim is not None:
        _check_not_tuple_of_2_elements(xlim, 'xlim')
    else:
        xlim = (0, max(values) * 1.1)
    ax.set_xlim(xlim)

    if ylim is not None:
        _check_not_tuple_of_2_elements(ylim, 'ylim')
    else:
        ylim = (-1, len(values))
    ax.set_ylim(ylim)

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.grid(grid)
    return ax
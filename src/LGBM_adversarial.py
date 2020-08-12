from tqdm.notebook import tqdm as tqdm
from sklearn.model_selection import ParameterGrid
import pandas as pd
import numpy as np
import lightgbm as lgb

def make_dataset(df,ID,y):
    X = df.drop([ID,y],axis=1)
    y = df[y]
    return X,lgb.Dataset(X,y)

def lgb_modelling(train,test,option_params,ID,y):
    
    #lightGBM用のデータセットに変換
    X_train,train_lgb = make_dataset(train,ID,y)
    X_test,_ = make_dataset(test,ID,y)
    
    #ハイパラの選択肢を作る
    params = list(ParameterGrid(option_params))
    
    #GridSearchを行う
    min_score = np.inf
    for param in tqdm(params):
        result = lgb.cv(params=param,
                        num_boost_round=1000,
                        train_set = train_lgb,
                        early_stopping_rounds=10,
                        stratified=True,
                        verbose_eval=False)
        
        metric = param['metric']
        score = min(result[metric+'-mean'])
        print(param,score)
        if min_score > score:
            min_score = score
            min_score_iteration = len(result[metric+'-mean'])
            best_hyper_param = param
    
    #このハイパラでもう一度学習予測
    bst = lgb.train(params=best_hyper_param,
                    num_boost_round=min_score_iteration,
                    train_set=train_lgb,
                    verbose_eval=False)
    
    #テストデータに対して予測をする
    pred_test = bst.predict(X_test)
    pred_test = np.exp(pred_test)
    pred_test[pred_test<0] == 0
    
    return pred_test,min_score,bst
from tqdm.notebook import tqdm as tqdm
from sklearn.model_selection import ParameterGrid
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold

def make_dataset(df,ID,y):
    X = df.drop([ID,y],axis=1)
    y = df[y]
    return X,lgb.Dataset(X,y)

def lgb_modelling(train,test,option_params,ID,y):
    
    #対数変換
    train[y] = np.log1p(train[y])
    
    #lightGBM用のデータセットに変換
    X_train,train_lgb = make_dataset(train,ID,y)
    X_test,_ = make_dataset(test,ID,y)
    
    #ハイパラの選択肢を作る
    params = list(ParameterGrid(option_params))
    
    #GridSearchを行う
    min_score = np.inf
    for param in tqdm(params):
        result = lgb.cv(params=param,
                        num_boost_round=10000,
                        train_set = train_lgb,
                        early_stopping_rounds=10,
                        stratified=False,
                        verbose_eval=False)
        
        metric = param['metric']
        score = min(result[metric+'-mean'])
        if min_score > score:
            min_score = score
            min_score_iteration = len(result[metric+'-mean'])
            best_hyper_param = param
        
        print(param,len(result[metric+'-mean']))
        print(score)
    print(min_score,best_hyper_param)
        
    #このハイパラでもう一度学習予測
    bst = lgb.train(params=best_hyper_param,
                    num_boost_round=min_score_iteration,
                    train_set=train_lgb,
                    verbose_eval=False)
    
    #テストデータに対して予測をする
    pred_test = bst.predict(X_test)
    pred_test = np.expm1(pred_test)
    pred_test[pred_test<0] == 0
    
    return pred_test,min_score,bst

def lgb_modelling_weight_averaging(train,test,option_params,ID,y):
    
    #対数変換
    train[y] = np.log1p(train[y])
    
    X_train,train_lgb = make_dataset(train,ID,y)
    X_test,_ = make_dataset(test,ID,y)
    
    #5fold weight averagingを行う
    kf = KFold(n_splits=5)
    preds = []
    scores = []
    fis = []
    best_boosters = []
    
    for train_fold_index, valid_index in tqdm(list(kf.split(train))):
        train_fold, valid = train.loc[train_fold_index,:], train.loc[valid_index,:]
        
        #lightGBM用のデータセットに変換
        X_train_fold,train_fold_lgb = make_dataset(train_fold,ID,y)
        X_valid,valid_lgb = make_dataset(valid,ID,y)
        
        #ハイパラの選択肢を作る
        params = list(ParameterGrid(option_params))
        
        min_score = np.inf
        for param in tqdm(params):

            #train,validでearly_stoppingをかけて学習
            bst = lgb.train(params=param,
                            num_boost_round=10000,
                            train_set=train_fold_lgb,
                            early_stopping_rounds = 10,
                            valid_sets=valid_lgb,
                            verbose_eval=False)

            #学習結果のvalidの最小スコアを取っておく。これでweight averagingする
            metric = param['metric']
            score = bst.best_score['valid_0'][metric]
            
            if min_score > score:
                min_score = score
                min_score_iteration = bst.best_iteration
                best_hyper_param = param
        print(best_hyper_param)
        
        best_booster = lgb.train(params=best_hyper_param,
                                    num_boost_round=min_score_iteration,
                                    train_set=train_lgb,
                                    verbose_eval=False)
        
        scores.append(min_score)
        
        #feature_importance
        fi = dict(zip(best_booster.feature_name(),best_booster.feature_importance(importance_type='gain')))
        fi = pd.Series(fi)
        fis.append(fi)
        print(min_score,fi[fi==0].index)
        
        #テストデータの予測を出力
        pred = best_booster.predict(X_test)
        preds.append(pred)
    
    #weight averaging用のweight算出
    inv_scores = 1/np.array(scores)
    weight = inv_scores/inv_scores.sum()
    
    final_pred = np.average(np.array(preds),weights=weight,axis=0)
    final_pred = np.expm1(final_pred)
    return fis,final_pred,scores
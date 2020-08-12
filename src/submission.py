import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from LGBM import one_loop_lgb_modelling

def make_submission_cv(raw_train,test,option_params,ID,y,estimator=one_loop_lgb_modelling):

    #CV用の準備
    kf = KFold(n_splits=5,random_state=0,shuffle=True)
    predicts = []
    scores = []
    
    #CV
    for train_index, valid_index in kf.split(raw_train):
        #学習データを検証用データに分ける
        train = raw_train.loc[train_index,:]
        valid = raw_train.loc[valid_index,:]

        #ハイパラを調整した上で最もよいハイパラで再度学習予測
        predict,score = one_loop_lgb_modelling(train,test,option_params,ID,y)
        predicts.append(predict)
        scores.append(score)

    #ハイパラ調整時のベストvalidationスコアで重み調整（⇒そのうちこの重みも調整できるように）
    scores = np.array(scores)
    weight = (1/scores)/((1/scores).sum())
    predict = np.average(predicts,axis=0,weights=weight)
    
    return predict,scores
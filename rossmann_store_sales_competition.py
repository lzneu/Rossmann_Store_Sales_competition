#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : rossmann_store_sales_competition.py
# @Author: 投笔从容
# @Date  : 2018/4/3
# @Desc  : 便利店销量预测

import pandas as pd
import datetime
import csv
import numpy as np
import os
import scipy as sp
import xgboost as xgb
import itertools
import operator
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import TransformerMixin
from sklearn import cross_validation
from matplotlib import pylab as plt
plot = True

goal = 'Sales'
myid = 'Id'

#定义一些变换和评判标准
'''
使用不同的loss function的时候要特别注意这个
'''
def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1. / (y[ind]**2)
    return w

def rmspe(yhat, y):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat)**2))
    return rmspe

def rmspe_xg(yhat, y):
    # y是log平滑后的结果
    y = y.get_label()
    y = np.exp(y) - 1
    yhat = np.exp(yhat) - 1
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat)**2))

    return 'rmspe', rmspe


# 加载数据，设定数值型和非数值型数据
def load_data():
    '''

    :return: (train,test,features,features_non_numeric)
    '''

    store = pd.read_csv('./data/store.csv')
    train_org = pd.read_csv('./data/train.csv')
    test_org = pd.read_csv('./data/test.csv')
    train = pd.merge(train_org, store, on='Store', how='left')
    test = pd.merge(test_org, store, on='Store', how='left')
    features = test.columns.tolist()

    # 类别型
    cat_col = ['StateHoliday', 'SchoolHoliday', 'Promo', 'Promo2']
    for col in cat_col:
        train[col] = train[col].astype('str')
        test[col] = test[col].astype('str')

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    features_numeric = test.select_dtypes(include=numerics).columns.tolist()
    features_non_numeric = [f for f in features if f not in features_numeric]

    return (train, test, features, features_non_numeric)

# 数据与特征处理
def process_data(train, test, features, features_non_numeric):
    """
        Feature engineering and selection.
    """
    train = train[train['Sales'] > 0]
    for data in [train, test]:
        # 年 月 日
        data['year'] = data.Date.apply(lambda x: x.split('-')[0])
        data['year'] = data['year'].astype(float)
        data['month'] = data.Date.apply(lambda x: x.split('-')[1])
        data['month'] = data['month'].astype(float)
        data['day'] = data.Date.apply(lambda x: x.split('-')[2])
        data['day'] = data['day'].astype(float)

        # PromoInterval  促销间隔 "Jan,Apr,Jul,Oct"
        data['promojan'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if 'Jan' in x else 0)
        data['promofeb'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if 'Feb' in x else 0)
        data['promomar'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if 'Mar' in x else 0)
        data['promoapr'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if 'Apr' in x else 0)
        data['promomay'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if 'May' in x else 0)
        data['promojun'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if 'Jun' in x else 0)
        data['promojul'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if 'Jul' in x else 0)
        data['promoaug'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if 'Aug' in x else 0)
        data['promosept'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if 'Sept' in x else 0)
        data['promootc'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if 'Oct' in x else 0)
        data['promonov'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if 'Nov' in x else 0)
        data['promodec'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if 'Dec' in x else 0)

    features = test.columns.tolist()
    # Features set.
    noisy_features = [myid, 'Date', 'Open']
    features = [c for c in features if c not in noisy_features]
    features_non_numeric = [c for c in features_non_numeric if c not in noisy_features]



    #fill NA
    class DataFrameImputer(TransformerMixin):
        # http://stackoverflow.com/questions/25239958/impute-categorical-missing-values-in-scikit-learn
        def __init__(self):
            '''
            填补缺失值
            类别型：填补列的众数
            数值型：均值
            '''
        def fit(self, X, y=None):
            self.fill = pd.Series([X[c].value_counts().index[0]  # mode
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],  # mean
            index=X.columns)
            return self
        def transform(self, X, y=None):
            return X.fillna(self.fill)

    train = DataFrameImputer().fit_transform(train)
    test = DataFrameImputer().fit_transform(test)

    # 预处理类型数据
    le = LabelEncoder()
    for col in features_non_numeric:
        # sca_data1 = list(train[col]) + list(test[col])
        le.fit(list(train[col]) + list(test[col]))
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])

    # LR 和 神经网络这种模型都对于输入数据的幅度极为敏感，需要先做归一化处理
    scaler = StandardScaler()
    for col in set(features) - set(features_non_numeric) - set([]):  # TODO set([])是为了存放无需归一化的特征
        sca_data = np.array(list(train[col]) + list(test[col]))
        scaler.fit(sca_data.reshape(-1, 1))
        train[col] = scaler.transform(train[col].reshape(-1, 1))
        test[col] = scaler.transform(test[col].reshape(-1, 1))

    return (train, test, features, features_non_numeric)


# 训练与分析
def XGB_native(train, test, features, features_non_numeric):
    # TODO 格点搜索
    depth = 13
    eta = 0.01
    ntrees = 8000
    mcw = 3
    params = {
        'objective': 'reg:linear',
        'booster': 'gbtree',
        'eta': eta,
        'max_depth': depth,
        'min_child_weight': mcw,
        'subsample': 0.9,
        'colsample_bytree': 0.7,
        'silent': 1
    }
    print('Running with params: ' + str(params))
    print('Running with ntrees: ' + str(ntrees))
    print('Running with features: ' + str(features))

    # train model with local split
    tsize = 0.05
    X_train, X_test = cross_validation.train_test_split(train, test_size=tsize)
    dtrain = xgb.DMatrix(X_train[features], np.log(X_train[goal] + 1))
    dvaild = xgb.DMatrix(X_test[features], np.log(X_test[goal] + 1))
    watchlist = [(dvaild, 'eval'), (dtrain, 'train')]
    gbm = xgb.train(params, dtrain, ntrees, watchlist, early_stopping_rounds=100,
                    feval=rmspe_xg, verbose_eval=True)
    # eval 放评价数据,可观察
    train_probs = gbm.predict(xgb.DMatrix(X_test[features]))
    indices = train_probs < 0
    #TODO 看下这个数据结构
    train_probs[indices] = 0
    error = rmspe(np.exp(train_probs) - 1, X_test[goal].values)
    print(error)

    # predict and export
    test_probs = gbm.predict(xgb.DMatrix(test[features]))
    indices = test_probs < 0
    test_probs[indices] = 0

    # 提交
    submission = pd.DataFrame({myid: test[myid], goal: np.exp(test_probs) - 1})
    if not os.path.exists('result/'):
        os.makedirs('result/')
    submission.to_csv('./result/dat-xgb_d%s_eta%s_ntree%s_mcw%s_tsize%s.csv' % (str(depth),str(eta),str(ntrees),str(mcw),str(tsize)) , index=False)

    # feature importance
    if plot:
        outfile = open('xgb.fmap', 'w')
        i = 0
        for feat in features:
            outfile.write('{0}\t{1}\tq\n'.format(i, feat))
            i = i + 1
        outfile.close()

        importance = gbm.get_fscore(fmap='xgb.fmap')
        importance = sorted(importance.items(), key=operator.itemgetter(1))
        df = pd.DataFrame(importance, columns=['feature', 'fscore'])
        df['fscore'] = df['fscore'] / df['fscore'].sum()

        # Plotitup
        plt.figure()
        plt.plot()
        df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(25, 15))
        plt.title('XGboost Feature Importance')
        plt.xlabel('relative importance')
        plt.gcf().savefig('Feature_Importance_xgb_d%s_eta%s_ntree%s_mcw%s_stize%s.png' % (str(depth), str(eta), str(ntrees), str(mcw), str(tsize)))




if __name__ == '__main__':
    print('=>载入数据中...')
    train, test, features, features_non_numeric = load_data()
    print('=>处理数据与特征工程中...')
    train, test, features, features_non_numeric = process_data(train, test, features, features_non_numeric)
    print('=>使用XGBoost建模中...')
    XGB_native(train, test, features, features_non_numeric)

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
df_all = pd.read_csv('./cache/output/service.csv',index_col=0,low_memory=False)

def plot_features(booster, figsize):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    return plot_importance(booster=booster, ax=ax)

gender = ['female', 'male']
df_all = df_all.query('gender in @gender and race !="unknown"').copy()
df_all = df_all.dropna(subset='duration_seconds')


race_dummies = pd.get_dummies(df_all['race'], prefix='race')
df_all = pd.concat([df_all, race_dummies], axis=1)

df_all['gender'] = df_all['gender'].map({'male': 1, 'female': 0})
scaler = StandardScaler()


df_all[['user_followers_count',
        'user_following_count', 'user_tweet_count', 'user_listed_count', 'public_retweet_count', 'public_reply_count',
        'public_like_count',
        'public_quote_count', 'public_impression_count']] =\
scaler.fit_transform(df_all[['user_followers_count',
                             'user_following_count', 'user_tweet_count', 'user_listed_count', 'public_retweet_count',
                             'public_reply_count', 'public_like_count',
                             'public_quote_count', 'public_impression_count']])

feature_list = ['race_asian', 'race_black', 'race_hispanic', 'race_white', 'user_followers_count',
                'user_following_count', 'user_tweet_count', 'user_listed_count', 'public_retweet_count',
                'public_reply_count', 'public_like_count',
                'public_quote_count', 'public_impression_count', 'gender']



X = df_all[feature_list]
y = df_all['duration_seconds']

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25)

xgb_parms = {
    # 'max_depth':4,
    'learning_rate':0.05,
    # 'subsample':0.8,
    # 'colsample_bytree':0.6,
    # 'eval_metric':'logloss',
    # 'random_state':231
}

my_model = XGBRegressor(n_estimators=1000,learning_rate=0.05,max_depth=4,random_state=231)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, verbose=False)

# make predictions
predictions = my_model.predict(test_X)

from sklearn.metrics import mean_absolute_error

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))

plot_features(my_model, (20, 12))


import re
import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"

df_all['user_followers_count'].max()
g = sns.lmplot(df_all,x='user_followers_count',y='duration_seconds')

df_all['gender'] = df_all['gender'].astype('category')
sns.boxplot(df_all,x='duration_seconds',y='gender',hue='race')
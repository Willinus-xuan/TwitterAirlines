import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)

#
# def plot_features(booster, figsize):
#     fig, ax = plt.subplots(1, 1, figsize=figsize)
#     g = plot_importance(booster=booster, ax=ax)
#     plt.savefig('./cache/plots/regression.png')
#     return


df = pd.read_csv('./cache/output/for_DL.csv', index_col=0)


gender = ['female', 'male']
df_reg = df.query('gender in @gender and race !="unknown"').copy()

# extract geners




# feature engineering
race_dummies = pd.get_dummies(df_reg['race'], prefix='race')
df_reg = pd.concat([df_reg, race_dummies], axis=1)

service_dummies = pd.get_dummies(df_reg['service'], prefix='service')
df_reg = pd.concat([df_reg, service_dummies], axis=1)

df_reg['gender'] = df_reg['gender'].map({'male': 1, 'female': 0})

scaler = StandardScaler()

df_reg[['user_followers_count',
        'user_following_count', 'user_tweet_count', 'user_listed_count', 'public_retweet_count', 'public_reply_count',
        'public_like_count',
        'public_quote_count', 'public_impression_count']] =\
scaler.fit_transform(df_reg[['user_followers_count',
                             'user_following_count', 'user_tweet_count', 'user_listed_count', 'public_retweet_count',
                             'public_reply_count', 'public_like_count',
                             'public_quote_count', 'public_impression_count']])


import seaborn as sns

t = df[df['public_like_count']==df['public_like_count'].max()]


sns.scatterplot(df_reg,x = 'public_like_count',y = 'duration_minutes')



feature_list = ['race_asian', 'race_black', 'race_hispanic', 'race_white', 'user_followers_count',
                'user_following_count', 'user_tweet_count', 'user_listed_count', 'public_retweet_count',
                'public_reply_count', 'public_like_count',
                'public_quote_count', 'public_impression_count', 'service_AF', 'service_AN', 'service_CM', 'service_JB',
                'service_JK',
                'service_KF', 'service_LT', 'service_MT', 'service_RQ', 'service_SZ', 'gender']

# build a xgboost model

X = df_reg[feature_list]
y = df_reg['duration_seconds']

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25)

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.02)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, verbose=False)

# make predictions
predictions = my_model.predict(test_X)

from sklearn.metrics import mean_absolute_error

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))


# fig, ax = plt.subplots(1, 1, figsize=(20,8))
# sorted_idx = my_model.feature_importances_.argsort()
# plt.barh(my_model.feature_names_in_[sorted_idx], my_model.feature_importances_[sorted_idx])
# plt.xlabel("Xgboost Feature Importance")
# plt.savefig('./cache/plots/feature_importance.png')
#
# plot_features(my_model, (20, 12))
# plot_importance(my_model)

def plot_importance(type):
    f_importance = my_model.get_booster().get_score(importance_type=type)
    importance_df = pd.DataFrame.from_dict(data=f_importance,
                                           orient='index')

    sorted_idx = np.array([*f_importance.values()]).argsort()

    fig, ax = plt.subplots(1, 1, figsize=(20,8))
    plt.barh(importance_df.index[sorted_idx], np.array([*f_importance.values()])[sorted_idx])
    plt.xlabel("Xgboost Feature Importance")
    plt.savefig('./cache/plots/feature_importance_{}.png'.format(type))

plot_importance(type='gain')

plot_importance(type='weight')
# import re
# import pandas as pd
# import os
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# import plotly.express as px
# import plotly.io as pio
# pio.renderers.default = "browser"
#
# df_reg['user_followers_count'].max()
# g = sns.lmplot(df_reg,x='user_followers_count',y='duration_seconds')
# g.set(xlim=(0,None))
# plt.show()
# df_reg['gender'] = df_reg['gender'].astype('category')
# sns.boxplot(df_reg,x='duration_seconds',y='gender',hue='race')




import re
import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from functools import reduce
pio.renderers.default = "browser"

df = pd.read_csv('./cache/output/service.csv',index_col=0,low_memory=False)
t = df.query('service !="unknown"').copy()
top10 = list(t.service.value_counts().iloc[:10].index)
df_top = df[df['service'].isin(top10)].copy()

ps = PorterStemmer()
def preprocess_text(text):
    if isinstance(text,float)!=True:
        text = text.lower()
        text = re.sub('[^a-zA-Z]+', ' ', text)
        words = word_tokenize(text)
        stemmed_sentence = reduce(lambda x, y: x + " " + ps.stem(y), words, "")
        return stemmed_sentence.strip()
    else:
        return np.nan


sns.scatterplot(df_top,x = 'public_like_count',y = 'response_time')


df_top['reply_text'] = df_top['reply_text'].apply(lambda x:preprocess_text(x))

service = 'JB'
def plot_frequency(service):
    df_kf = df_top.query('service ==@service')
    text_list = df_kf.reply_text.tolist()

    frequency = {}
    length = 0
    for text in text_list:
        str_list = text.split()
        for word in str_list:
            length+=1
            frequency[word] = frequency.setdefault(word, 0) + 1

    sorry_term = ['apolog','sorri','dispointed']
    thank_term = ['thank']
    dm_term = ['dm']
    def cal_frequency(term):
        output = 0
        for i in term:
            try:
                output+=frequency[i]
            except:
                continue
        return output

    sorry = round(cal_frequency(sorry_term)/length,4)
    thank = round(cal_frequency(thank_term)/length,4)
    dm = round(cal_frequency(dm_term)/length,4)
    data = [['sorry', sorry], ['thank', thank], ['dm', dm]]
    df_word = pd.DataFrame(columns=['term', 'frequency'], data=data)
    g = sns.barplot(df_word,x='term',y='frequency')
    g.set(title='word usage of representative:{}'.format(service))
    g.set(ylim=(0,0.04))
    plt.savefig('./cache/plots/service({})'.format(service))
    plt.show()


for i in top10:
    plot_frequency(i)













#
# # duration_seconds, public_impression_count, user_follower_count
# g = sns.lmplot(df_top,x='public_impression_count',y='duration_seconds',col='service',col_wrap = 5,aspect=0.5, fit_reg=True,facet_kws=dict(sharex=False, sharey=False))
# plt.savefig('./cache/plots/public_impression.png')
#
# t = df_top.query('service=="MT"')
#
# # data visualization of duration seconds, public impression counts and user follower count relationship
# g = sns.lmplot(df_top,x='user_followers_count',y='duration_seconds',col='service',col_wrap = 5,aspect=0.5, fit_reg=True,facet_kws=dict(sharex=False, sharey=False))
# plt.savefig('./cache/plots/user_follower.png')
#
# g = sns.lmplot(df_top,x='public_like_count',y='duration_seconds',col='service',col_wrap = 5,aspect=0.5, fit_reg=True,facet_kws=dict(sharex=False, sharey=False))
# plt.savefig('./cache/plots/pubic_like.png')


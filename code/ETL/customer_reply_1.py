import re
import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"

import plotly.graph_objects as go
df = pd.read_csv('./cache/output/clean_gender_race.csv',index_col=0,low_memory=False)


## split followers varibales
df['public_retweet_count'] = df['public_metrics'].apply(lambda x:eval(x)['retweet_count'])
df['public_reply_count'] = df['public_metrics'].apply(lambda x:eval(x)['reply_count'])
df['public_like_count'] = df['public_metrics'].apply(lambda x:eval(x)['like_count'])
df['public_quote_count'] = df['public_metrics'].apply(lambda x:eval(x)['quote_count'])
df['public_impression_count'] = df['public_metrics'].apply(lambda x:eval(x)['impression_count'])
df['user_followers_count'] = df['user_metrics'].apply(lambda x:eval(x)['followers_count'])
df['user_following_count'] = df['user_metrics'].apply(lambda x:eval(x)['following_count'])
df['user_tweet_count'] = df['user_metrics'].apply(lambda x:eval(x)['tweet_count'])
df['user_listed_count'] = df['user_metrics'].apply(lambda x:eval(x)['listed_count'])


df['service'] = df['reply_text'].apply(lambda x:x.split('^')[-1] if x not in [np.nan] and len(x.split('^'))!=1 else np.nan)
df['service'] = df['service'].str.replace(r'http\S+','',regex=True)
df['service'] = df['service'].fillna('unknown')
df['service'] = df['service'].str.strip()
df['reply_text'] = df['reply_text'].fillna('')


t = df.query('service !="unknown"').copy()

top10 = list(t.service.value_counts().iloc[:10].index)
g = sns.countplot(t,x='service',order = top10,orient='v')
g.set_title('The number of reply texts of Top 10 client representatives ')
plt.savefig('./cache/plots/service.png')

g= sns.barplot(df[df['service'].isin(top10)],x='service',y='duration_minutes',order = top10)
g.set_title('The response minutes of Top 10 client representatives')
plt.savefig('./cache/plots/response_time.png')

s1 = t.service.value_counts().iloc[:10]
s2 = df[df['service'].isin(top10)].groupby('service')['duration_minutes'].mean()

df_corr = pd.concat([s1,s2],axis=1).reset_index()
df_corr.rename(columns={'index':'representative','service':'number_of_replies','duration_minutes':'mean_duration_minutes'},inplace=True)
df_corr.mean_duration_minutes = df_corr.mean_duration_minutes.apply(lambda x:round(x,2))


fig = px.scatter(df_corr,x='number_of_replies',y='mean_duration_minutes',color='representative',size='mean_duration_minutes')
fig.update_layout(
    title={
        'text': 'response time v.s. number of replies',
        'x': 0.5,
        'xanchor': 'center',

    },
    xaxis_title='number of replies',
    yaxis_title='the average response time(by minutes)',
    height=600
)
# fig.update_traces(hovertemplate='county: %{label}<br>crime Rate(per 1,000 residents): %{customdata[1]:.2f}<br>population:%{customdata[2]}<br>')
fig.update_xaxes(showspikes=True, spikecolor="green", spikesnap="cursor", spikemode="across")
fig.update_yaxes(showspikes=True, spikecolor="orange", spikethickness=2)
fig.update_layout(
    hoverlabel=dict(
        bgcolor="white",
        font_size=16,
        font_family="Rockwell"
    )
)
fig.write_html('./cache/plots/bubble_plots.html')
fig.show()

def preprocess_text(text):
    if isinstance(text,float)!=True:
        text = text.lower()
        text = re.sub('[^a-zA-Z]+', ' ', text)
        return text.strip()
    else:
        return np.nan

df['reply_text'] = df['reply_text'].apply(lambda x:preprocess_text(x))
# word cloud analysis
df_top = df[df['service'].isin(top10)].copy()

from wordcloud import WordCloud, STOPWORDS
kf = df_top.query('service=="KF"')['reply_text'].tolist()
comment_kf = ''

stopwords = list(set(STOPWORDS))
stopwords.extend(['https','hi','hg','kf'])
stopwords=set(stopwords)
for i in kf:
    comment_kf = comment_kf+i+' '


wordcloud = WordCloud(width=800, height=800,
                      background_color='white',
                      stopwords=stopwords,
                      min_font_size=10).generate(comment_kf)

# plot the WordCloud image
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig('./cache/plots/wordcloud_example.png')
plt.show()

df['service'].nunique()

df_top['author_id'].nunique()

df.to_csv('./cache/output/service.csv')

# test ----------------------------
# to get the information of resonse time
# digging into the visualization plots
# def convert_days(x):
#     if isinstance(x,float)!=True:
#         days, time_str = x.split(' days ')
#         return days
#     else:
#         return -999
#
# df['days'] = df['response_time'].apply(lambda x:convert_days(x))
# df['days'] = df['days'].astype('int')
#
# tt = df.query('days >=1')
#
# tt['reply_text'][2473]
#
# m = tt[tt['reply_text'].str.contains('\^')]
#
# tt['profile_image_url'].tolist()[-1]
# m.reset_index(inplace=True)
# m.iloc[1,:]['name']
#
# m['text'].tolist()[2]
# m['profile_image_url'].tolist()[2]
# m['reply_text'].tolist()[2]
# m['name'].tolist()[2]



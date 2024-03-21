# try with using cosine similarity and word2vec instead of Bert + Unsupervised learning
import pandas as pd
import numpy as np
import gensim
import re
from gensim.models import Word2Vec
import gensim.downloader as api

glove_vectors = api.load('glove-twitter-25')

def convert_seconds(x):
    if isinstance(x,float)!=True:
        days, time_str = x.split(' days ')
        hours, minutes, seconds = map(int, time_str.split(':'))
        total_seconds = int(days) * 86400 + hours * 3600 + minutes * 60 + seconds
        return total_seconds
    else:
        return np.nan

def convert_minutes(x):
    if isinstance(x,float)!=True:
        days, time_str = x.split(' days ')
        hours, minutes, seconds = map(int, time_str.split(':'))
        total_seconds = int(days) * 1440 + hours * 60 + minutes + seconds/60
        return total_seconds
    else:
        return np.nan

# word_vectors = api.load("glove-wiki-gigaword-100")  # load pre-trained word-vectors from gensim-data
# word_vectors.n_similarity(['sushi', 'shop'], ['japanese', 'restaurant'])

# read the dataframe
df = pd.read_csv('./data/United Data 04.01.21-03.31.23/matched_data_United_variable_format.csv',low_memory=False)
print('the number of rows of unique author id:',df.author_id.nunique())

# df.drop_duplicates(subset='author_id',keep='first',inplace=True)
df['duration_seconds'] = df['response_time'].apply(lambda x:convert_seconds(x))
df['duration_minutes'] = df['duration_seconds'].apply(lambda x:x/60 if x !=[np.nan] else x)

df['name']= df['name'].replace(['', None], np.nan)
df.dropna(subset='name',inplace=True)


def preprocess_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]+', ' ', text)
    return text.strip()

df['name'] = df['name'].apply(lambda x:preprocess_text(x))
df['name']= df['name'].replace(['', None], np.nan)
df.dropna(subset='name',inplace=True)
print('the number of unique author id is {} after dropping names that are None'.format(df.author_id.nunique()))


name_list = df['name'].tolist()
# predefined two lists for cosine similarity calculation
predefined_human = ['james','will','eric','kristina']
predefined_others = ['money','trees','survey','big','small','time','travel','president','star','mrs','governor','hopeful','texas','hotdog','sir','captain','silence','cowboy']

df['similarity'] = df['name'].apply(lambda x:glove_vectors.n_similarity(predefined_human,x.split()) if x is not None else -1)
df_human = df[df['similarity']>0.5].copy()
df_human['similarity_2'] = df_human['name'].apply(lambda x:glove_vectors.n_similarity(predefined_others,x.split()) if x is not None else -1)
df_human = df_human[df_human['similarity_2']<0.7].copy()

df_human['temp'] = df_human['name'].str.split()

# drop name that has very few letters
def rule(x):
    for i in x:
        if len(i)<=2:
            return False
    return True

df_human['temp'] = df_human['temp'].apply(lambda x:rule(x))
df_human = df_human[df_human['temp']==True].copy()
df_human.reset_index(inplace=True,drop=True)
df_human.drop(columns=['temp'],inplace=True)
print('the number of unique author id of human name accounts is {} after dropping names that are None'.format(df_human.author_id.nunique()))


author_id_list = df_human.author_id.tolist()
df_others = df[~df['author_id'].isin(author_id_list)].copy()
# df_others = df_others.dropna()
df_others.reset_index(inplace=True,drop=True)
print('the number of unique author id of non-human name accounts is {} after dropping names that are None'.format(df_others.author_id.nunique()))


df_human.author_id = df_human.author_id.astype('str')
# this is the human csv we want
df_human.to_csv('./cache/output/real_names.csv')
df_others.to_csv('./cache/output/other_names.csv')




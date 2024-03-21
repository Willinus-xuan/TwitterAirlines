import pandas as pd
import numpy as np
df = pd.read_csv('../../cache/output/real_names.csv', low_memory=False, index_col=0)

df = pd.read_csv('../cache/output/results.csv',low_memory=False,index_col=0)

df_2 = pd.read_csv('../../cache/output/sentiment.csv', low_memory=False, index_col=0)

print(df.author_id.nunique())
print(df_2.author_id.nunique())
set1 = set(df.author_id)
set2 = set(df_2.author_id)
set1-set2

df = pd.read_csv('./cache/results.csv',low_memory=False,index_col=0)
def rule(x):
    if x['gender_name'] in ['mostly_male','male'] and x['gender_description']=='male' and x['gender_nltk']=='male':
        return True
    elif x['gender_name'] in ['mostly_female','female'] and x['gender_description']=='female' and x['gender_nltk']=='female':
        return True
    else:
        return False

def rule2(x):
    if x['gender_name'] in ['mostly_male','male'] and x['gender_nltk']=='male':
        return True
    elif x['gender_name'] in ['mostly_female','female'] and x['gender_nltk']=='female':
        return True
    else:
        return False

t = df[~df.apply(lambda x:rule(x),axis=1)]

tt = df[df.apply(lambda x:rule2(x),axis=1)]

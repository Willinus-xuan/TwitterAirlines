import pandas as pd
import numpy as np
import re
import gensim.downloader as api
# get the race
from ethnicolr import census_ln, pred_census_ln,pred_fl_reg_name,pred_fl_reg_name_five_cat
glove_vectors = api.load('glove-twitter-25')


def preprocess_text(text):
    if isinstance(text,float)!=True:
        text = text.lower()
        text = re.sub('[^a-zA-Z]+', ' ', text)
        return text.strip()
    else:
        return np.nan

def rules(x):
    if isinstance(x,float)==True:
        return x
    else:
        if x =='nh_white':
            return 'white'
        elif x=="asian":
            return "asian"
        elif x=='hispanic':
            return 'hispanic'
        elif x=='nh_black':
            return 'black'
def race(x):
    if x['race_x'] in ['unknown',np.nan]:
        return x['race_y']
    else:
        return x['race_x']

def convert_api(x):
    if x['last_name']=="":
        x['race'] = 'unknown'
    return x['race']

# df_others = pd.read_csv('./cache/output/other_names.csv',lineterminator='\n',low_memory=False)
# df_others.author_id.unique().__len__()

df_real = pd.read_csv('./cache/output/real_names.csv',lineterminator='\n',low_memory=False)


# derive first name and last name
df_real['name'] = df_real['name'].str.split('_').apply(lambda x:' '.join(x))
df_real['first_name'] = df_real['name'].apply(lambda x:x.split()[0])
df_real['last_name'] = df_real['name'].apply(lambda x:x.split()[-1] if len(x.split())>1 else '')


# same last name has the same race
tttt = df_real.copy()
tttt = pred_census_ln(tttt, 'last_name')[['author_id','race']]

# tttt= pred_census_ln(tttt, 'last_name','first_name',conf_int=0.9)
df_real = df_real.merge(tttt,on='author_id',how='left')
df_real['race'] = df_real.apply(lambda x:convert_api(x),axis=1)
df_real.race = df_real.race.apply(lambda x:'asian' if x=="api" else x)

t = df_real[df_real.race.isin(['unknown',np.nan])].copy() # 89234
t = pred_fl_reg_name(t,'last_name','first_name',conf_int=0.9) # 32895，48744
t = t[['author_id','race_y']]
t.race_y.unique()


df_real = df_real.merge(t,how='left',on='author_id')

df_real.race_y = df_real.race_y.apply(lambda x: rules(x))
df_real.rename(columns={'race':'race_x'},inplace=True)

df_real['race'] = df_real.apply(lambda x:race(x),axis=1)
df_real.race.value_counts(dropna=False)

# text analysis based on descriptions
df_real['description'] = df_real['description'].apply(lambda x:preprocess_text(x))
male = ['dad','male','man','husband','he','him','his','masculinist']
female = ['mom','female','woman','wife','she','her','femalist','virgin','lesbian']


# step1: to extract gender
def find_gender(x):
    if isinstance(x,float)!=True and len(x)>1:
        input = x.split()
        for i in input:
            if i in male:
                return 'male'
            elif i in female:
                return 'female'
    return np.nan


df_real['gender_description'] = df_real['description'].apply(lambda x:find_gender(x))

import gender_guesser.detector as gender
d = gender.Detector(case_sensitive=False)
# guess the gender from api
df_real['gender_name'] = df_real['first_name'].apply(lambda x:d.get_gender(x))
# df_real[['gender_description','gender_name']].drop_duplicates()

# t = df_real.query("gender_description=='female' and gender_name=='male'")

def rule(x):
    if x['gender_description'] in [np.nan]:
        return x['gender_name']
    elif x['gender_description'] ==x['gender_name']:
        return x['gender_name']
    elif x['gender_name']=='unknown' or x['gender_name']=='andy':
        return x['gender_description']
    elif x['gender_description'] =='male' and x['gender_name'] =='mostly_male':
        return 'male'
    elif x['gender_description'] == 'female' and x['gender_name'] == 'mostly_female':
        return 'female'
    else: # 如果gender_description 是male, gender_name 是female 直接删掉了
        return 'conflict'

df_real['gender'] = df_real.apply(lambda x:rule(x),axis=1)

t = df_real[df_real['gender']=='conflict']
print('the number of accounts that can\'t determine genders is {} '.format(t.author_id.nunique()))

df_real.to_csv('./cache/output/race_gender.csv')





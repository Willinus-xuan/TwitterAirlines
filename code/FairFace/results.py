import pandas as pd
import os
import numpy as np


df = pd.read_csv('./code/FairFace/predict_outputs.csv')
df['index_order'] = df['face_name_align'].apply(lambda x:x.split('/')[1]).apply(lambda x:x.split('_')[0])
df['index_order'] = df['index_order'].astype('int')
df.sort_values('index_order',inplace=True)
df.reset_index(inplace=True,drop=True)

df.rename(columns={'race':'race7','gender':'gender_model'},inplace=True)

#
df_ = pd.read_csv('./cache/output/for_DL.csv',index_col=0)
df_.reset_index(inplace=True,drop=True)
df_['index_order'] = df_.index
df_ = df_.merge(df,on='index_order',how='left')
df_.drop_duplicates(subset='author_id',keep='first',inplace=True)
df_.race7 = df_.race7.apply(lambda x:x.lower() if x not in [np.nan] else x)
df_.race4 = df_.race4.apply(lambda x:x.lower() if x not in [np.nan] else x)
df_.gender_model = df_.gender_model.apply(lambda x:x.lower() if x not in [np.nan] else x)

df_.race.value_counts(dropna=False)
df_.race7.value_counts(dropna=False)
df_.race4.value_counts(dropna=False)


def convert7(x):
    if 'asian' in x:
        return 'asian'
    elif 'hispanic' in x:
        return 'hispanic'
    else:
        return x
df_.race7 = df_.race7.apply(lambda x:convert7(x) if x not in [np.nan] else x)
df_.race7.value_counts(dropna=False)

# create a confidence column
def confidence(col,threshold=0.7):
    race_list = []
    for y in df_.loc[:, col].tolist():
        if isinstance(y, str) == True:
            x = [float(x) for x in y.strip('[]').split()]
            output = 'less confident'
            for i in x:
                if i > threshold:
                    output = 'confident'
            race_list.append(output)
        else:
            race_list.append(np.nan)
    return race_list


df_['confident_level_4'] = confidence('race_scores_fair_4')
df_['confident_level_7'] = confidence('race_scores_fair')

df_.race7.unique()
df_.race4.unique()
df_.race.unique()

# inspired by bagging, vote majority
def majority(x):
    vote = {}
    vote[x['race']] = vote.get(x['race'], 0) + 1
    vote[x['race4']] = vote.get(x['race4'], 0) + 1
    vote[x['race7']] = vote.get(x['race7'], 0) + 1
    vote_sort = dict(sorted(vote.items(), key=lambda item: item[1], reverse=True))
    if list(vote_sort.values())[0] != 1:
        major = list(vote_sort.keys())[0]
    else:
        major = 'non-white'
    return major

# def race(x): an another method not vote
#     if x['confident_level_4']=='confident' and x['confident_level_7']=="confident": # 1558
#         return x['race4']
#     elif x['confident_level_4']=='confident' and x['confident_level_7']=="less confident": # 851
#         return x['race4']
#     elif x['confident_level_4']=='less confident' and x['confident_level_7']=="confident": # 156
#         return x['race7']
#     elif x['confident_level_4']=='less confident' and x['confident_level_7']=="less confident": # 328
#         vote = {}
#         vote[x['race']] = vote.get(x['race'], 0) + 1
#         vote[x['race4']] = vote.get(x['race4'],0) + 1
#         vote[x['race7']] = vote.get(x['race7'], 0) + 1
#         vote_sort = dict(sorted(vote.items(), key=lambda item: item[1], reverse=True))
#         if list(vote_sort.values())[0]!=1:
#             major = list(vote_sort.keys())[0]
#         else:
#             major = 'non-white'
#         return major

# step 1. race
df_['race_model'] = df_.apply(lambda x: majority(x),axis=1)
df_['race_model'].value_counts(dropna=False)
df_['race'].value_counts(dropna=False)
# g = df_[df_['race']!=df_['race_model']][['race','race_model','name','face_name_align']]
#
# gg = g[~g.race_model.isin([np.nan])] # 948
def modified_race(x):
    if x['race_model'] not in [np.nan]:
        return x['race_model']
    else:
        return x['race']

df_['final_race'] = df_.apply(lambda x:modified_race(x),axis=1)

t = df_[['race','race_model','name','face_name_align']]

# step 2. gender

df_.gender_model.value_counts(dropna=False)
df_['confident_level_gender'] = confidence('gender_scores_fair',threshold=0.9)


t = df_[~df_.loc[:,'gender_model'].isin([np.nan])][['gender','gender_model','name','face_name_align','confident_level_gender']].copy()

tt = t[t['gender_model']!=t['gender']]
tt = tt[tt.loc[:,'gender'].isin(['female','male'])]

confident = tt.query('confident_level_gender=="confident"')
less_confident = tt.query('confident_level_gender=="less confident"')

g = t[t['gender_model']==t['gender']] # 2260/2887

def convert_gender(x):
    if x['gender']=="mostly_female" :
        if x['gender_model']=="female":
            return x['gender_model']
        elif x['gender_model'] in [np.nan]:
            return 'female'
        else:
            if x['confident_level_gender']=='confidence':
                return x['gender_model']
            else:
                return 'undetermined'
    elif x['gender']=="mostly_male":
        if x['gender_model']=="male":
            return x['gender_model']
        elif x['gender_model'] in [np.nan]:
            return 'male'
        else:
            if x['confident_level_gender']=='confidence':
                return x['gender_model']
            else:
                return 'undetermined'
    elif x['gender'] =="undetermined":
        return x['gender_model']
    else:
        return x['gender']

df_['final_gender'] = df_.apply(lambda x:convert_gender(x),axis=1)
df_['final_gender'] = df_['final_gender'].fillna('undetermined')
df_['final_gender'].value_counts(dropna=False)

df_.to_csv('./cache/output/results(top10).csv')

t = df_[df_.loc[:,'final_gender'].isin([None])][['gender','gender_model','face_name_align','confident_level_gender']]
df_.columns
df_output = df_.drop(columns=['similarity',
       'similarity_2','race_x', 'race_y', 'race','gender_description', 'gender_name','gender_nltk','gender','index_order','face_name_align',
       'race7', 'race4', 'gender_model','race_scores_fair',
       'race_scores_fair_4', 'gender_scores_fair', 'age_scores_fair',
       'confident_level_4', 'confident_level_7', 'race_model','confident_level_gender']).copy()

df_output.final_race.value_counts(dropna=False)
df_output.final_gender.value_counts(dropna=False)
df_output.rename(columns={'final_race':'race','final_gender':
                          'gender'},inplace=True)


df_output.to_csv('./cache/output/clean_results(top10).csv')

df = pd.read_csv('./cache/output/clean_results(top10).csv',index_col=0)

t = df_[['race_model','race']]
tt = t[~t.race_model.isin([np.nan])]
g = tt[tt['race_model']!=tt['race']]

df_['race'].value_counts(dropna=False)
df_['race_model'].value_counts(dropna=False)
df_['final_race'].value_counts(dropna=False)

# it sould be noticed that there are a lot of white may not be white
m = df_.query('race=="white" and race_model!="white"')
m = m[~m.race_model.isin([np.nan])][['race_model','race']]
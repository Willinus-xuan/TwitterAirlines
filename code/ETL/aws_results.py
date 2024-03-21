import pandas as pd
import os
import numpy as np

df = pd.read_csv('./cache/predictions.csv')
df_model = pd.read_csv('./cache/output/results(top10).csv',index_col=0)


df_model['name'] = df_model['face_name_align'].apply(lambda x:x.split('/')[-1].split('_')[0] +'.png' if x not in [np.nan] else x)


t = df_model[['name','gender','gender_model','final_gender']]

img_data = []
for img in df.name.unique():
    df_temp = df.query('name==@img').copy()
    for label in df_temp.label.tolist():
        if label in ['Male','Female']:
            confidence = round(df_temp.query('label==@label')['confidence'].values[0],2)
            img_data.append([img,label,confidence])
            break
        if label == df_temp.label.tolist()[-1] and label not in ['Male','Female']:
            img_data.append([img,np.nan,np.nan])


img_data = sorted(img_data,key=lambda x:int(x[0].split('.')[0]))
df_aws = pd.DataFrame(columns=['name','gender_aws','confidence'],data = img_data)

tt = t[~t.name.isin([np.nan])]

df_aws = df_aws.merge(tt,on='name',how='left')

df_aws['gender_aws'] = df_aws['gender_aws'].apply(lambda x: 'male' if x == "Male" else 'female' if x == "Female" else x)
m = df_aws[df_aws['gender_aws']!=df_aws['gender_model']]
m = m.dropna(how='any')









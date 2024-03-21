import pandas as pd
import os
import pathlib as Path




path = '/'.join(os.getcwd().split('/')) + '/data/images'

path_list = os.listdir(path)
path_list.remove('.DS_Store')
path_list = sorted(path_list, key=lambda x: int(x.split('.')[0]))

path_list = [path + '/' + i for i in path_list]
df = pd.DataFrame()
df['img_path'] = path_list

df.to_csv('./code/FairFace/predict_imgs.csv',index=False)



df = pd.DataFrame()

df['img_path'] = path_list[:20]
df.to_csv('./code/FairFace/test_imgs.csv',index=False)


df_ = pd.read_csv('./code/FairFace/test_outputs2.csv')
df_['index_order'] = df_['face_name_align'].apply(lambda x:x.split('/')[1]).apply(lambda x:x.split('_')[0])
df_ = df_.query('index_order!="race"').copy()
df_['index_order'] = df_['index_order'].astype('int')

df_.sort_values('index_order',inplace=True)
df_.reset_index(inplace=True,drop=True)







# ip代理的知识
# 透明代理，低匿代理，高匿代理


import requests
import pandas as pd

df = pd.read_csv('./cache/output/service.csv',index_col=0,low_memory=False)
df.gender.value_counts(dropna=False)
t = df.query('service !="unknown"').copy()
top10 = list(t.service.value_counts().iloc[:10].index)
df_top = df[df['service'].isin(top10)].copy()
df_top.drop_duplicates(subset='author_id',keep='first',inplace=True)

df_top.to_csv('./cache/output/for_DL.csv')

image_url_list = df_top.profile_image_url.tolist()

for cnt,url in enumerate(image_url_list):
    r = requests.get(url)
    if r.status_code ==200:
        img_data = r.content
        with open('./data/images/'+str(cnt)+'.png','wb') as fp:
            fp.write(img_data)
    else:
        print('failed access',r.status_code)
        print(cnt)
        continue


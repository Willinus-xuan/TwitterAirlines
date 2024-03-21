import pandas as pd
import re
import numpy as np
import nltk


df = pd.read_csv('./cache/output/race_gender.csv',index_col=0,low_memory=False)
# df = df[~df['response_time'].isna()].copy()

def preprocess_text(text):
    if isinstance(text,float)!=True:
        # replace @... to None
        text = text.lower()
        pattern = r'@\w+'
        text = re.sub(pattern, '', text)
        text = re.sub('[^a-zA-Z]+', ' ', text)
        # text = re.sub('united','',text)
        return text.strip()
    else:
        return np.nan

df['text'] = df['text'].apply(lambda x: preprocess_text(x))
from transformers import pipeline

# Set up the inference pipeline using a model from the 🤗 Hub
sentiment_analysis = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")

# Let's run the sentiment analysis on each text
sentiments = []
for text in df['text']:
    try:
        sentiment = sentiment_analysis(text)
        sentiments.append(sentiment[0]['label'])

    except:
        pass

df['sentiment'] = sentiments
df['sentiment'].value_counts(dropna=False)
df.to_csv("./cache/output/sentiment.csv")







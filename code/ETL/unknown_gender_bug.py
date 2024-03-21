import random

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import names
# data visualization
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('./cache/output/sentiment.csv',index_col=0,low_memory=False)


others = pd.read_csv('./cache/output/other_names.csv',index_col=0,lineterminator='\n',low_memory=False)
# real_name = pd.read_csv('./cache/real_names.csv')
t = df[df['gender_name']=='unknown'].copy()
tt = t[(t['similarity_2']>0.65) | (t['similarity']<0.55)]
m = t[(t['similarity_2']>0.65) | (t['similarity']<0.55)].author_id.tolist()
df = df[~df['author_id'].isin(m)].copy()
print('drop non-human name again from human name list and we get {} unique accounts named by human'.format(df.author_id.nunique()))


# nltk names
def gender_features(word):
    """ feature extractor for the name classifier
    The feature evaluated here is the last letter of a name
    feature name - "last_letter"
    """
    feature = {}
    feature['last_letter'] = word[-1].lower()
    feature['suffix2'] =  word[-2:]
    feature['suffix3'] = word[-3:]
    return feature  # feature set

def demo():
    # Extract the data sets
    labeled_names = ([(name, "male") for name in names.words("male.txt")] +
                     [(name, "female") for name in names.words("female.txt")])

    # print(len(labeled_names))  # 7944 names

    # Shuffle the names in the list
    random.shuffle(labeled_names)

    # Process the names through feature extractor
    feature_sets = [(gender_features(n), gender)
                    for (n, gender) in labeled_names]

    # Divide the feature sets into training and test sets
    train_set, test_set = feature_sets[500:], feature_sets[:500]

    # Train the naiveBayes classifier
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print(classifier.show_most_informative_features(5))
    print(nltk.classify.accuracy(classifier, test_set))
    return classifier

# print(classifier.classify(gender_features("neo")))
classifier = demo()

# evaluate
df['gender_nltk'] = df['first_name'].apply(lambda x:classifier.classify(gender_features(x)) if x not in [np.nan,''] else 'unknown')

test = df.copy()
def rule(x):
    if x =="mostly_female":
        return 'female'
    elif x=="mostly_male":
        return "male"
    else:
        return x

test['gender'] = test['gender'].apply(lambda x: rule(x))
test = test[~test['gender'].isin(['andy','unknown','conflict'])].copy()
tt = test[test['gender'] ==test['gender_nltk']]
print("accuracy_rate:",len(tt)/len(test))


# apply
def gender(x):
    if x['gender'] in ['unknown','andy']:
        return x['gender_nltk']
    elif x['gender']=='mostly_female' and x['gender_nltk'] =='female':
        return 'female'
    elif x['gender'] =='mostly_male' and x['gender_nltk'] =='male':
        return 'male'
    elif x['gender']=='conflict':
        return 'undetermined'
    else:
        return x['gender'] # rely more on gender name and gender_description than nltk

df['final_gender'] = df.apply(lambda x:gender(x),axis=1)

df['race'] = df['race'].fillna('unknown')
g = sns.countplot(df,x='race')
g.set_title('Race Distribution of United Airlines')
plt.savefig('./cache/plots/race_distribution.png')


g = sns.countplot(df,x='final_gender')
g.set_title('Gender Distribution of United Airlines')
plt.savefig('./cache/plots/gender_distribution.png')


df.drop(columns=['Unnamed: 0','gender'],inplace=True)
df = df.rename({'final_gender':'gender'},axis=1)

df.to_csv('./cache/output/clean_gender_race.csv')




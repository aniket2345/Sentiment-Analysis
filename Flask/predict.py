from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
import pickle
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


def vectorize(text) :
    text_series = pd.Series(text)
    X = vectorizer.transform(text_series)
    return X

def word_processing(sentence):

    sentence = sentence.lower()
    sentence = re.sub(r'@[A-Za-z0-9]+','',sentence)
    sentence = re.sub(r'#','',sentence)
    sentence = re.sub(r'RT[\s]+','',sentence)
    sentence = re.sub(r'https?:\/\/\S+','',sentence)

    sentence = sentence.translate(str.maketrans("","",string.punctuation))
    sentence_tokens = word_tokenize(sentence)
    filtered = [word for word in sentence_tokens if word not in stop_words]
    """
    ps = PorterStemmer()
    sw = [ps.stem(w) for w in filtered]

    l = WordNetLemmatizer()
    l_words = [l.lemmatize(w, pos='a') for w in sw]"""
    return " ".join(filtered)


classifier = open("svm.pickle","rb")
clf = pickle.load(classifier)

vectorizer = TfidfVectorizer(min_df = 5, max_df = 0.8, sublinear_tf = True, use_idf = True)
                             
trainData = pd.read_csv("https://raw.githubusercontent.com/Vasistareddy/sentiment_analysis/master/data/train.csv")

fitvector = vectorizer.fit_transform(trainData['Content'])


apple_data = pd.read_csv("apple_data.csv",encoding="latin-1")
print(apple_data.columns)
sentiment=[]
counter=0
appender = 0
for index,row in apple_data.iterrows() :
    text = []
    text.append(word_processing(row["text"]))
    X = vectorize(text)
    prediction = clf.predict(X)
    
    if counter<4:
        if prediction=='pos':
            appender += 1
        else:
            appender -= 1
    else :
        if prediction=='pos':
            appender += 1
        else:
            appender -= 1
        sentiment.append(appender)
        counter=0
        continue
    counter+=1

sentiment_df = pd.DataFrame(data=sentiment,columns=["sentiment"],index=None)

print(sentiment_df.shape)

trainData['sentiment']=sentiment_df
print(trainData.shape)

def random_dates2(start, end, n, unit='D', seed=None):
    if not seed:  # from piR's answer
        np.random.seed(0)

    ndays = (end - start).days + 1
    return start + pd.to_timedelta(
        np.random.randint(0, ndays, n), unit=unit
    )

dates = random_dates2(pd.to_datetime('2015-01-01'),pd.to_datetime('2018-01-01'),450)

print(type(dates))


date_df = pd.DataFrame(data=dates,columns=["date"],index=None)
date_df = date_df.sort_values(by=["date"])

trainData['Date'] = date_df
trainData = trainData.set_index('Date')

print(trainData[:500])

final = trainData
final = final.drop(['Content','Label'],axis=1)
print(final)
plt.bar(final.index,final['sentiment'])
plt.show()

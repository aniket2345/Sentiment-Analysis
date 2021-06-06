import praw
import pickle
from datetime import datetime
import pandas as pd
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import os

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

classifier = open("../flask/svm.pickle","rb")
clf = pickle.load(classifier)

vectorizer = TfidfVectorizer(min_df = 5, max_df = 0.8, sublinear_tf = True, use_idf = True)
                             
trainData = pd.read_csv("https://raw.githubusercontent.com/Vasistareddy/sentiment_analysis/master/data/train.csv")

fitvector = vectorizer.fit_transform(trainData['Content'])


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



client_id = "AGJh9efV3xi-wQ"
client_secret = "IoEbma35g1-2-XKwnTumZuD4Iu3Scw"
user_agent = "SEProj"
username = "abillionasians"
password = "Shubhamjazz1"

reddit = praw.Reddit(client_id = "AGJh9efV3xi-wQ",client_secret = "IoEbma35g1-2-XKwnTumZuD4Iu3Scw",user_agent = "SEProj",username = "abillionasians",password = "Shubhamjazz1")

all = "iPad+iPhone+Mac+AppleTV+tvPlus+AppleHelp+AirPods+AppleWatch+AppleCard+AppleMusic+AirTags+VintageApple+AppleSwap+AppleBandMarket+macOS+OSX+Macbook+MacbookPro+MacBookAir+macapps+MacGaming+Hackintosh+BootCamp+MacSetups+iOS+HomeKit+Shortcuts+iPhoneography+iOSGaming+iOSsetups+apphookup+CarPlay+iphonewallpapers+ipadwallpapers+applewatchwallpapers+iOSBeta+iOSXBeta+tvOSBeta+watchOSBeta"


def sentiment():

    data = pd.DataFrame(columns=['Date','Content','Author'])

    print(dir(reddit.subreddit("apple")))

    #list_of_all=[]


    lis_fram_date=[]
    lis_fram_content=[]
    lis_fram_author=[]

    counter=0
    #for submission in reddit.subreddit(all).stream.submissions():
    for submission in reddit.subreddit(all).new(limit=10) :

        subdate = str(datetime.fromtimestamp(submission.created_utc))
        subcontent = submission.title
        subauthor = submission.author

        counter+=1
        print(counter)
        
        #temp=[]
        #temp.append([subdate,subcontent,subauthor])
        
        lis_fram_date.append(subdate)
        lis_fram_content.append(subcontent)
        lis_fram_author.append(subauthor)
        for comment in submission.comments.list() :
            try:
            
                comdate = str(datetime.fromtimestamp(comment.created_utc))
                comcontent = comment.body
                comauthor = comment.author
                
                #temp.append([comdate,comcontent,comauthor])
                
                lis_fram_date.append(comdate)
                lis_fram_content.append(comcontent)
                lis_fram_author.append(comauthor)

            except Exception as e:
              print("An exception occurred: ", e)
        #list_of_all.append(temp)


    data['Date']=lis_fram_date
    data['Content']=lis_fram_content
    data['Author']=lis_fram_author

    data = data.set_index('Date')

    data['full_time'] = data.index
    data['second'] = pd.to_datetime(data.index).second
    data['hour'] = pd.to_datetime(data.index).hour
    data['minute'] = pd.to_datetime(data.index).minute

    ###################WORDCLOUD##################



    file = open("wordcloud_apple.pickle","wb")
    pickle.dump(data,file)

    """
    for index,row in data_hour.iterrows():
        text=[]
        text.append(word_processing(row["Content"]))
        X = vectorize(text)
        prediction = clf.predict(X)
        data_hour.loc[index,'Sentiment'] = prediction

    print(data_hour)"""
    #########################################SECONDS########################################

    data_second = data
    #data_second = data_second.set_index(data_second['second'])
    data_second = data_second.set_index(['hour','minute','second'])
    data_second = data_second.sort_index()

    first = True
    second_check = 0
    second_counter = 0
    hour_check=0
    minute_check=0
    final_second = pd.DataFrame(columns=['Minutes_Seconds','Sentiment'])

    second_average_sentiment = 0

    Minutes_Seconds = []
    Sentiment = []



    for index,row in data_second.iterrows():
        if first :
            hour_check = index[0]
            minute_check = index[1]
            second_check = index[2]
            first = False
        
        if second_check==index[2] :
            blob = TextBlob(row["Content"])
            print(blob.sentiment)
            prediction=blob.sentiment[0]
            #data_second.loc[index[1],'Sentiment'] = prediction
            if prediction=="pos":
                second_average_sentiment += prediction
            else :
                second_average_sentiment -= prediction
            second_counter+=1
        else :
            second_average_sentiment = second_average_sentiment/second_counter
            
            #Minutes_Seconds.append(str(hour_check)+"H-"+str(minute_check)+"M-"+str(second_check)+"S")
            Minutes_Seconds.append(str(minute_check)+"M-"+str(second_check)+"S")
            Sentiment.append(second_average_sentiment)
            
            second_average_sentiment=0
            second_counter=0
            second_check=index[2]
            minute_check=index[1]
            hour_check=index[0]
            
            blob = TextBlob(row["Content"])
            print(blob.sentiment)
            prediction=blob.sentiment[0]
            #data_second.loc[index,'Sentiment'] = prediction
            if prediction=="pos":
                second_average_sentiment += prediction
            else :
                second_average_sentiment -= prediction
            second_counter+=1

            
    final_second['Minutes_Seconds'] = Minutes_Seconds
    final_second['Sentiment'] = Sentiment

    print(final_second)

    file = open("final_second_apple.pickle","wb")
    pickle.dump(final_second,file)

    #########################################MINUTE########################################


    data_minute = data
    #data_minute = data_minute.set_index(data_minute['minute'])
    data_minute = data_minute.set_index(['hour','minute'])
    data_minute = data_minute.sort_index()

    first = True
    minute_check = 0
    minute_counter = 0
    hour_check=0
    final_minute = pd.DataFrame(columns=['Hour_Minutes','Sentiment'])

    minute_average_sentiment = 0

    Hour_Minutes = []
    Sentiment = []

    for index,row in data_minute.iterrows():
        if first :
            hour_check = index[0]
            minute_check = index[1]
            first = False
        
        if minute_check==index[1] :
            blob = TextBlob(row["Content"])
            print(blob.sentiment)
            prediction=blob.sentiment[0]
            #data_minute.loc[index[1],'Sentiment'] = prediction
            if prediction=="pos":
                minute_average_sentiment += prediction
            else :
                minute_average_sentiment -= prediction
            minute_counter+=1
        else :
            minute_average_sentiment = minute_average_sentiment/minute_counter
            
            Hour_Minutes.append(str(hour_check)+"H-"+str(minute_check)+"M")
            Sentiment.append(minute_average_sentiment)
            
            minute_average_sentiment=0
            minute_counter=0
            minute_check=index[1]
            hour_check=index[0]
            
            blob = TextBlob(row["Content"])
            print(blob.sentiment)
            prediction=blob.sentiment[0]
            #data_minute.loc[index,'Sentiment'] = prediction
            if prediction=="pos":
                minute_average_sentiment += prediction
            else :
                minute_average_sentiment -= prediction
            minute_counter+=1

            
    final_minute['Hour_Minutes'] = Hour_Minutes
    final_minute['Sentiment'] = Sentiment

    print("SENT : ",final_minute['Sentiment'].mean())

    file = open("final_minute_apple.pickle","wb")
    pickle.dump(final_minute,file)

    #########################################HOURS########################################
            
    """final_hour = pd.DataFrame(columns=['Hour','Sentiment'])

    data_hour = data
    #data_hour = data_hour.set_index(data_hour['hour'])
    data_hour = data_hour.set_index('hour')
    data_hour = data_hour.sort_index()
    data_hour['Sentiment'] = data_hour['Content']
    print(data_hour)
            
    first = True
    hour_average_sentiment = 0
    Hours = []
    Hour_Sentiment = []
    hour_check = 0
    hour_counter = 0

    for index,row in data_hour.iterrows():
        if first :
            hour_check = index
            first = False

        if hour_check==index :
            text=[]
            text.append(word_processing(row["Content"]))
            X = vectorize(text)
            prediction = clf.predict(X)
            if prediction=="pos":
                hour_average_sentiment += 1
            else :
                hour_average_sentiment -= 1
            hour_counter+=1
        else :
            hour_average_sentiment = hour_average_sentiment/hour_counter
            
            Hours.append(str(hour_check)+"H")
            Hour_Sentiment.append(hour_average_sentiment)
            
            hour_average_sentiment=0
            hour_counter=0
            hour_check=index
            
            text=[]
            text.append(word_processing(row["Content"]))
            X = vectorize(text)
            prediction = clf.predict(X)
            #data_minute.loc[index,'Sentiment'] = prediction
            if prediction=="pos":
                hour_average_sentiment += 1
            else :
                hour_average_sentiment -= 1
            hour_counter+=1
            

    final_hour['Hour'] = Hours
    final_hour['Sentiment'] = Hour_Sentiment

    print(final_hour)

    file = open("../flask/final_hour.pickle","wb")
    pickle.dump(final_hour,file)"""
    
    

sentiment()

#print(data_hour)


#red_obj = reddit.subreddit("iPad+iPhone+Mac+AppleTV+tvPlus+AppleHelp+AirPods+AppleWatch+AppleCard+AppleMusic+AirTags+VintageApple+AppleSwap+AppleBandMarket+macOS+OSX+Macbook+MacbookPro+MacBookAir+macapps+MacGaming+Hackintosh+BootCamp+MacSetups+iOS+HomeKit+Shortcuts+iPhoneography+iOSGaming+iOSsetups+apphookup+CarPlay+iphonewallpapers+ipadwallpapers+applewatchwallpapers+iOSBeta+iOSXBeta+tvOSBeta+watchOSBeta")

"""


red_obj = reddit.subreddit("apple")



new = red_obj.new(limit=10)
counter=0



for i in new :
    print("NEW TITLE \n",i.title)
    submission = reddit.submission(id=i)
    for j in submission.comments.list() :
        print("NEW COMMENT \n")
        print(j.body)


print("done")

"""

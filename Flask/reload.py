from datetime import datetime

from flask import Flask
from werkzeug.serving import run_simple


from flask import Flask, request, render_template, send_file
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
import pickle
import pandas as pd
import numpy as np
import re
import os
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import time
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from collections import Counter
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

import seaborn as sns
import matplotlib.pyplot as plt
import io

import matplotlib
matplotlib.use('Agg')




nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

nltk.download('punkt')
nltk.download('wordnet')
stop = set(STOPWORDS)
stopwords = set(STOPWORDS)




# set to True to inform that the app needs to be re-created
to_reload = False


def get_app():
    print("create app now")
    app = Flask(__name__)
    

    ams_bool_list = []

    def preprocess_news(df):
        corpus=[]
        title_lem=WordNetLemmatizer()
        for title in df :
            TC=[w for w in word_tokenize(title) if (w not in stop)]
            TC=[title_lem.lemmatize(w) for w in TC if len(w)>2]
            corpus.append(TC)
        return corpus
        

    def show_wordcloud(dataset, name):
        WC = WordCloud(background_color='black',stopwords=stopwords,max_words=100,max_font_size=30,scale=3,random_state=1)
        WC=WC.generate(str(dataset))
        image = plt.imshow(WC)
        save_here = "static/pics/"+name
        plt.savefig(save_here)



    def meanfunc(df,open):
        average_minute_sentiment=0
        for index,row in df.iterrows():
            average_minute_sentiment+=row['Sentiment']
        average_minute_sentiment = average_minute_sentiment/len(open)
        average_minute_sentiment = float(str((average_minute_sentiment+100)/2)[:5])
        return average_minute_sentiment


    def does_all(name):

        ### MINUTE AND SECOND
        open_minute = "final_minute_"+name+".pickle"
        open_second = "final_second_"+name+".pickle"
        
        file = open(open_minute,"rb")
        final_minute = pickle.load(file)
        labels_minute = [i for i in final_minute['Hour_Minutes'].values] ###########################################################
        values_minute = [i for i in final_minute['Sentiment'].values] ###########################################################
            
        average_minute_sentiment = meanfunc(final_minute,open_minute)###########################################################
        #print("AVERAGE MINUTE : ", average_minute_sentiment)
        """ams_bool=""

        if average_minute_sentiment>50 :
            ams_bool="POSITIVE"###########################################################
        elif average_minute_sentiment==50:
            ams_bool="NEUTRAL"
        else :
            ams_bool="NEGATIVE
            
        ams_bool_list.append(ams_bool)"""


        file = open(open_second,"rb")
        final_second = pickle.load(file)
        labels_second = [i for i in final_second['Minutes_Seconds'].values]###########################################################
        values_second = [i for i in final_second['Sentiment'].values]###########################################################

        average_second_sentiment = meanfunc(final_second,open_second)###########################################################
        #print("AVERAGE SECOND : ", average_second_sentiment)


        ##### WORDCLOUD #####

        open_wordcloud = "wordcloud_"+name+".pickle"
        save_wordcloud = "wordcloud_"+name+".jpg"
        
        file = open(open_wordcloud,"rb")########################################################### will need to change name here specific name saving required
        wordcloud = pickle.load(file)
        corpus=preprocess_news(wordcloud['Content'])
        #print("WORDCLODU FEWAF H : ",corpus)
        show_wordcloud(corpus,save_wordcloud)

        picfolder = os.path.join('static','pics')
        app.config['UPLOAD_FOLDER'] = picfolder

        ###### COMMENTS #####

        comments = wordcloud['Content'].values[:9] ###########################################################

        all_list=[labels_minute,values_minute,labels_second,values_second,average_minute_sentiment,average_second_sentiment,comments]
        
        return all_list


    def ams(name) :
        open_minute = "final_minute_"+name+".pickle"
        file = open(open_minute,"rb")
        final_minute = pickle.load(file)
        labels_minute = [i for i in final_minute['Hour_Minutes'].values] ###########################################################
        values_minute = [i for i in final_minute['Sentiment'].values] ###########################################################
           
        average_minute_sentiment = meanfunc(final_minute,open_minute)###########################################################
        #print("AVERAGE MINUTE : ", average_minute_sentiment)
        ams_bool=""

        if average_minute_sentiment>50 :
            ams_bool="POSITIVE"###########################################################
        elif average_minute_sentiment==50:
            ams_bool="NEUTRAL"
        else :
            ams_bool="NEGATIVE"
           
        ams_bool_list.append(ams_bool)
        
    def heatmap():
        file = open("final_minute_apple.pickle","rb")
        minute_apple = pickle.load(file)
        
        file = open("final_minute_amazon.pickle","rb")
        minute_amazon = pickle.load(file)
        
        file = open("final_minute_tesla.pickle","rb")
        minute_tesla = pickle.load(file)
        
        correlation_df = pd.DataFrame(columns=['Apple','Amazon','Tesla'])
        
        correlation_df['Apple'] = minute_apple['Sentiment']
        correlation_df['Tesla'] = minute_tesla['Sentiment']
        correlation_df['Amazon'] = minute_amazon['Sentiment']
        
        print(correlation_df)
        
        sns.heatmap(correlation_df.corr())
        plt.show()
        
        save_here = "static/pics/heatmap.png"
        plt.savefig(save_here)
        """
        bytes_image = io.BytesIO()
        plt.savefig(bytes_image, format='png')
        bytes_image.seek(0)
        return bytes_image"""





    ams("apple")
    ams("tesla")
    ams("amazon")

    # to make sure of the new app instance
    now = datetime.now()

    
    @app.route('/')
    def home_page():
        print(ams_bool_list)
        return render_template('home_page.html',ams=ams_bool_list)
        
    """
    @app.route('/MATRIX', methods=['GET'])
    def correlation_matrix():
        bytes_obj = heatmap()
        
        return send_file(bytes_obj,
                         attachment_filename='plot.png',
                         mimetype='image/png')
    """

    @app.route('/APPLE')
    def APPLE():
        
        wrapper = does_all("apple")
        word_pic = os.path.join(app.config['UPLOAD_FOLDER'],'wordcloud_apple.jpg') ###########################################################
        
        return render_template('APPLE.html', labels_minute=wrapper[0],
                                values_minute=wrapper[1],
                                labels_second = wrapper[2],
                                values_second = wrapper[3],
                                wordcloud = word_pic,
                                ams=wrapper[4],
                                ass=wrapper[5],
                                reviews = wrapper[6]
                                )

    @app.route('/TESLA')
    def TESLA():
        wrapper = does_all("tesla")
        word_pic = os.path.join(app.config['UPLOAD_FOLDER'],'wordcloud_tesla.jpg') ###########################################################
        return render_template('TESLA.html', labels_minute=wrapper[0],
                                values_minute=wrapper[1],
                                labels_second = wrapper[2],
                                values_second = wrapper[3],
                                wordcloud = word_pic,
                                ams=wrapper[4],
                                ass=wrapper[5],
                                reviews = wrapper[6]
                                )
                                
    @app.route('/AMAZON')
    def AMAZON():
        wrapper = does_all("amazon")
        word_pic = os.path.join(app.config['UPLOAD_FOLDER'],'wordcloud_amazon.jpg') ###########################################################
        return render_template('AMAZON.html', labels_minute=wrapper[0],
                                values_minute=wrapper[1],
                                labels_second = wrapper[2],
                                values_second = wrapper[3],
                                wordcloud = word_pic,
                                ams=wrapper[4],
                                ass=wrapper[5],
                                reviews = wrapper[6]
                                )

    @app.route('/reload')
    def reload():
        global to_reload
        to_reload = True
        return "reloaded"

    return app


class AppReloader(object):
    def __init__(self, create_app):
        self.create_app = create_app
        self.app = create_app()

    def get_application(self):
        global to_reload
        if to_reload:
            self.app = self.create_app()
            to_reload = False

        return self.app

    def __call__(self, environ, start_response):
        app = self.get_application()
        return app(environ, start_response)


# This application object can be used in any WSGI server
# for example in gunicorn, you can run "gunicorn app"
application = AppReloader(get_app)

if __name__ == '__main__':
    run_simple('localhost', 5000, application,
               use_reloader=True, use_debugger=True, use_evalex=True)

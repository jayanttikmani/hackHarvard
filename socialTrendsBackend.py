
# coding: utf-8

# In[10]:

import sys
import warnings
warnings.filterwarnings('ignore')
def socialMediaAnalytics(keyword):
    import tweepy
    from textblob import TextBlob
    
    from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
    import plotly.plotly as py
    import plotly.figure_factory as ff
    import plotly.graph_objs as go
    init_notebook_mode(connected=True)
    
    from IPython.display import display, Markdown
    def printmd(string):
        display(Markdown(string))
    
    printmd('**---------------------------------------------------------------- &nbsp; F &nbsp; A &nbsp; C &nbsp; E &nbsp; B &nbsp; O &nbsp; O &nbsp; K &nbsp; &nbsp; &nbsp; A &nbsp; N &nbsp; A &nbsp; L &nbsp; Y &nbsp; T &nbsp; I &nbsp; C &nbsp; S &nbsp; ----------------------------------------------------------------**')
    
    
    import requests
    import time
    import pickle
    import random
    import pandas as pd
    import string
    import json
    from operator import itemgetter
    token = "EAACEdEose0cBAEfKr2OnCQdJ2kf17zGNrnZAYdS2GX0fKxSLXn3vN2XDQpwCNLiis2GoufWLBhiu0QpRZBZAvSlMyx09NnEpKM1cJU8sk5lKez4E2OsqjWxeMA2t0Ne09lFBOTMHFkQubVfDC3Ly7yjir4iFymDiXGc1sgcLmRTrY4q3MDu32INjWjIn3fie8EoaxgsVgZDZD"
    req = keyword+"?fields=posts.limit(5){message,created_time,likes.limit(0).summary(true),comments.limit(0).summary(true)}"
    
    r = requests.get("https://graph.facebook.com/v2.10/" + req , {'access_token' : token})
    
    
    results=r.json()

    data = []
    results=results['posts']
    i=0
    while True:

        try:
            time.sleep(random.randint(2,5))
            data.extend(results['data'])
            r=requests.get(results['paging']['next'])
            results=r.json()
            i+=1

            if i>2:
                break
        except:
            print ("done")
            break

    pickle.dump(data,open("steam_data.pkl","wb"))
    loaded_data=pickle.load(file=open("steam_data.pkl" ,"rb"))
    
    import collections
    df_test = pd.DataFrame(columns=['message','date','likes','comments'])
    def flatten(d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, collections.MutableMapping):
                items.extend(flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    for each in loaded_data:
        #d=dict(each)
        d1 = flatten(each)
        s1 = json.dumps(d1)
        d2 = json.loads(s1)
        the_dict = json.loads(s1)
        #print (the_dict)
        df = pd.io.json.json_normalize(data=the_dict)
        if 'message' in df.columns.values:
            df = df[['message','created_time','likes_summary_total_count','comments_summary_total_count']]
            df_test=df_test.append(df)

    postSentScore = []
    postSent = []
    for comment in list(df_test['message']):
        commentAnalysis = TextBlob(comment)
        postSentScore.append(commentAnalysis.sentiment[0])
        if (commentAnalysis.sentiment[0]>0):
            postSent.append('Positive')
        elif (commentAnalysis.sentiment[0]<0):
            postSent.append('Negative')
        else:
            postSent.append('Neutral')
        
    #creating a dataframe for @ for easy accessibility
    import pandas as pd
    commentsDf = pd.DataFrame({'Sentiment Score':postSentScore, 'Sentiment':postSent}, index=list(df_test['message']))
        

    trace0 = go.Bar(
        x= commentsDf['Sentiment'].value_counts().index.values.tolist(),
        y= commentsDf['Sentiment'].value_counts().values.tolist(),
        marker=dict(
            color=['rgba(44, 160, 44, 0.8)', 'rgba(31, 119, 180, 0.8)','rgba(222,45,38,0.8)']),
    )

    data = [trace0]
    layout = go.Layout(
        title='Facebook Sentiments',
        xaxis = dict(
            title = 'Polarity Categories'),
        yaxis = dict(
            title = 'Frequency')
    )

    fig = go.Figure(data=data, layout=layout)
    
    iplot(fig)

    #visualizing Customer Sentiment Scores - Delta Airlines
    trace0 = go.Histogram(
        x=commentsDf['Sentiment Score'].tolist(),
        autobinx=False,
        xbins=dict(
            start=-1.0,
            end=1.0,
            size=0.28
        )
    )

    data = [trace0]
    layout = go.Layout(
        title='Facebook Sentiment Scores',
        xaxis = dict(
            title = 'Sentiment Score'),
        yaxis = dict(
            title = 'Frequency')
    )

    fig = go.Figure(data=data, layout=layout)

    iplot(fig)
    
    
    #creating a corpus after pre-processing
    import re
    import re
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    from nltk.stem import WordNetLemmatizer
    corpus = []
    for i in range(0,len(commentsDf)):
        #removing numbers, punctuations and keeping only characters of the form a-z or A-Z
        tweet = re.sub('[^a-zA-Z]', ' ', commentsDf.index.values[i])
        #transforming every word to its lowercase
        tweet = tweet.lower()
        tweet = tweet.split()
        #Lemmatization
        lm = WordNetLemmatizer()
        #lemmatization and removing stop words
        tweet = [lm.lemmatize(word) for word in tweet if not word in set(stopwords.words('english'))]
        tweet = ' '.join(tweet)
        corpus.append(tweet)
        
    #creating the Bag of Words model
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features = 45)
    X = cv.fit_transform(corpus).toarray()
    
    #creating a dataframe for Term document matrix 
    tdm = pd.DataFrame(X, columns=cv.get_feature_names())
    tdm.index = commentsDf.index.values
    tdmTrans = tdm.T
    
    #Explarotary tools
    
    import numpy as np
    
    #inspecting frequent words 
    freq = np.ravel(X.sum(axis=0))
    freqdf = pd.DataFrame(freq, columns = ['Frequency'], index=tdm.columns.values)
    freqdf = freqdf.sort_values('Frequency', ascending=False)
    
    #visualizing frequent terms

    trace0 = go.Bar(
            x=freqdf['Frequency'],
            y=freqdf.index.values,
            orientation = 'h'
    )
    
    data = [trace0]
    
    layout = go.Layout(
        title='Frequently Occuring Words',
        xaxis = dict(
            title = 'Frequency')
    )
    

    fig = go.Figure(data=data, layout=layout)

    iplot(fig)
    
    #Normalizing the data
    import numpy as np
    from sklearn import preprocessing
    tdmTrans_scaled_X = preprocessing.scale(tdmTrans.ix[:,:])
    
    #hierarchical clustering
    from scipy.cluster.hierarchy import dendrogram, linkage
    
    # generate the linkage matrix
    Z_ward = linkage(tdmTrans_scaled_X, 'ward')
    
    #plot dendograms
    from scipy.cluster import hierarchy
    import matplotlib
    import matplotlib.pyplot as plt
    # calculate full dendrogram
    matplotlib.rcParams.update({'font.size': 22})
    plt.figure(figsize=(40, 20))
    plt.title('Word Groups Occuring Together')
    plt.ylabel('Height')
    def llf(id):
        return tdmTrans.index.values[id]
    dendrogram(
        Z_ward,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=22.,  # font size for the x axis labels
        leaf_label_func=llf
    )
    plt.show()
    
    
    #Enter your key, token and secret obtained after creating Twitter app (apps.twitter.com)
    consumer_key = '6OVyYwN57p8RnwhjvyFTD2y7m'
    consumer_secret = 'sS09pzdqnkHhYoBr3a6gVhaKvGbEo8oAeYgBHDyeISD0zbdI9H'

    access_token = '801984474540081152-JDD1IJPE6FoZsoFRIU5oqL9uaOYGIvJ'
    access_token_secret = '08a1j9kgswtaJBgHlP7ZByF6nQkUjoP3dX57mhvOMcptT'

    #OAuth authorization - pass consumer key, secret and set access token
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    #access 5000 tweets for ''
    api = tweepy.API(auth)
    max_tweets = 5000
    searched_tweets = []
    searched_tweets_created = []
    last_id = -1
    while len(searched_tweets) < max_tweets:
        count = max_tweets - len(searched_tweets)
        try:
            #avoiding repeated tweets by filtering out retweets
            new_tweets = api.search(q=keyword+' -filter:retweets', count=count, max_id=str(last_id - 1))
            if not new_tweets:
                break
            searched_tweets.extend(new_tweets)
            last_id = new_tweets[-1].id
        except tweepy.TweepError as e:
            # depending on TweepError.code, one may want to retry or wait
            # to keep things simple, we will give up on an error
            break
        
    #obtaining tweet text, polarity score and final sentiment for each tweet
    hiltonTweets = []
    hiltonSentScore = []
    hiltonSent = []
    hilton_tweet_created = []
    for tweet in searched_tweets:
        hiltonTweets.append(tweet.text)
        analysis = TextBlob(tweet.text)
        hiltonSentScore.append(analysis.sentiment[0])
        hilton_tweet_created.append(tweet.created_at)
        if (analysis.sentiment[0]>0):
            hiltonSent.append('Positive')
        elif (analysis.sentiment[0]<0):
            hiltonSent.append('Negative')
        else:
            hiltonSent.append('Neutral')
        
    #creating a dataframe for @ for easy accessibility
    import pandas as pd
    hiltonTweetsDf = pd.DataFrame({'Sentiment Score':hiltonSentScore, 'Sentiment':hiltonSent, 'Time':hilton_tweet_created}, index=hiltonTweets)

    # plotting customer sentiments for Airlines
    

    trace0 = go.Bar(
        x= hiltonTweetsDf['Sentiment'].value_counts().index.values.tolist(),
        y= hiltonTweetsDf['Sentiment'].value_counts().values.tolist(),
        marker=dict(
            color=['rgba(44, 160, 44, 0.8)', 'rgba(31, 119, 180, 0.8)','rgba(222,45,38,0.8)']),
    )

    data = [trace0]
    layout = go.Layout(
        title='Twitter Sentiments',
        xaxis = dict(
            title = 'Polarity Categories'),
        yaxis = dict(
            title = 'Frequency')
    )

    fig = go.Figure(data=data, layout=layout)
    printmd('**---------------------------------------------------------------- &nbsp; T &nbsp; W &nbsp; I &nbsp; T &nbsp; T &nbsp; E &nbsp; R &nbsp; &nbsp; &nbsp; A &nbsp; N &nbsp; A &nbsp; L &nbsp; Y &nbsp; T &nbsp; I &nbsp; C &nbsp; S &nbsp; ----------------------------------------------------------------**')
    iplot(fig)

    #visualizing Customer Sentiment Scores - Delta Airlines
    trace0 = go.Histogram(
        x=hiltonTweetsDf['Sentiment Score'].tolist(),
        autobinx=False,
        xbins=dict(
            start=-1.0,
            end=1.0,
            size=0.28
        )
    )

    data = [trace0]
    layout = go.Layout(
        title='Twitter Sentiment Scores',
        xaxis = dict(
            title = 'Sentiment Score'),
        yaxis = dict(
            title = 'Frequency')
    )

    fig = go.Figure(data=data, layout=layout)

    iplot(fig)
    
    
    #creating a corpus after pre-processing
    import re
    import re
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    from nltk.stem import WordNetLemmatizer
    corpus = []
    for i in range(0,len(hiltonTweetsDf)):
        #removing numbers, punctuations and keeping only characters of the form a-z or A-Z
        tweet = re.sub('[^a-zA-Z]', ' ', hiltonTweetsDf.index.values[i])
        #transforming every word to its lowercase
        tweet = tweet.lower()
        tweet = tweet.split()
        #Lemmatization
        lm = WordNetLemmatizer()
        #lemmatization and removing stop words
        tweet = [lm.lemmatize(word) for word in tweet if not word in set(stopwords.words('english'))]
        tweet = ' '.join(tweet)
        corpus.append(tweet)
        
    #creating the Bag of Words model
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features = 45)
    X = cv.fit_transform(corpus).toarray()
    
    #creating a dataframe for Term document matrix 
    tdm = pd.DataFrame(X, columns=cv.get_feature_names())
    tdm.index = hiltonTweetsDf.index.values
    tdmTrans = tdm.T
    
    #Explarotary tools
    
    import numpy as np
    
    #inspecting frequent words 
    freq = np.ravel(X.sum(axis=0))
    freqdf = pd.DataFrame(freq, columns = ['Frequency'], index=tdm.columns.values)
    freqdf = freqdf.sort_values('Frequency', ascending=False)
    
    #visualizing frequent terms

    trace0 = go.Bar(
            x=freqdf['Frequency'],
            y=freqdf.index.values,
            orientation = 'h'
    )
    
    data = [trace0]
    
    layout = go.Layout(
        title='Frequently Occuring Words',
        xaxis = dict(
            title = 'Frequency')
    )
    

    fig = go.Figure(data=data, layout=layout)

    iplot(fig)
    
    #Normalizing the data
    import numpy as np
    from sklearn import preprocessing
    tdmTrans_scaled_X = preprocessing.scale(tdmTrans.ix[:,:])
    
    #hierarchical clustering
    from scipy.cluster.hierarchy import dendrogram, linkage
    
    # generate the linkage matrix
    Z_ward = linkage(tdmTrans_scaled_X, 'ward')
    
    #plot dendograms
    from scipy.cluster import hierarchy
    import matplotlib
    import matplotlib.pyplot as plt
    # calculate full dendrogram
    matplotlib.rcParams.update({'font.size': 22})
    plt.figure(figsize=(40, 20))
    plt.title('Word Groups Occuring Together')
    plt.ylabel('Height')
    def llf(id):
        return tdmTrans.index.values[id]
    dendrogram(
        Z_ward,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=22.,  # font size for the x axis labels
        leaf_label_func=llf
    )
    plt.show()
    
    printmd('**---------------------------------------------------------------- &nbsp; R &nbsp; E &nbsp; D &nbsp; D &nbsp; I &nbsp; T &nbsp; &nbsp; &nbsp; A &nbsp; N &nbsp; A &nbsp; L &nbsp; Y &nbsp; T &nbsp; I &nbsp; C &nbsp; S &nbsp; ----------------------------------------------------------------**')
    
    #PREREQUISITES:
    #1. Reddit Account
    username='prawAPI'
    password='prawAPI'
    #2. Client ID and Client Secret
    client_id = 'rUmddn612aieqg'
    client_secret = 'jKX9Wwx7TrEMH9Kr9xpP8wu9S4k'
    #3. User Agent
    user_agent = 'windows:com.redditText.prawAPI:v1.0.4 (by /u/prawAPI)'
    
    #install praw: pip install praw
    import praw
    #Authorized Reddit Instance
    reddit = praw.Reddit(client_id=client_id,
                         client_secret=client_secret,
                         user_agent=user_agent,
                         username=username,
                         password=password)
    
    #Iterating all submissions (and corresponding comments) with keyword 'opioid'
    submissions = []
    import time
    iteration = 0
    for submission in reddit.subreddit('all').search(keyword, limit=5):
        submission.comments.replace_more(limit=None)
        time.sleep(1)
        #Flat tree structure - containing all comments (and nested replies) for all submissions
        submissions.append({"title": submission.title, "submission author": submission.author.name, "submission time":submission.created_utc , "comments": [{"body":comment.body, "comment author": comment.author.name, "comment time": comment.created_utc,"id":comment.name, "parent_id":comment.parent_id, "children":[]} if (comment.author != None) else {"body":comment.body, "comment author": "None", "comment time": comment.created_utc,"id":comment.name, "parent_id":comment.parent_id, "children":[]} for comment in submission.comments.list()]})
        
    commentText = []
    for submission in submissions:
        for comment in submission['comments']:
            commentText.append(comment['body'])
    
    commentSentScore = []
    commentSent = []
    for comment in commentText:
        commentAnalysis = TextBlob(comment)
        commentSentScore.append(commentAnalysis.sentiment[0])
        if (commentAnalysis.sentiment[0]>0):
            commentSent.append('Positive')
        elif (commentAnalysis.sentiment[0]<0):
            commentSent.append('Negative')
        else:
            commentSent.append('Neutral')
        
    #creating a dataframe for @ for easy accessibility
    import pandas as pd
    commentsDf = pd.DataFrame({'Sentiment Score':commentSentScore, 'Sentiment':commentSent}, index=commentText)
        

    trace0 = go.Bar(
        x= commentsDf['Sentiment'].value_counts().index.values.tolist(),
        y= commentsDf['Sentiment'].value_counts().values.tolist(),
        marker=dict(
            color=['rgba(44, 160, 44, 0.8)', 'rgba(31, 119, 180, 0.8)','rgba(222,45,38,0.8)']),
    )

    data = [trace0]
    layout = go.Layout(
        title='Reddit Sentiments',
        xaxis = dict(
            title = 'Polarity Categories'),
        yaxis = dict(
            title = 'Frequency')
    )

    fig = go.Figure(data=data, layout=layout)
    
    iplot(fig)

    #visualizing Customer Sentiment Scores - Delta Airlines
    trace0 = go.Histogram(
        x=commentsDf['Sentiment Score'].tolist(),
        autobinx=False,
        xbins=dict(
            start=-1.0,
            end=1.0,
            size=0.28
        )
    )

    data = [trace0]
    layout = go.Layout(
        title='Reddit Sentiment Scores',
        xaxis = dict(
            title = 'Sentiment Score'),
        yaxis = dict(
            title = 'Frequency')
    )

    fig = go.Figure(data=data, layout=layout)

    iplot(fig)
    
    
    #creating a corpus after pre-processing
    import re
    import re
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    from nltk.stem import WordNetLemmatizer
    corpus = []
    for i in range(0,len(commentsDf)):
        #removing numbers, punctuations and keeping only characters of the form a-z or A-Z
        tweet = re.sub('[^a-zA-Z]', ' ', commentsDf.index.values[i])
        #transforming every word to its lowercase
        tweet = tweet.lower()
        tweet = tweet.split()
        #Lemmatization
        lm = WordNetLemmatizer()
        #lemmatization and removing stop words
        tweet = [lm.lemmatize(word) for word in tweet if not word in set(stopwords.words('english'))]
        tweet = ' '.join(tweet)
        corpus.append(tweet)
        
    #creating the Bag of Words model
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features = 45)
    X = cv.fit_transform(corpus).toarray()
    
    #creating a dataframe for Term document matrix 
    tdm = pd.DataFrame(X, columns=cv.get_feature_names())
    tdm.index = commentsDf.index.values
    tdmTrans = tdm.T
    
    #Explarotary tools
    
    import numpy as np
    
    #inspecting frequent words 
    freq = np.ravel(X.sum(axis=0))
    freqdf = pd.DataFrame(freq, columns = ['Frequency'], index=tdm.columns.values)
    freqdf = freqdf.sort_values('Frequency', ascending=False)
    
    #visualizing frequent terms

    trace0 = go.Bar(
            x=freqdf['Frequency'],
            y=freqdf.index.values,
            orientation = 'h'
    )
    
    data = [trace0]
    
    layout = go.Layout(
        title='Frequently Occuring Words',
        xaxis = dict(
            title = 'Frequency')
    )
    

    fig = go.Figure(data=data, layout=layout)

    iplot(fig)
    
    #Normalizing the data
    import numpy as np
    from sklearn import preprocessing
    tdmTrans_scaled_X = preprocessing.scale(tdmTrans.ix[:,:])
    
    #hierarchical clustering
    from scipy.cluster.hierarchy import dendrogram, linkage
    
    # generate the linkage matrix
    Z_ward = linkage(tdmTrans_scaled_X, 'ward')
    
    #plot dendograms
    from scipy.cluster import hierarchy
    import matplotlib
    import matplotlib.pyplot as plt
    # calculate full dendrogram
    matplotlib.rcParams.update({'font.size': 22})
    plt.figure(figsize=(40, 20))
    plt.title('Word Groups Occuring Together')
    plt.ylabel('Height')
    def llf(id):
        return tdmTrans.index.values[id]
    dendrogram(
        Z_ward,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=22.,  # font size for the x axis labels
        leaf_label_func=llf
    )
    plt.show()


import csv
from html.parser import HTMLParser
import logging
import os
from pprint import pprint
import re
import time
from matplotlib import  pyplot as plt
from textblob import TextBlob
from tweepy import OAuthHandler, Stream
import tweepy 
from tweepy.streaming import StreamListener

import simplejson as json

logging.getLogger("TweetListener").setLevel(logging.DEBUG)


class TweetListener(StreamListener):
    """
    This class extend from StreamListener
    """

    def __init__(self, api=None):
        """
        Initialize data and setting constant 
        """
        super(StreamListener, self).__init__()
        # Number of tweets to collect
        self.num_tweets_collect = 5000
        # Tweets counter
        self.num_tweets_counter = 0
        # File name to store content
        self.store_data_file_csv = "travel_content.csv"
    
        self.emoji_pattern = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)
        self.html_parser = HTMLParser()
        self.url_re = re.compile(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))')
        # Reference https://github.com/nltk/nltk/blob/develop/nltk/tokenize/casual.py#L327
        self.twitter_handle_re = re.compile(r"(?<![A-Za-z0-9_!@#\$%&*])@(([A-Za-z0-9_]){20}(?!@))|(?<![A-Za-z0-9_!@#\$%&*])@(([A-Za-z0-9_]){1,19})(?![A-Za-z0-9_]*@)")

    def pre_precess(self, text):
        
        result_text = self.html_parser.unescape(text)
        result_text = result_text.encode('ascii', 'ignore').decode('utf-8')
        result_text = self.url_re.sub('', result_text)
        result_text = result_text.strip()
        result_text = result_text.replace('\r', '').replace('\n', '')

        logging.debug(result_text)
        return result_text
    
    def extract_feature(self, tweet):
        """
        Process input data (each tweet) by selecting only interest information.
        Result is array of object (csv row)
        """
        row = []
        row.append(tweet["created_at"])
        processed_text = self.pre_precess(tweet["text"])
        row.append(processed_text)
        
        analyzed = self.analyze_sentiment(processed_text)
        row.append(analyzed.sentiment[0])
        row.append(analyzed.sentiment[1])
       
        if not tweet["entities"] is None and len(tweet["entities"]["hashtags"]) > 0:
            hashtags = tweet["entities"]["hashtags"]
            text_hashtags = []
            for hashtag in hashtags:
                text_hashtags.append(hashtag["text"])
           
            str_hashtags = ", ".join(text_hashtags)
            row.append(str_hashtags)
            logging.debug(str_hashtags)
        else:
            row.append("")
        if not tweet["place"] is None:
            row.append(tweet["place"]["name"])
            row.append(tweet["place"]["place_type"])
            row.append(tweet["place"]["full_name"])
            row.append(tweet["place"]["country_code"])
            row.append(tweet["place"]["country"])
            
            coordinate1 = tweet["place"]["bounding_box"]["coordinates"][0][0]
            coordinate2 = tweet["place"]["bounding_box"]["coordinates"][0][1]
            coordinate3 = tweet["place"]["bounding_box"]["coordinates"][0][2]
            coordinate4 = tweet["place"]["bounding_box"]["coordinates"][0][3]
            avg_x = (coordinate1[0] + coordinate2[0] + coordinate3[0] + coordinate4[0]) / 4;
            avg_y = (coordinate1[1] + coordinate2[1] + coordinate3[1] + coordinate4[1]) / 4;
            row.append(avg_x)
            row.append(avg_y)
            row.append(coordinate1[0])
            row.append(coordinate2[0])
            row.append(coordinate3[0])
            row.append(coordinate4[0])
            row.append(coordinate1[1])
            row.append(coordinate2[1])
            row.append(coordinate3[1])
            row.append(coordinate4[1])
            
        else:
            row.append("")  # avg_x
            row.append("")  # avg_y
            row.append("")
            row.append("")
            row.append("")
            row.append("")
            row.append("")
            row.append("")
            row.append("")
            row.append("")
            row.append("")
            row.append("")
            row.append("")
            row.append("")
            row.append("")        
        return row
        
        # friends_count
    def analyze_sentiment(self, text): 
        analysis = TextBlob(text)
        pprint(analysis.sentiment)
        return analysis

    def on_data(self, data):
        """
        The on_data method of Tweepy’s StreamListener conveniently passes data 
        from statuses to the on_status method.
        
        Manage every coming tweet
        """
        try:
            # open file and append, put tweet at the end of the file
            with open(self.store_data_file_csv, 'a') as f:
                # write data to the file
                # keep text of json and print it
                tweet = json.loads(data)
                if not tweet["text"].startswith("RT "):
                    print('----- process tweet data -------')
                    row = self.extract_feature(tweet)
                
                    logging.debug(row)
                    
                    writer = csv.writer(f)
                    writer.writerow(row)
                    # We can modify this 
                    self.num_tweets_counter += 1
                    if self.num_tweets_counter < self.num_tweets_collect:
                        return True
                    else: 
                        return False
                
        except BaseException as e:
            print("Error on_data: ", str(e))
        return True
    
    def on_error(self, status):
        """
        Print error message
        """
        print("Error: ", status)
        return False
    
    def on_status(self, status):
        """
        Overwriting on status method 
        """
        print("-----", status, "-----")
        print("text \t", status.text)
        print("screen name \t", status.author.screen_name)
        print("create at \t", status.created_at)
        print("source \t", status.source)
        print("location = ", status.locations)
        # Filter out re-tweet
        print("-----", status, "-----")
        if not status.text[:3] == "RT ":
            print("-----", status, "-----")
            print("text \t", status.text)
            print("screen name \t", status.author.screen_name)
            print("create at \t", status.created_at)
            print("source \t", status.source)
            print("location = ", status.locations)
            print('-----------------------------------------')
    
    def on_timeout(self):
        print("Listener timeout")
        return true
    
    
def get_authenticated():
    consumer_key = 'QUDV4cMi6b4PVwNKe9REjem6A'
    consumer_secret = '0kjmAb0ve7zJ9CcCfEFMifCZknCOF5NsHI3ovS1AIq0QHeQ2LS'
    access_token = '1083734025100017664-jzVgFpfib28Nkv8NY1xMPnn2bagZqe'
    access_secret = '4Xa8CBVdQv9m3Io2TgA2os3Ho756S1RsS2pKf72PRcWBu'
    
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    # Construct the API instance
    api = tweepy.API(auth) 
    return auth, api


def read_data(file_content_name):
    content_data = []
    with open(file_content_name) as f:
        print(content)
        return content


auth, api = get_authenticated()
twitter_stream = Stream(auth, TweetListener())
# change words in track [] to any filter.
# It looks for the word here, not hashtag.

# Find location from http://boundingbox.klokantech.com/
# THAILAND_LOC = [97.34, 5.61, 105.64, 20.46]
# locations = THAILAND_LOC;

locale = 'en'
topics = ["travel"]
# twitter_stream.filter(track=topics, languages=languages, locations=locations) 
twitter_stream.filter(track=topics, languages=["en"])


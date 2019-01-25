from collections import Counter
import csv
import logging
import os
from os import path
from pprint import pprint
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from PIL import Image
import conda
import lda
from matplotlib import  pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import mpl_toolkits
import numpy as np
import pandas as pd
import simplejson as json

# For BaseMap library
conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ["PROJ_LIB"] = proj_lib
logging.getLogger("AnalyzeData").setLevel(logging.DEBUG)
from mpl_toolkits.basemap import Basemap


def read_data(file_content_name):
    content_data = []
    with open(file_content_name) as f:
        print(content)
        return content


def get_tokens(tweet_texts):
    tokenizer = TopicTokenizer()
    token_list = []
    for text in tweet_texts:
        if not text is np.nan: 
            list_tags = text.split(',')
            for tag in list_tags:
                token_list.append(tokenizer.tokenize(text))
    
    return token_list


def join_data(column):
    results = []
    for data in column:
        if not data is np.nan: 
            tokens = data.split(",");
            for t in tokens:
                results.append(t)
    
    return results


df = pd.read_csv("travel_content.csv",
                 names=["created_at", "text", "polarity", "subjectivity", "hashtags", "place_name",
                        "place_type", "place_fullname",
                        "place_country_code", "place_country",
                        "place_bounding_box_centroid_x", "place_bounding_box_centroid_y",
                        "place_bounding_box_coordinate1x", "place_bounding_box_coordinate1y",
                        "place_bounding_box_coordinate2x", "place_bounding_box_coordinate2y",
                        "place_bounding_box_coordinate3x", "place_bounding_box_coordinate3y",
                        "place_bounding_box_coordinate4x", "place_bounding_box_coordinate4y"
                        ])
# Passing tweet text to process token
hashtags_column = df["place_name"].values  # text column
all_hashtags = join_data(hashtags_column)

stopwords = [];
# https://www.flaticon.com/free-icon/world-map_290185#term=world%20map&page=3&position=19
image_name = "earth-globe.png"
mask = np.array(Image.open(image_name))
wordcloud = WordCloud(stopwords=stopwords,
                          background_color="white",
                          mode="RGBA",
                          max_words=1000, mask=mask).generate(",".join(all_hashtags))

# create coloring from image
image_colors = ImageColorGenerator(mask)
plt.figure(figsize=[7, 7])
plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")
# store to file
plt.savefig("wordcloud.png", format="png")
 
df_sentiment = df.drop(['created_at', 'text', 'place_type', "hashtags", "place_type", "place_fullname",
                        "place_country_code", "place_country"], axis=1)
df_sentiment = df_sentiment.dropna(subset=['place_name'])
sort_by_city = df_sentiment.sort_values("place_name")
print(sort_by_city.head(n=100))

# group by city and find mean
sum_data = df_sentiment.groupby(['place_name'])['polarity', 'subjectivity'].mean().reset_index()
print(sum_data)
fig = plt.figure()
ax = fig.add_subplot(111)

map = Basemap(projection='merc',
               llcrnrlat=-60,
               urcrnrlat=80,
               llcrnrlon=-180,
               urcrnrlon=180,
               lat_ts=20,
               resolution='c')

# 
# draw elements onto the world map
map.drawcountries()
map.drawcoastlines(antialiased=False,
                      linewidth=0.005)
cities = df_sentiment['place_name'].values
for city in cities:
    centroid_long = df_sentiment.loc[df_sentiment['place_name'] == city]['place_bounding_box_centroid_x'].values[0]
    centroid_lat = df_sentiment.loc[df_sentiment['place_name'] == city]['place_bounding_box_centroid_y'].values[0]
    mean_polarity = sum_data.loc[sum_data['place_name'] == city]['polarity'].values[0]
    color = ""
    if mean_polarity == 0:
        # neutral
        color = 'yo'
    elif mean_polarity > 0:
        # positive
        color = 'go'
    else:
        # negative
        color = 'ro'
    x, y = map(centroid_long, centroid_lat)  
    map.plot(x, y, color, markersize=4, alpha=0.5) 
plt.show()

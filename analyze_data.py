# coding: utf-8
# Install required libraries
# -m pip install tweepy
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

from _util import make_document_term_matrix
import mpl_toolkits
import numpy as np
import pandas as pd
import simplejson as json
from topic_tokenizer import TopicTokenizer

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


def get_vocabulary_helper(topic_numbers, number=5):
    vocab = np.array(list(vocabulary.keys()))
    topic_models = model.topic_word_
    result = []
    for topic_number in topic_numbers:
        words = vocab[np.argsort(topic_models[topic_number])][:-(number + 1):-1]
        result.append(words)
        
    return result


def print_word_contri(vocabulary):
    n_large_contri = 8
    vocab_array = np.array(list(vocabulary.keys()))
    for i, topic_dist in enumerate(model.topic_word_):
        # 8 largest contribution
        temp = np.argpartition(-topic_dist, n_large_contri)
        result = temp[:n_large_contri]
        word_result = []
        for r in result:
            word_result.append(vocab_array[r])
        
        print('Topic {}: {}'.format(i, " ".join(word_result)))


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

hashtags_column = df["hashtags"].values  # text column
all_hashtags = join_data(hashtags_column)
# token_list = get_tokens(tweet_texts)

wordcount = Counter(all_hashtags)
for word in wordcount:
    print(word)
    print(wordcount[word]);
    
print(all_hashtags)

# stopwords = set(STOPWORDS)
# stopwords.update(all_hashtags)
stopwords = [];
# https://www.flaticon.com/free-icon/world-map_290185#term=world%20map&page=3&position=19
# Generate a word cloud image
# image_name = "world-map.png"
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
plt.savefig("fra_wine.png", format="png")

# ## Analyze location and sentiment
# TODO group data by city and analyze sentiment

# drop NaN row at city 
# "place_bounding_box"
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
# map = Basemap(projection='merc', lat_0=50, lon_0=-100,
#                      resolution='l', area_thresh=5000.0,
#                      llcrnrlon=-140, llcrnrlat=-55,
#                      urcrnrlon=160, urcrnrlat=70)
# set resolution='h' for high quality
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
# shapefile = path.join('data', 'ne_10m_admin_0_countries')
# map.readshapefile(shapefile, 'countries', color='black', linewidth=.5)
# Draw the outer map
# map.drawmapboundary(color='#d3d3d3')
# read in the shapefile
# add coordinates as red dots

cities = df_sentiment['place_name'].values
for city in cities:
    centroid_long = df_sentiment.loc[df_sentiment['place_name'] == city]['place_bounding_box_centroid_x'].values[0]
    centroid_lat = df_sentiment.loc[df_sentiment['place_name'] == city]['place_bounding_box_centroid_y'].values[0]
    mean_polarity = sum_data.loc[sum_data['place_name'] == city]['polarity'].values[0]
    color = ""
    if mean_polarity == 0:
        color = 'bo'
    elif mean_polarity > 0:
        color = 'go'
    else:
        color = 'ro'
    x, y = map(centroid_long, centroid_lat)  
    map.plot(x, y, color, markersize=6)
# alpha
# neutral = plt.plot(z, "yo", markersize=15)
# positive = plt.plot(z, "go", markersize=15)
# negative = plt.plot(z, "ro", markersize=15)
# Put a white cross over some of the data.
# white_cross, = plt.plot(z[:5], "w+", markeredgewidth=3, markersize=15)

plt.legend([neutral, positive, negative], ["Positive", "Neutral", "Negative"])
plt.legend()   
plt.show()

patches = []
map.readshapefile(shapefile, 'countries', color='black', linewidth=.5)
for index, (info, shape) in enumerate(zip(map.countries_info,
                                          map.countries)):
    print(info)
    patches.append(Polygon(np.array(shape), True))
    ax.add_collection(PatchCollection(patches, facecolor='m', edgecolor='k', linewidths=1., zorder=2))
plt.show()
#      # `iso3` is a 3 letter country code
#      iso3 = info['ADM0_A3']
#      # don't map the antartic
#      if iso3 == 'ATA':
#          continue
#      # convert shape to numpy array for use with basemap
#      shape = np.array(shape)
#      # basemap/matplotlib specific data wrangling
#      polygon = Polygon(shape, True)
#      patches = [polygon]
#      # store the (iso_name, path) in cache. Will use `contains_points`
#      # method to later determine in which country tweets fall
#      cache_path.append((iso3, matplotlib.path.Path(shape)))
# 
#      # basemap/matplotlib specific data wrangling
#      patch_collection = PatchCollection(patches)
#      # store the (iso_name, patch_collection) to change the country
#      # color
#      cache_patches.append((iso3, patch_collection))
#      # Set default country facecolor to be gray.
#      patch_collection.set_facecolor('#d3d3d3')
#      # basemap/matplotlib specific data wrangling
#      map.add_collection(patch_collection)

print("break")
# # Convert token to sparse matrices
# document_matrix, vocabulary = make_document_term_matrix(token_list)
# 
# # Use LDA to find most frequency used words.
# model = lda.LDA(n_topics=10, n_iter=100, random_state=1)
# # model.fit(document_matrix)
# classified_data = model.fit_transform(document_matrix)
# 
# # Get Top topic
# # Most contributing probability
# classified_data = classified_data.argmax(1)
# count = Counter(classified_data)
# print(classified_data)
# print(count)
# 
# keys_count = np.array(count.most_common())
# # key is the key in vocab dictionary
# keys = keys_count[:, 0]
# # counting keys
# counts = keys_count[:, 1]
# print(keys, counts)
# 
# # plt.barh(np.arange(len(counts), 0, -1), counts)
# # plt.show()
# 
# word_list = get_vocabulary_helper(keys)
# 
# # fig = plt.figure()
# # axis = fig.add_subplot(111)
# # X, Y = fig.get_dpi() * fig.get_size_inches()
# # h = Y / (20)
# # 
# # for row, words in enumerate(word_list):
# #     y = Y - (row * h) - h
# # 
# #     axis.text(0.3, y, ' '.join(words), fontsize=(h * 0.8),
# #             horizontalalignment='left',
# #             verticalalignment='center')
# # 
# # axis.set_ylim(0, Y)
# # axis.set_axis_off()
# 
# fig = plt.figure()
# axis = fig.add_subplot(111)
# X, Y = fig.get_dpi() * fig.get_size_inches()
# num_print = 10
# h = Y / num_print
# 
# y_s = []
# for row, words in enumerate(word_list[:num_print]):
#     y = Y - (row * h) - h
#     y_s.append(y)
#     axis.text(1, y, ' '.join(words),
#               fontsize=(h * .2),
#               horizontalalignment='left',
#               verticalalignment='center',
#               color='black')
# y_s[-1] = 0
# # axis.set_ylim(-20, Y)
# # axis.set_xlin(0, )
# axis.barh(y_s, counts[:num_print], height=30, color='yellow')
# axis.get_yaxis().set_visible(False)
# plt.show()
# 
# print("end")

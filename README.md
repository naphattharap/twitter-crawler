# Twitter Crawler

## About this project
This project is to retrieve data from twitter and processing data by analyzing users’ sentiment.
The topic **Travel** is selected and there are 2 types of data visualization implemented: **Hot Hashtags** and **Users’ sentiment
related to location**.

Data has been retrieved from twitter with track "travel" with selected language (English). 
In order to store only interested information for visualization, feature extraction and pre-processing are performed.

1. Feature extraction: Only interested attributes are extracted from tweets and stored in file such as Tweet Date,
Hashtag, Tweet message, Place information, Coordinates from geographic location.

2. Pre-processing: Tweet messages contain unnecessary information such as links, new line etc. Therefore, the messages
have been processed by removing those information and sentimental analysis results (polarity, subjectivity)
are stored in the file at this step.

## Data 
About 5000 tweets were collected on 24 January 2019 for analyzing.

## Visualization
1. Hot mentioned cities. To visualize which cities are mentioned as its frequency, cities’ name are represented in
Word Cloud.
![alt text](https://github.com/naphattharap/twitter-crawler/blob/master/viz-hot-tags.png)

2. Sentimental Analysis is represented in world map to visualize how twitter’s users felt about the city that they have
mentioned in green, yellow and red which mean to positive, neutral and negative respectively.
![alt text](https://github.com/naphattharap/twitter-crawler/blob/master/viz-worldmap-sentiment.png)

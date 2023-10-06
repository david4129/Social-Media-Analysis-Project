#!/usr/bin/env python
# coding: utf-8


#!/usr/bin/env python
# coding: utf-8


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from wordcloud import WordCloud
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import seaborn as sns
import re
import math
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from scipy.stats import spearmanr
import dataframe_image as dfi
from PIL import Image

# Load pre-processed data
df = pd.read_csv('cleaned_df.csv')
df1 = pd.read_csv('features_added.csv')

#Display Title
with st.container():
    st.title('Social Media Data Analysis')
    st.write('Explore insights from your social media data.')

# Quick Exploratory Data Analysis
with st.container():
    st.header('Quick Exploratory Data Analysis')

    # Display data summary
    st.subheader('Data Summary')
    st.write('Dataset shape: {}'.format(df.shape))
    st.write(df.head())

# Display Basic Stats
with st.container():
    st.subheader('Basic Stats')
    st.write(df.describe())

#Distribution Of Posts Over Time
with st.container():
    st.subheader('Distribution Of Posts Over Time')
    st.image(https://github.com/david4129/Social-Media-Analysis-Project/blob/main/Images/eda1.JPG, caption='', use_column_width=True)

#Count Plot of Posts By Network
with st.container():
    st.subheader('Count Plot of Posts By Network')
    image1 = Image.open(r"C:\Users\DELL\Downloads\Social Media Analysis\Social-Media-Analysis-Project\Images\eda2.JPG")
    st.image(image1, caption='', use_column_width=True)

#Are There Any Trends Over Time?
with st.container():
    st.subheader('Count Plot of Posts By Network')
    image2 = Image.open(r"C:\Users\DELL\Downloads\Social Media Analysis\Social-Media-Analysis-Project\Images\eda3.JPG")
    st.image(image2, caption='', use_column_width=True)

    st.subheader('Count Plot of Posts By Network')
    image3 = Image.open(r"C:\Users\DELL\Downloads\Social Media Analysis\Social-Media-Analysis-Project\Images\eda3.1.JPG")
    st.image(image3, caption='', use_column_width=True)

#Is The Data Stationary Or Not
with st.container():
    st.subheader('Is The Data Stationary Or Not')
    st.write('To check if the data is stationary or not, we perform the Augmented Dickey - Fuller Test as seen below')
    st.write(adfuller(df['Engagements']))
    st.write("""From the Augmented Dickey - Fuller test, the p-value(1.5186408675097521e-28) obtained is less than 0.05 
    therefore the feature 'Engagements' is not stationary, this implies that the statistical properties do not 
    change over time.""")

# Analysis and Insights
with st.container():
    st.header('Analysis and Insights')
    st.subheader('Top Posts By Engagement Rate')
    image4 = Image.open(r"C:\Users\DELL\Downloads\Social Media Analysis\Social-Media-Analysis-Project\Images\Top Posts By Engagement Rates.JPG")
    st.image(image4, caption='', use_column_width=True)

with st.container():
    st.subheader('Which Social Media Network Performs Best?')
    image5 = Image.open(r"C:\Users\DELL\Downloads\Social Media Analysis\Social-Media-Analysis-Project\Images\Question2.JPG")
    st.image(image5, caption='', use_column_width=True)

# Create two columns for layout
with st.container():
    col1, col2 = st.columns(2)

    st.subheader('When Are Peak Engagement Times?')

    with col1:
        image6 = Image.open(r"C:\Users\DELL\Downloads\Social Media Analysis\Social-Media-Analysis-Project\Images\Question3.JPG")
        st.image(image6, caption='', use_column_width=True)

    with col2:
        image7 = Image.open(r"C:\Users\DELL\Downloads\Social Media Analysis\Social-Media-Analysis-Project\Images\Question3.2.JPG")
        st.image(image7, caption='', use_column_width=True)

with st.container():
    st.subheader('What Is the Relationship Between Impressions and Engagement?')
    image8 = Image.open(r"C:\Users\DELL\Downloads\Social Media Analysis\Social-Media-Analysis-Project\Images\Question4.JPG")
    st.image(image8, caption='', use_column_width=True)

with st.container():
    st.subheader('Which Content Types Receive the Most Shares?')
    image9 = Image.open(r"C:\Users\DELL\Downloads\Social Media Analysis\Social-Media-Analysis-Project\Images\Question5.JPG")
    st.image(image9, caption='', use_column_width=True)

with st.container():
    st.subheader('How Effective Are Hashtags in Driving Engagement?')
    image10 = Image.open(r"C:\Users\DELL\Downloads\Social Media Analysis\Social-Media-Analysis-Project\Images\Question6.JPG")
    st.image(image10, caption='', use_column_width=True)

with st.container():
    st.subheader('Is There a Seasonal or Periodic Pattern in Engagement?')
    image11 = Image.open(r"C:\Users\DELL\Downloads\Social Media Analysis\Social-Media-Analysis-Project\Images\Question7.3.JPG")
    st.image(image11, caption='', use_column_width=True)

with st.container():
    image12 = Image.open(r"C:\Users\DELL\Downloads\Social Media Analysis\Social-Media-Analysis-Project\Images\Question7.JPG")
    st.image(image12, caption='', use_column_width=True)

with st.container():
    image13 = Image.open(r"C:\Users\DELL\Downloads\Social Media Analysis\Social-Media-Analysis-Project\Images\Question7.2.JPG")
    st.image(image13, caption='', use_column_width=True)

with st.container():
    st.subheader('What Is the Overall Engagement Rate and How Does It Vary Across Platforms?')
    image14 = Image.open(r"C:\Users\DELL\Downloads\Social Media Analysis\Social-Media-Analysis-Project\Images\Question8.JPG")
    st.image(image14, caption='', use_column_width=True)

with st.container():
    st.subheader('How Effective Are Hashtags in Driving Engagement?')
    image15 = Image.open(r"C:\Users\DELL\Downloads\Social Media Analysis\Social-Media-Analysis-Project\Images\Question9.JPG")
    st.image(image15, caption='', use_column_width=True)



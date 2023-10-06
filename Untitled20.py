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
from pathlib import Path

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
    image_path = Path("C:\Users\DELL\Downloads\Social Media Analysis\Social-Media-Analysis-Project\Images\eda1.JPG")
    st.image(image_path, caption='', use_column_width=True)

#Count Plot of Posts By Network
with st.container():
    st.subheader('Count Plot of Posts By Network')
    image_path = Path("C:\Users\DELL\Downloads\Social Media Analysis\Social-Media-Analysis-Project\Images\eda2.JPG")
    st.image(Image.open(image_path), caption='', use_column_width=True)

#Are There Any Trends Over Time?
with st.container():
    st.subheader('Count Plot of Posts By Network')
    st.image(Image.open('https://raw.githubusercontent.com/david4129/Social-Media-Analysis-Project/main/Images/eda3.JPG?token=GHSAT0AAAAAACH6UOH3P6B7QGY2IY3HKCQUZJAG52A'), caption='', use_column_width=True)

    st.subheader('Count Plot of Posts By Network')
    st.image(Image.open('https://raw.githubusercontent.com/david4129/Social-Media-Analysis-Project/main/Images/eda3.1.JPG?token=GHSAT0AAAAAACH6UOH3BN235H3PQSQCWNKWZJAG6TA'), caption='', use_column_width=True)

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
    st.image(Image.open('https://raw.githubusercontent.com/david4129/Social-Media-Analysis-Project/main/Images/Top%20Posts%20By%20Engagement%20Rates.JPG?token=GHSAT0AAAAAACH6UOH36ZDDU6INU4ZJFCRUZJAHAVQ'), caption='', use_column_width=True)

with st.container():
    st.subheader('Which Social Media Network Performs Best?')
    st.image(Image.open('https://raw.githubusercontent.com/david4129/Social-Media-Analysis-Project/main/Images/Question2.JPG?token=GHSAT0AAAAAACH6UOH3ZWOHQ7MWQJIVIKUSZJAHBPA'), caption='', use_column_width=True)

# Create two columns for layout
with st.container():
    col1, col2 = st.columns(2)

    st.subheader('When Are Peak Engagement Times?')

    with col1:
        st.image(Image.open('https://raw.githubusercontent.com/david4129/Social-Media-Analysis-Project/main/Images/Question3.JPG?token=GHSAT0AAAAAACH6UOH2FFAXIEXBNCJTILAIZJAHDBQ'), caption='', use_column_width=True)

    with col2:
        st.image(Image.open('https://raw.githubusercontent.com/david4129/Social-Media-Analysis-Project/main/Images/Question3.2.JPG?token=GHSAT0AAAAAACH6UOH3ZXTIH7J3VY6Y2ZAAZJAHD2A'), caption='', use_column_width=True)

with st.container():
    st.subheader('What Is the Relationship Between Impressions and Engagement?')
    st.image(Image.open('https://raw.githubusercontent.com/david4129/Social-Media-Analysis-Project/main/Images/Question4.JPG?token=GHSAT0AAAAAACH6UOH3VGPKUF2YJ747YNLIZJAHFEA'), caption='', use_column_width=True)

with st.container():
    st.subheader('Which Content Types Receive the Most Shares?')
    st.image(Image.open('https://raw.githubusercontent.com/david4129/Social-Media-Analysis-Project/main/Images/Question5.JPG?token=GHSAT0AAAAAACH6UOH3CAFI2R7HFQTLGJEWZJAHGGA'), caption='', use_column_width=True)

with st.container():
    st.subheader('How Effective Are Hashtags in Driving Engagement?')
    st.image(Image.open('https://raw.githubusercontent.com/david4129/Social-Media-Analysis-Project/main/Images/Question6.JPG?token=GHSAT0AAAAAACH6UOH3UMCTULEDXNVC5LTCZJAHJHA'), caption='', use_column_width=True)

with st.container():
    st.subheader('Is There a Seasonal or Periodic Pattern in Engagement?')
    st.image(Image.open('https://raw.githubusercontent.com/david4129/Social-Media-Analysis-Project/main/Images/Question7.3.JPG?token=GHSAT0AAAAAACH6UOH2TY5JT3EN772MJQ64ZJAHH4A'), caption='', use_column_width=True)

with st.container():
    st.image(Image.open('https://raw.githubusercontent.com/david4129/Social-Media-Analysis-Project/main/Images/Question7.JPG?token=GHSAT0AAAAAACH6UOH2UGRYAVCA3SCDDIPKZJAHKGQ'), caption='', use_column_width=True)

with st.container():
    st.image(Image.open('https://raw.githubusercontent.com/david4129/Social-Media-Analysis-Project/main/Images/Question7.2.JPG?token=GHSAT0AAAAAACH6UOH3YMVRM67BB4Z5ZP6SZJAHK6A'), caption='', use_column_width=True)

with st.container():
    st.subheader('What Is the Overall Engagement Rate and How Does It Vary Across Platforms?')
    st.image(Image.open('https://raw.githubusercontent.com/david4129/Social-Media-Analysis-Project/main/Images/Question8.JPG?token=GHSAT0AAAAAACH6UOH2EDG57ZQPQ5JOO4ZAZJAHMFA'), caption='', use_column_width=True)

with st.container():
    st.subheader('How Effective Are Hashtags in Driving Engagement?')
    st.image(Image.open('https://raw.githubusercontent.com/david4129/Social-Media-Analysis-Project/main/Images/Question9.JPG?token=GHSAT0AAAAAACH6UOH2VNTA6VF633C4J36EZJAHMXA'), caption='', use_column_width=True)



# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 11:39:53 2024

@author: rosep
"""

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np

# Load the CSV file
def load_data(csv_file):
    """Loads data from a CSV file"""
    return pd.read_csv(csv_file)

# Clean and preprocess the text data
def clean_text(text):
    """Cleans and preprocesses the text data"""
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join the tokens back into a string
    text = ' '.join(tokens)
    
    return text

# Filter Kitwe-related news
def filter_kitwe_news(text):
    """Filters Kitwe-related news"""
    kitwe_keywords = ['Kitwe', 'Kitwe news', 'Zambia', 'Copperbelt']
    for keyword in kitwe_keywords:
        if keyword in text:
            return True
    return False

# Remove duplicate articles
def remove_duplicates(data):
    """Removes duplicate articles"""
    return data.drop_duplicates(subset='text')

# Extract news content from URLs (optional)
def extract_news_content(url):
    """Extracts news content from a URL"""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.get_text()

# Perform feature engineering
def feature_engineering(data):
    """Performs feature engineering"""
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_data = vectorizer.fit_transform(data['text'])
    
    # Dimensionality reduction using Truncated SVD
    svd = TruncatedSVD(n_components=100)
    svd_data = svd.fit_transform(tfidf_data)
    
    # Standard scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(svd_data)
    
    # Create a new DataFrame with the engineered features
    engineered_data = pd.DataFrame(scaled_data, index=data.index)
    
    return engineered_data

# Main data pipeline function
def data_pipeline(csv_file):
    """Runs the data pipeline"""
    # Load the data
    data = load_data(csv_file)
    
    # Clean and preprocess the text data
    data['text'] = data['text'].apply(clean_text)
    
    # Filter Kitwe-related news
    data = data[data['text'].apply(filter_kitwe_news)]
    
    # Remove duplicate articles
    data = remove_duplicates(data)
    
    # Extract news content from URLs (optional)
    if 'url' in data.columns:
        data['content'] = data['url'].apply(extract_news_content)
    
    # Perform feature engineering
    engineered_data = feature_engineering(data)
    
    return data, engineered_data

# Save the data
def save_data(data, engineered_data, output_file):
    """Saves the data"""
    data.to_csv(output_file + '_cleaned.csv', index=False)
    engineered_data.to_csv(output_file + '_engineered.csv', index=False)
    with open(output_file + '_vectorizer.pkl', 'wb') as f:
        pickle.dump(TfidfVectorizer(max_features=5000), f)
    with open(output_file + '_svd.pkl', 'wb') as f:
        pickle.dump(TruncatedSVD(n_components=100), f)
    with open(output_file + '_scaler.pkl', 'wb') as f:
        pickle.dump(StandardScaler(), f)

# Get the CSV file path from the user
csv_file = input(r"C:\Users\rosep\OneDrive\Pictures\Desktop\Ktwe Local News dataset\data\Final_dataset2.csv")

# Run the data pipeline
output_file = 'kitwe_news_data'
data, engineered_data = data_pipeline(csv_file)
save_data(data, engineered_data, output_file)

print("Data pipeline completed successfully!")
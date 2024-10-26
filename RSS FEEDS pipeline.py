# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 21:46:11 2024

@author: rosep
"""

import feedparser
import json
import logging
import os
import schedule
import time

# Define logging configuration
logging.basicConfig(filename='rss_aggregator.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define RSS feeds
rss_feeds = {
    'Daily Nations Zambia': 'https://dailynationzambia.com/search/kitwe/feed/rss2/',
    'Lusaka Star': 'https://lusakastar.com/search/kitwe/feed/rss2/',
    'Lusaka Voice': 'https://lusakavoice.com/search/kitwe/feed/rss2/',
    'Mwebantu': 'https://www.mwebantu.com/search/kitwe/feed/rss2/',
    'Zambia365': 'https://zambianews365.com/search/kitwe/feed/rss2/',
    'Zambia Eye': 'https://zambianeye.com/search/kitwe/feed/rss2/',
    'Zambia Reports': 'https://zambiareports.news/search/kitwe/feed/rss2/'
}

# Define data storage schema
data_schema = {
    'title': '',
    'link': '',
    'description': '',
    'published': ''
}

# Function to fetch and parse RSS feeds
def fetch_rss_feeds():
    aggregated_data = []
    for source, url in rss_feeds.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                data = {
                    'source': source,
                    'title': entry.title,
                    'link': entry.link,
                    'description': entry.description,
                    'published': entry.published
                }
                aggregated_data.append(data)
        except Exception as e:
            logging.error(f"Failed to fetch {source}: {str(e)}")
    return aggregated_data

# Function to store data locally
def store_data_locally(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

# Function to print aggregated data
def print_aggregated_data(data):
    print("Aggregated RSS Data:")
    for index, item in enumerate(data):
        print(f"\nItem {index+1}:")
        print(f"Source: {item['source']}")
        print(f"Title: {item['title']}")
        print(f"Link: {item['link']}")
        print(f"Description: {item['description']}")
        print(f"Published: {item['published']}")

# Schedule data aggregation
def schedule_aggregation(file_path):
    schedule.every(1).hours.do(lambda: store_data_locally(fetch_rss_feeds(), file_path))  # Run every hour

# Run scheduler
def run_scheduler(file_path):
    schedule_aggregation(file_path)
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    file_path = 'C:/Users/rosep/kitwe local news aggregator/aggregated_rss_data.json'  # Specify your local file path
    # Initial data aggregation, storage and printing
    aggregated_data = fetch_rss_feeds()
    store_data_locally(aggregated_data, file_path)
    print_aggregated_data(aggregated_data)

    # Start scheduler
    run_scheduler(file_path)
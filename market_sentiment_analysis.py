import requests
import tweepy
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import os

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_TOKEN")

# fetch news articles using News API
def fetch_news(query, api_key, pagesize=10):
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}&pageSize={pagesize}"
    response = requests.get(url)
    articles = response.json()['articles']
    news_df = pd.DataFrame([(article['title'], article['description']) for article in articles],
                           columns=['Title', 'Description'])
    return news_df


# ---Fetch recent tweets using Tweepy client ---
# Authenticate with Tweepy v2 Client
client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)


def fetch_tweets(query, max_results=10):
    tweets = client.search_recent_tweets(query=query,
                                         max_results=max_results,
                                         tweet_fields=['text', 'created_at'],
                                         expansions=None
                                         )
    tweet_data = [tweet.data for tweet in tweets.data] if tweets.data else []
    return pd.DataFrame(tweet_data, columns=['India'])


# -- Visualize sentiment distribution
def plot_sentiment(data, title):
    sentiment_counts = data['Sentiment'].value_counts()
    sentiment_counts.plot(kind='bar', title=title, color=['green', 'blue', 'red'])
    plt.title(title)
    plt.show()


def main():
    # Fetch news data
    news_data = fetch_news("Bhandup", NEWS_API_KEY)
    print("News Data:")
    print(news_data.head())

    # Fetch tweets
    # tweets_data = fetch_tweets("Apple", max_results=20)
    # print("\nTweets Data:")
    # print(tweets_data.head())

    # Initialize sentiment analysis
    sentiment_analysis = pipeline("sentiment-analysis")

    # Analyze news sentiment
    news_data['Sentiment'] = news_data['Title'].apply(lambda x: sentiment_analysis(x)[0]['label'])
    print("\nNews Sentiment Analysis:")
    print(news_data.head())

    for index, row in news_data.iterrows():
        print(f"Title: {row['Title']}")
        print(f"Description: {row['Description']}")
        print(f"Sentiment: {row['Sentiment']}")
        print("---")

    # Plot sentiment distribution
    plot_sentiment(news_data, "News Sentiment Distribution for Apple")


if __name__ == "__main__":
    main()

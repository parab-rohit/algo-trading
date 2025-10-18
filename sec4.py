from transformers import pipeline

#Load sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

headlines = [
    "Apple reports record quarterly revenue despite economic slowdown",
    "Tech stocks make new highs as investors anticipate rate cuts",
    "Market uncertainty grows amid geopolitical tensions"
]

# analyze the sentiments of the headlines
sentiments = sentiment_analyzer(headlines)

for headline, sentiment in zip(headlines, sentiments):
    print(f"Headline: {headline}")
    print(f"Sentiment: {sentiment['label']} (Score: {sentiment['score']:.2f})\n")
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

nltk.download('vader_lexicon')

# Define some example tweets
tweets = [
    "I love the new iPhone! 😍",
    "This is the worst service ever.",
    "The movie was okay, not great but not bad.",
    "I’m so happy with my results!",
    "I'm not sure how I feel about this..."
]

# Create DataFrame
df = pd.DataFrame(tweets, columns=['tweet'])

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Classify each tweet
def classify_sentiment(text):
    score = sia.polarity_scores(text)['compound']
    if score > 0.05:
        return 'Positive'
    elif score < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment'] = df['tweet'].apply(classify_sentiment)

# Print results
print(df)

# Plot results
df['sentiment'].value_counts().plot(kind='bar', color=['green', 'red', 'gray'])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Tweets")
plt.show()

import pandas as pd 
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Load dataset with text reviews
df = pd.read_csv('reviews.csv')
print("Original dataset:")
print(df.head())


# Remove special characters and links
df['clean_text'] = df['text'].apply(lambda x: re.sub(r'http\S+|[^a-zA-Z\s]', '', x))
print("\nAfter removing special characters and links:")
print(df[['text', 'clean_text']].head())


# Convert to lowercase and remove stopwords
stop_words = set(stopwords.words('english'))
df['clean_text'] = df['clean_text'].apply(lambda x: ' '.join([word.lower() for word in word_tokenize(x) if word.lower() not in stop_words]))
print("\nAfter converting to lowercase and removing stopwords:")
print(df['clean_text'].head())


# Lemmatization
lemmatizer = WordNetLemmatizer()
df['clean_text'] = df['clean_text'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
print("\nAfter lemmatization:")
print(df['clean_text'].head())
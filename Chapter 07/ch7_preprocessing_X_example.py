import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


# Download required resources (if not already installed)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Sample tweet
tweet = "I loved the product, but delivery was disappointing!"

# Tokenize the sentence into words
tokens = word_tokenize(tweet)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Define stopwords set
stop_words = set(stopwords.words('english'))

# Lemmatize, lowercase, remove non-alphanumeric, and optionally remove stopwords
cleaned_tokens = [
    lemmatizer.lemmatize(word.lower())
    for word in tokens
    if word.isalnum() and word.lower() not in stop_words
]

print(cleaned_tokens)  # Output: ['example', 'sentence', '123']

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample dataset
emails = [
    "Win a free vacation! Reply now to claim your prize.",
    "Meeting scheduled for 3 PM tomorrow.",
    "Congratulations! You’ve won a gift card worth $500.",
    "Please review the attached report and provide feedback.",
    "Earn extra income working from home. Apply now."
]
labels = [1, 0, 1, 0, 1]  # 1 = Spam, 0 = Legitimate

# Text to numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Train a Naive Bayes classifier
nb_model = MultinomialNB()
nb_model.fit(X, labels)

# Test new emails
new_emails = [
    "Free lottery tickets available. Don't miss out!",
    "Team meeting rescheduled to 4 PM today."
]
X_new = vectorizer.transform(new_emails)
predictions = nb_model.predict(X_new)
print("Predictions:", predictions)  # 1 = Spam, 0 = Legitimate

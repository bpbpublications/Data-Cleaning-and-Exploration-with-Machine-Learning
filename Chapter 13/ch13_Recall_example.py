from sklearn.model_selection import cross_val_score
from sklearn.metrics import recall_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Sample dataset
df = pd.read_csv("diabetes_data.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Model and recall scorer
model = RandomForestClassifier()
recall_scorer = make_scorer(recall_score)

# Cross-validation
scores = cross_val_score(model, X, y, scoring=recall_scorer, cv=5)
print(f"Average Recall across folds: {scores.mean():.2f}")


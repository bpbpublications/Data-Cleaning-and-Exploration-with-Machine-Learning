from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
# Toy dataset
X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
y = [0, 1, 0, 1, 0]

# Set up 3-fold cross-validation
kf = KFold(n_splits=3, shuffle=True, random_state=42)
model = LogisticRegression()

# Perform CV and collect accuracy scores
scores = cross_val_score(model, X, y, cv=kf)
print(f"Cross-validation scores: {scores.round(3)}")

# Initialize Stratified K-Fold with 3 splits
skf = StratifiedKFold(n_splits=3)
model = RandomForestClassifier(n_estimators=100)

# Perform cross-validation
scores = cross_val_score(model, X, y, cv=skf)

print(f"Stratified Cross-Validation Scores: {scores}")

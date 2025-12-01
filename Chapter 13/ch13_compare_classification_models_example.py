from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from yellowbrick.classifier import ROCAUC
import pandas as pd
from sklearn.model_selection import train_test_split

#Data preparation
data = pd.read_csv('diabetes_data.csv')
X = data.drop('Outcome', axis=1)
y = data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Models to compare
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier()
}

# Evaluate using cross-validation
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=5)
    print(f"{name} Accuracy: {scores.mean():.2f}")


# Define model
model = RandomForestClassifier()

# Create visualizer
visualizer = ROCAUC(model, classes=['Class 0', 'Class 1'])
visualizer.fit(X_train, y_train)  # Fit the model
visualizer.score(X_test, y_test)  # Score the model
visualizer.show()  # Display the ROC curve

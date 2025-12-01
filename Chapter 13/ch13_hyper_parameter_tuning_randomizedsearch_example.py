from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import numpy as np
import pandas as pd



data = pd.read_csv('diabetes_data.csv')
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = data[feature_cols]
y = data['Outcome']

# Split dataset into train-test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
# Define the parameter distributions
param_distributions = {
    'n_estimators': [int(x) for x in np.linspace(50, 200, num=4)],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the model and RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=50,  # Number of random combinations to try
    cv=5,
    scoring='accuracy',
    random_state=42,
    verbose=1
)

# Fit the random search to the data
random_search.fit(X_train, y_train)

# Display the best parameters and score
print(f"Best Parameters: {random_search.best_params_}")
print(f"Best Cross-Validation Score: {random_search.best_score_}")


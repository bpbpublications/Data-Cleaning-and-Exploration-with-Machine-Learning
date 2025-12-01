from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from yellowbrick.classifier import ConfusionMatrix
from hyperopt import fmin, tpe, hp, Trials

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Define model
model = RandomForestClassifier(random_state=42)

# Perform cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy Scores: {scores}")
print(f"Mean Accuracy: {scores.mean():.2f}")



# Load dataset
data = load_iris()
X, y = data.data, data.target

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Visualize confusion matrix
cm = ConfusionMatrix(model, classes=data.target_names)
cm.score(X, y)
cm.show()


# Define search space
search_space = {
    'n_estimators': hp.choice('n_estimators', range(50, 201)),
    'max_depth': hp.choice('max_depth', range(5, 21)),
    'min_samples_split': hp.uniform('min_samples_split', 0.1, 1.0)
}

# Define objective function
def objective(params):
    model = RandomForestClassifier(**params, random_state=42)
    score = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
    return -score

# Run optimization
trials = Trials()
best_params = fmin(fn=objective, space=search_space, algo=tpe.suggest, max_evals=50, trials=trials)
print(f"Best Parameters: {best_params}")


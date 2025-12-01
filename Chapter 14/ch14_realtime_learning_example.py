import numpy as np
from sklearn.linear_model import SGDClassifier

# Simulated feedback data
features = np.array([[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]])
labels = np.array([1, 0, 1])  # 1: Positive feedback, 0: Negative feedback

# Train an online learning model
model = SGDClassifier()
model.partial_fit(features, labels, classes=np.array([0, 1]))

# Update model with new feedback
new_features = np.array([[0.6, 0.4]])
new_labels = np.array([1])
model.partial_fit(new_features, new_labels)

print("Initial predictions:", model.predict(features))
print("New predictions:", model.predict(new_features))

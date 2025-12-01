from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import pandas as pd#added this

#added this
df = pd.read_csv('ecommerce_data.csv')

x = df.drop(columns=['purchased'])

y = df['purchased']

#added till here

# Initialize the model and RFE selector
model = LogisticRegression()
selector = RFE(model, n_features_to_select=2)
selector.fit(x, y)

# Print boolean mask of selected features
print("Selected features mask:", selector.support_)

# Print selected feature names
selected_features = x.columns[selector.support_]
print("Selected features:", list(selected_features))


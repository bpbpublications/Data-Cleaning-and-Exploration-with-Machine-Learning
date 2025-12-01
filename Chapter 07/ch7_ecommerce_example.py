from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split #added this
from sklearn.metrics import classification_report #added this
import pandas as pd #added this


#added this
# Load dataset
df = pd.read_csv('ecommerce_data.csv')  # Make sure this file exists and is in the correct format
print("Dataset loaded successfully.")
print(df.head())

# Separate features and target
X = df[['age', 'income', 'gender', 'region']]  # feature columns
y = df['purchased']  # target column (assumes binary classification: 0 or 1)

# Split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split into training and testing sets.")
#added till here

# Preprocessing for numerical columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical columns
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine into a column transformer
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, ['age', 'income']),
    ('cat', categorical_transformer, ['gender', 'region'])
])

# Full pipeline: preprocessing + classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Fit the pipeline
pipeline.fit(x_train, y_train)

# Make predictions
y_pred = pipeline.predict(x_test)


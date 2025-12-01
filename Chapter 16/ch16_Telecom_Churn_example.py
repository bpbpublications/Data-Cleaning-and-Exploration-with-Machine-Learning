import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv('telecom_churn.csv')

# Handle missing values
data.fillna(data.median(), inplace=True)

# Standardize numerical variables
scaler = StandardScaler()
data[['Tenure', 'MonthlyCharges']] = scaler.fit_transform(data[['Tenure', 'MonthlyCharges']])

# Encode categorical variables
data = pd.get_dummies(data, columns=['ContractType'], drop_first=True)


X = data.drop('Churn', axis=1)
y = data['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

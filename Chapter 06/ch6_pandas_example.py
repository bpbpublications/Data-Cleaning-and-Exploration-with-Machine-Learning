import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Sample dataset
data = {'Age': np.random.randint(18, 80, 1000), 'Income': np.random.randint(20000, 120000, 1000)}
df = pd.DataFrame(data)

# Descriptive statistics
print(df.describe())


# Generate random age data
ages = np.random.randint(18, 80, 1000)

# Plot histogram
plt.hist(ages, bins=15, edgecolor='black', color='skyblue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# Generate random data
ad_spend = np.random.randint(1000, 5000, 50)
sales = ad_spend * 0.4 + np.random.randint(0, 500, 50)

# Plot scatter plot
plt.scatter(ad_spend, sales, color='green')
plt.title('Advertising Spend vs. Sales')
plt.xlabel('Advertising Spend')
plt.ylabel('Sales')
plt.grid(True)
plt.show()


# Generate random salary data
departments = ['HR', 'IT', 'Finance', 'Marketing']
salaries = [
    np.random.randint(30000, 60000, 50),
    np.random.randint(50000, 100000, 50),
    np.random.randint(40000, 80000, 50),
    np.random.randint(35000, 70000, 50)
]

# Plot box plot
plt.boxplot(salaries, labels=departments, patch_artist=True, 
            boxprops=dict(facecolor='lightblue'))
plt.title('Salary Distribution by Department')
plt.xlabel('Department')
plt.ylabel('Salary')
plt.show()


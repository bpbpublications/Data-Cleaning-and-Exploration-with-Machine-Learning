import pandas as pd

# Example: Monthly sales in USD
sales = pd.Series([4500, 5200, 3900, 6100, 7200])
print("Sales:",sales)

# Example: Employee salary data
data = {
    'Employee': ['Alice', 'Bob', 'Charlie'],
    'Monthly Salary ($)': [5000, 6200, 5800]
}

df = pd.DataFrame(data)

# Inspect the first few rows
print("First few rows:",df.head())

#Select rows that meet certain conditions
df_filtered = df[df['Monthly Salary ($)'] > 5500]
print("Filtered rows:",df_filtered)

#Sorting
df_sorted = df.sort_values(by='Monthly Salary ($)')
print("Sorting:",df_sorted)

#Aggregating
average_salary = df['Monthly Salary ($)'].mean()
print("Average salary:",average_salary)

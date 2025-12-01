import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import skew, kurtosis

study_hours = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

scores = [55, 67, 70, 70, 72, 75, 80, 85, 90, 95]

# Range Calculation
range_value = max(scores) - min(scores)

# Variance and Standard Deviation Calculation
variance = np.var(scores)
std_dev = np.std(scores)


data_skewness = skew(scores)
data_kurtosis = kurtosis(scores)

print(f"Range: {range_value}, Variance: {variance}, Standard Deviation: {std_dev}")

print(f"Skewness: {data_skewness}, Kurtosis: {data_kurtosis}")

# charts- Histogram
plt.hist(scores, bins=5, edgecolor='black')
plt.title('Histogram of Exam Scores')
plt.xlabel('Scores')
plt.ylabel('Frequency')
plt.show()

# charts- Density


sns.kdeplot(scores, shade=True)
plt.title('Density Plot of Exam Scores')
plt.xlabel('Scores')
plt.ylabel('Density')
plt.show()

# charts- Box

plt.boxplot(scores, vert=False)
plt.title('Box Plot of Exam Scores')
plt.xlabel('Scores')
plt.show()


# charts- Scatter Plot

plt.scatter(study_hours, scores)
plt.title('Study Hours vs. Exam Scores')
plt.xlabel('Study Hours')
plt.ylabel('Exam Scores')
plt.show()

# charts- Heatmap

data = {'Scores': scores, 'Study_Hours': study_hours}
df = pd.DataFrame(data)

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


#outliers
z_scores = stats.zscore(scores)
outliers = np.where(np.abs(z_scores) > 3)

print(f"Outliers at positions: {outliers}")

Q1 = np.percentile(scores, 25)
Q3 = np.percentile(scores, 75)
IQR = Q3 - Q1

outliers = [x for x in scores if x < Q1 - 1.5 * IQR or x > Q3 + 1.5 * IQR]
print(f"Outliers detected using IQR method: {outliers}")

clean_scores = [x for x in scores if Q1 - 1.5 * IQR <= x <= Q3 + 1.5 * IQR]
print(f"Cleaned dataset: {clean_scores}")



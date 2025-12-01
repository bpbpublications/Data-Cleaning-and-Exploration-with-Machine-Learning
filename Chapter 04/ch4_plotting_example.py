import matplotlib.pyplot as plt
import numpy as np  # Ensure NumPy is imported for array handling

# Data
x = np.array([0, 1, 2, 3])
y = x ** 2  # Quadratic function

# Best practice: use fig and ax
fig, ax = plt.subplots()
ax.plot(x, y, label='y = x²')

# Add labels, title, and legend
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_title('Line Plot')
ax.legend()

# Adjust layout and display
plt.tight_layout()
plt.show()

#Bar Plot

categories = ['A', 'B', 'C']
values = [1, 4, 2]

fig, ax = plt.subplots()
ax.bar(categories, values)

ax.set_xlabel('Category')
ax.set_ylabel('Value')
ax.set_title('Bar Plot')

plt.tight_layout()
plt.bar()
plt.show()

#Scatter Plot

x = np.random.rand(50)
y = np.random.rand(50)

fig, ax = plt.subplots()
ax.scatter(x, y)

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_title('Scatter Plot')

plt.tight_layout()
plt.show()

#adding titles and labels

plt.plot(x, y)
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Customized Line Plot')
plt.grid(True)
plt.show()

#adding colors & Line Styles

fig, ax = plt.subplots()
ax.plot(x, y, color='red', linestyle='--', marker='o', label='Sample Data')

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_title('Customized Line Plot')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()

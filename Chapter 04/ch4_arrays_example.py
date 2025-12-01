import numpy as np
array = np.array([1, 2, 3, 4, 5])
print("Array:",array)

#Using built-in functions
zeros_array = np.zeros(5)       # Array of zeros
ones_array = np.ones(5)         # Array of ones
random_array = np.random.rand(5) # Random numbers between 0 and 1

print("Zeros Array:", zeros_array)
print("Ones Array:", ones_array)
print("Random Array:", random_array)

#Creating number sequences efficiently

# Using np.arange(): start, stop, step
range_array = np.arange(0, 10, 2)
print("Range Array:",range_array)

# Using np.linspace(): evenly spaced numbers
linear_array = np.linspace(0, 1, 5)
print("Linear Array:",linear_array)

#indexing & slicing

element = array[2]
print("Element:",element)

sub_array = array[1:4]
print("Sub Array:",sub_array)

#Addition and subtraction:
array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])
result = array1 + array2
print("Addition:",result)

#Multiplication and division:
result = array1 * array2
print("Multiplication",result)



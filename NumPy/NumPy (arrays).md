[[Array]] in Numpy have some restrictions:
- all elements of the array must be of the same data type
- the total size of array can't be changed once created
- the shape must be rectangular, not jagged
  e.g., each row of the 2-dimensional array must have the same number of columns

```
import numpy as np

# Creating a one-dimensional array
a = np.array([1, 2, 3, 4, 5, 6])
print(a)

# Accessing an element of the array
print(a[0])

# Slicing the array
print(a[:3])
```

```
import numpy as np

# Creating multi-dimensional array
c = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(c)

c.shape # (3, 4) <-- 3 rows and 4 columns

c.size # 12 <-- number of elements in the array
c.dtype # dtype('int32') <-- data type of the elements of array
```

Creating a basic array:

```
np.zeros(2) # array([0., 0.]) <-- array with 2 elements
np.ones([2, 2]) # array ([[1., 1.],
						  [1., 1.]])
np.empty(2) # array with 2 elements with random numbers
np.arange(4) # array([0, 1, 2, 3]) <-- array with 4 elements starting from 0
np.linspace(0, 10, num = 5) # array([0., 2.5, 5., 7.5, 10]) <-- array spaced linearly, with 5 elements as num = 5, starting from 0 to 10

x = np.ones(2, dtype = np.int64)
print(x) # array([1, 1]) 
```

Adding, removing, and sorting elements

```
arr = np.array([2, 1, 5, 3, 7, 4, 6, 8])
np.sort(arr)

a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])
np.concatenate((a, b))

x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6]])
np.concatenate((x, y), axis = 0)
# array([[1, 2],
		 [3, 4],
		 [5, 6]])
```

Reshaping an array:

```
a = np.arange(6)
print(a) # [1, 2, 3, 4, 5]

b = a.reshape(3, 2)
print(b)
# [[0, 1],
   [2, 3],
   [4, 5]]
```

Transpose of the matrix:

```
arr = np.arange(6).reshape((2,3))
print(arr)
# ([[0, 1, 2],
    [3, 4, 5]])

arr.transpose()
# ([[0, 3], 
    [1, 4],
    [2, 5]])
arr.T # works same as arr.transpose()
```

Reverse an array:

```
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

reversed_arr = np.flip(arr)
print(reversed_arr)
# [8, 7, 6, 5, 4, 3, 2, 1]
```
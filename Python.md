- Interpreted high-level object-oriented dynamically-typed scripting language
- Run time errors are usually encountered

==**Python Workings:**==

High level Python code --> Interpreter --> Machine Language
(Interpreter is responsible for translating high-level Python language to low-level machine language)

- Python Virtual Machine is created where the packages (libraries) are installed -- think of virtual machines as a container
- Python code is written in .py file
- CPython compiles the Python code to bytecode -- this bytecode is for the Python Virtual Machine

*In short, we  write Python code in one OS, copy it to another OS and simply run it*

==**Variables -- Object Types and Scopes:**==

Python supports numbers, strings, sets, lists, tuples, and dictionaries. These are the standard data types. 

Assign variable a value (use the equal sign) :
```
myFirstVariable = 1
mySecondVariable = 2
myFirstVariable = "Hello You"
```

*Data types are dynamically typed in Python*

If we want to assign the same value to more than one variable:
```
myFirstVariable = mySecondVariable = 1
```

Integers, decimals, floats:
```
value = 1 # integer
value = 1.2 # float with a floating point
valueLong = 9999999999L
```

Strings:
```
name = 'farhad'
name = "farhad"
name = """farhad"""

# Strings are immutable -- updating it will fail
a = 'me'
a[1] = 'y' # <-- this will give an error
```

```
# new object is created and 1 is stored there
# new pointer is created
# pointer connects a to 1
a = 1 

# new object is not created
# new pointer is created
# the pointer connects b to 1
b = a
```

Variables can have local or global scope:
```
# Local scope
# Variables declared within a function can only exist within a block
# Once the block exists, the variable also becomes inaccessible
def some_function():
	TestMode = False

print(TestMode) <-- Breaks as the variable doesn't exist outside the function

# If-Else and For-While loop doesn't create any local scope
for i in range(1, 11):
	test_scope = "variable inside for loop"

print(test_scope) # Output: variable inside for loop

# Global scope
# Variables can be accessed from any function have a global scope -- they exist in the __main__ frame
# We can declare a global variable outside of function
# Assign "global" keyword if you want to assign a global variable a new value
TestMode = True
def some_function():
	global TestMode
	TestMode = False

some_Function()
print(TestMode) # Output: False
# Removing "global TestMode" will only set the variable to False within the some_function() function
```

Operations:
```
1 // 3 # returns 0
1 / 3 # returns 0.333

# Exponentiation
2 ** 3 = 2 * 2 * 2 = 8

# Remainder
7 % 2 = 1

# Divmod in-built method -- returns divider and modulus
print(divmod(10, 3)) # prints 3 and 1 as 3 * 3 = 9 + 1 = 10

# Concat strings
'A' + 'B' = 'AB'

# Slicing
y = 'abc'
print(y[:-1]) # returns ab

# Finding index
name = 'farhad'
index = name.find('r') # returns 2
index = name.find('a', 2) # returns 4 <-- finds index of second a

# For Regex, use:
split() <-- splits a string into a list via regex
sub() <-- replaces matched string via regex
subn() <-- replaces matched string via regex and returns number of replacements

# Casting
str(x) <-- to string
int(x) <-- to integer
float(x) <-- to float

# Set
set = {9, 1, -1, 5, 2, 8, 3, 8}
print(set) # Output: {1, 2, 3, 5, 8, 9, -1} <-- duplicates are removed
# Set doesn't support indexing, slicing like lists
# Some set operations:
# set.add(item) <-- adds item to the set
# set.remove(item) <-- removes item from the set and raises error if it is not present
# set.discard(item) <-- removes item from set if it is present
# set.pop() <-- returns any item from the set, raises KeyError if the set is empty
# set.clear() <-- clears the set

# Intersect Sets
a = {1, 2, 3}
b = {3, 4, 5}
c = a.intersection(b)

# Difference in Sets
a = {1, 2, 3}
b = {3, 4, 5}
c = a.difference(b)

# Union of collections
a = {1, 2, 3}
b = {3, 4, 5}
c = a.union(b)

# Ternary Operator
# [if True] if [Expression] Else [if False]
Received = True if x == 'Yes' else False
```

**==Functions:==**

```
def my_new_function():
	print("this is my new function")

my_new_function()

# Function with arguments
def my_new_function(my_value):
	print("this is my new function with " + my_value)

# We can pass optional value
def my_new_function(my_value = "hello"):
	print(my_value)

my_new_function() <-- prints hello
my_new_function('test') <-- prints test

# *arguments -- if a function can take any number of arguments then add a * in front of the parameter name
def myfunc(*arguments):
	return a

myfunc(a)
myfunc(a, b)
myfunc(a, b, c)

# **arguments -- allows to pass a varying number of keyword arguments to a function

def test(*args, **kargs):
	print(args) # Prints all positional argument as a tuple
	print(kargs) # Prints all keyword argument as a dictionary
	print(args[0]) # Prints all first positional argument
	print(kargs.get('a')) # Accesses the value of key 'a' from the keyword arguments
alpha = 'alpha'
beta = 'beta'
test(alpha, beta, a = 1, b = 2)
# Positional arguments are alpha and beta
# Keyword arguments are a = 1, b = 2

# Final output:
('alpha', 'beta')
{'a': 1, 'b': 2}
alpha
1

# Positional argument
def greet(name, age):
	print(f"Hello {name}, you are {age} years old")
greet("Sai", 25)
"Sai" is assigned to name -- 1st position
25 is assigned to age -- 2nd position

# If we change the order, the meaning changes

# Keyword argument
def greet(name, age):
	print(f"Hello {name}, you are {age} years old")

greet(age = 25, name = "Sai")
# Output: Hello Sai, you are 25 years old
```

```
# Return
def my_function(input):
	return input + 2

# Lambda -- single expression anonymous function
# variable = lambda arguments: expression
my_lambda = lambda x, y, z: x - 100 + y - z
my_lambda(100, 100, 100) # returns 0
```

**==Modules==**

PYTHONPATH -- environment variable indicates where the Python interpreter needs to navigate to locate the module
PYTHONHOME -- alternative module search path

```
import my_module # To import everything in a module

# if the module contains a function or object named my_object then you will have to:
print(my_module.my_object)

from my_module import my_object
from my_module import * # * imports all objects
```

**==Conditions==**

```
if a == b:
	print 'a is b'
elif a < b:
	print 'a is less than b'
elif a > b:
	print 'a is greater than b'
else:
	print 'a is different'
```

**==Loops==**

```
while(input < 0): 
	do_something(input)
	input = input - 1

for i in range(0, 10)

for letter in 'hello':
	print(letter)
```

**==Recursion==**

function calling itself is called recursion

```
# Factorial function
def factorial(n):
	if n == 0:
		return 1
	else:
		return n * factorial(n-1)
```

**==Collections==**

Lists:
- List of data structures that can hold a sequence of values of any data types -- they are mutable (updateable)
- list are indexed by integers
- to create a list -- use square brackets

```
my_list = ['A', 'B']
my_list.append('C')
my_list[1] = 'D'
my_list.pop(1)
my_list.extend(another_list) # adds second list at the end
```

Tuples:
- faster than lists
- similar to lists -- objects can be of any type, can store a sequence of objects
- are immutable (non-updatable)

```
my_tuple = tuple()
or
my_tuple = 'f', 'm'
or
my_tuple = ('f', 'm')
```

Dictionaries:
- stores key / value pair objects

```
my_dictionary = dict()
my_dictionary['my_key'] = 1
my_dictionary['another_key'] = 2

or

my_dictionary = {'my_key' : 1, 'another_key' : 2}

for key in dictionary:
	print key, dictionary[key]
```

**==Object-Oriented Design -- Classes==**

- User defined types are known as classes -- classes have custom properties / attributes and functions
- supports encapsulation -- instance functions and variables

```
class MyClass:
	def MyClassFunction(self): # self <-- reference to the object
		return 5

m = MyClass()
returned_value = m.MyClassFunction()

# __init__ <-- function present in all classes -- executed when we require to instantiate object of a class
# __init__ can take any properties which we want to set

class MyClass:
	def __init__(self, first_property):
		self.first_property = first_property

	def MyClassFunction(self):
		return self.first_property

m = MyClass(123)
r = m.MyClassFunction() # 123
 
# Understanding classes
def __init__(self, first_property):
	self.first_property = first_property
__init__ <-- special method called constructor -- automatically executed when you create a new object from the class
self <-- refers to current object being created
first_property <-- input argument when we create the object
self.first_property = first_property <-- stores the argument as a property or attribute of the object

def MyClassFunction(self):
	return self.fisrt_property
this is a method -- function inside a class
self <-- allows function to access the object's properties
returns the value stored in self.first_property
```

```
another class example:
class StudentName:
	def __init__(self, name, address):
		self.name = name
		self.address = address
	def studentNameFunction(self):
		return self.name, self.address

classA = StudentName('Andy Andrews', 'Sunnyvile, California')
r = classA.studentNameFunction()
print(r) # ('Andy Andrews', 'Sunnyvile, California')
```